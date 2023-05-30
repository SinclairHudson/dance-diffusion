#!/usr/bin/env python3

from dataclasses import dataclass, asdict
from prefigure.prefigure import get_all_args, push_wandb_config
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path

import sys
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from tqdm import trange
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from einops import rearrange
import torchaudio
import wandb

from dataset.dataset import SampleDataset
from beat_entropy import beat_entropy

from audio_diffusion.models import DiffusionAttnUnet1D
from audio_diffusion.utils import ema_update
from viz.viz import audio_spectrogram_image, noise_schedule_plot


# Define the noise schedule and sampling loop
def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2

@torch.no_grad()
def sample(model, x, steps: int, eta):
    """Draws samples from a model given starting noise."""
    print(20 * "-" + "creating a sample" + 20 * "-")
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]

    t = get_crash_schedule(t)

    alphas, sigmas = get_alphas_sigmas(t)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.cuda.amp.autocast():
            v = model(x, ts * t[i]).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # If we are on the last timestep, output the denoised image
    return pred, t



class DiffusionUncond(pl.LightningModule):
    def __init__(self, global_args):
        super().__init__()

        self.diffusion = DiffusionAttnUnet1D(global_args, io_channels=2, n_attn_layers=4)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=global_args.seed)
        self.ema_decay = global_args.ema_decay
        self.loss_func = {
            'L1': F.l1_loss,
            'L2': F.mse_loss,
        }[global_args.loss_func]
        self.lr = global_args.lr

    def configure_optimizers(self):
        return optim.Adam([*self.diffusion.parameters()], lr=self.lr)

    def training_step(self, batch, batch_idx):
        reals = batch[0]

        # Draw uniformly distributed continuous timesteps
        t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)

        t = get_crash_schedule(t)

        # Calculate the noise schedule parameters for those timesteps
        alphas, sigmas = get_alphas_sigmas(t)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        with torch.cuda.amp.autocast():
            v = self.diffusion(noised_reals, t)
            mse_loss = F.mse_loss(v, targets)
            loss = self.loss_func(v, targets)

        log_dict = {
            'train/loss': loss.detach(),
            'train/mse_loss': mse_loss.detach(),
        }

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        ema_update(self.diffusion, self.diffusion_ema, decay)

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}', file=sys.stderr)


class DemoCallback(pl.Callback):
    def __init__(self, global_args):
        super().__init__()
        self.demo_every = global_args.demo_every
        self.num_demos = global_args.num_demos
        self.demo_samples = global_args.sample_size
        self.demo_steps = global_args.demo_steps
        self.sample_rate = global_args.sample_rate
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    #def on_train_epoch_end(self, trainer, module):
    def on_train_batch_end(self, trainer, module, one, two, three):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return
        if trainer.global_step <= 1:
            return  # don't demo in the first step, it'll be garbage
        self.last_demo_step = trainer.global_step
        noise = torch.randn([self.num_demos, 2, self.demo_samples]).to(module.device)

        try:
            fakes_batch, t = sample(module.diffusion_ema, noise, self.demo_steps, 0)

            # Put the demos together
            fakes = rearrange(fakes_batch, 'b d n -> d (b n)')
            fakes_batch = fakes_batch.clamp(-1, 1).cpu()

            beat_entropies = [beat_entropy(fake_clip[0], self.sample_rate) for fake_clip in fakes_batch.numpy()]
            avg_be = sum(beat_entropies)/len(beat_entropies)

            log_dict = {}

            filename = f'demo_{trainer.global_step:08}.wav'
            fakes = fakes.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filename, fakes, self.sample_rate)


            log_dict[f'demo'] = wandb.Audio(filename,
                                                sample_rate=self.sample_rate,
                                                caption=f'Demo')

            log_dict[f'demo_melspec_left'] = wandb.Image(audio_spectrogram_image(fakes))
            log_dict[f'noise_schedule'] = wandb.Image(noise_schedule_plot(t))
            log_dict[f'mean_beat_entropy'] = avg_be

            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        except Exception as e:
            print(f'{type(e).__name__}: {e}', file=sys.stderr)

def main(args):
    """
    Trains the model
    """
    save_path = None if args.save_path == "" else args.save_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print(f"Model will output samples of length {args.length_in_sec:.03f} seconds.")
    torch.manual_seed(args.seed)

    train_set = SampleDataset([args.training_dir], args)
    train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, persistent_workers=True, pin_memory=True)
    wandb_logger = pl.loggers.WandbLogger(project=args.name, log_model='all' if args.save_wandb=='all' else None)

    exc_callback = ExceptionCallback()
    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every,
                                                 # filename='{name}-{sample_rate:.2f}k-{length_in_sec:.1f}sec-{step}',
                                                 save_top_k=-1, dirpath=save_path)
    demo_callback = DemoCallback(args)

    diffusion_model = DiffusionUncond(args)

    wandb_logger.watch(diffusion_model)
    wandb.config.update(asdict(args))

    diffusion_trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        # num_nodes = args.num_nodes,
        # strategy='ddp',
        precision=16,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
    )

    diffusion_trainer.fit(diffusion_model, train_dl, ckpt_path=args.ckpt_path)

@dataclass
class Config():
    # NOTE: the type hints are required here to register these entries as fields (making them saveable to wandb)
    data: str="rainforest"
    name: str=f"{data}-dd"
    ckpt_path:str = "outputs/lofi-22kHz-459000.ckpt"
    training_dir:str = f"/media/sinclair/datasets/{data}-22k/train_splits"
    output_dir:str = "/home/sinclair/Documents/dance-diffusion/outputs"
    save_path:str="/home/sinclair/Documents/dance-diffusion/outputs"
    # model parameters
    sample_rate:int = 44100//2 # rate (Hz) at which the audio is sampled at. 
    # NOTE: resampling is extremely slow (expensive). It's better to have the dataset at the sample rate you want
    # sample rate should not be changed if you're going from pretrained, since it affects the scale of features in the waveform
    sample_size:int = 2 ** 17
    # length in seconds is sample_rate/sample_size
    length_in_sec: float = sample_size/sample_rate
    # training hyperparams
    random_crop:bool=True # crop audio at a random point
    checkpoint_every:int=3000 # steps
    num_workers:int=2
    batch_size: int = 2
    accum_batches: int = 2
    seed: int = 1337
    num_gpus: int = 1
    cache_training_data:bool=True
    save_wandb: str ="all" # all or none
    # demos, saved files to be listened to
    num_demos: int = 4 # number of samples outputted upon a demo
    demo_every: int = 1000 # steps
    demo_steps: int = 300 # number of denoising steps to run
    ema_decay: float = 0.9 # exponential moving average decay rate
    loss_func: str = "L2" # L1 or L2
    latent_dim: int = 0
    lr: float = 4e-5

    # augmentation
    augmentation_max_pitch_shift: int = 2

@dataclass
class DebugConfig(Config):
    # modifications on the original config for debugging
    num_demos: int = 2
    demo_every: int = 50
    demo_steps: int = 20
    load_frac: float = 0.10  # load only a fraction of the dataset
    # checkpoint_every:int=10 # steps


if __name__ == '__main__':
    args = Config()
    main(args)
