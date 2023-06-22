from repaint.repaint import repaint, CustomScheduler, Model
import torch
import numpy as np
import torchaudio
from audio_diffusion.models import DiffusionAttnUnet1D
from train_uncond import sample, get_crash_schedule, get_alphas_sigmas, DiffusionUncond


class DDModel(Model):
    def __init__(self, model: DiffusionAttnUnet1D):
        self.model = model

    def __call__(self, x: torch.Tensor, t: int):
        """
        :param x: input data
        :param t: timestep
        :return: predicted noise at the current timestep, to be subtracted
        """
        ts = x.new_ones([x.shape[0]]) * t
        v = self.model(x, ts).float()
        return v

def generate_long_song(model: DiffusionAttnUnet1D,
                       overlap_factor:float=0.5, sample_rate:int=44100//2,
                       segment_length:float=2**17,
                       steps:int=300,
                       duration:float=200, seed:int=0):

    # generate first segment:
    t = torch.linspace(1, 0, steps + 1)[:-1]
    t = get_crash_schedule(t)
    _, sigmas = get_alphas_sigmas(t)

    timesteps = torch.tensor(list(range(steps)))
    custom_scheduler = CustomScheduler(timesteps=timesteps, betas=sigmas)

    noise = torch.randn([1, 2, segment_length]).to("cuda")
    first_seg, t = sample(model, noise, steps, 0)
    initial = first_seg.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save("first_chunk.wav", initial[0], sample_rate)

    dmodel = DDModel(model)

    keep_mask = torch.ones_like(first_seg)
    keep_mask[:, :, int(segment_length - segment_length*overlap_factor):] = 0  # zero out part
    first_repaint = repaint(first_seg, keep_mask, dmodel, custom_scheduler)
    audio = first_repaint.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save(f'half_repainted.wav', audio[0], sample_rate)

from train_uncond import Config

base_model = DiffusionUncond.load_from_checkpoint("outputs/rainforest-22kHz-441000.ckpt",
                                                  global_args=Config())
base_model = base_model.diffusion.to("cuda")

generate_long_song(base_model)


