import os
import torch
import torchaudio
from torchaudio import transforms as T
from tqdm import tqdm
import librosa
from matplotlib import pyplot as plt

ROOT_DIR = "/media/sinclair/datasets/lofi/train_splits"
TARGET_SR = 44100//2
NEW_DIR = "/media/sinclair/datasets/lofi-22k-70bpm/train_splits"

def resample_and_copy(source:str, target:str, target_bpm: int=70)->None:
    """
    resamples a dataset, copying it, so that resampling only has to be done once
    """
    resampler_transforms = {}  # dictionary for different resampling transforms, created if required
    files = os.listdir(source)
    for file in tqdm(files):
        y, sr = librosa.load(os.path.join(source, file), mono=False)
        tempo, beats = librosa.beat.beat_track(y=y[0], sr=sr, units='time')
        if tempo > 100:
            # very little lofi is above 100 bpm, so this is probably a mistake
            tempo /= 2
        if tempo < 45:
            tempo *= 2

        factor = target_bpm/tempo  # for some reason, librosa returns twice the tempo
        y_stretch = librosa.effects.time_stretch(y, rate=factor)
        if sr not in resampler_transforms.keys():
            resampler_transforms[sr] = T.Resample(sr, TARGET_SR)
        audio = resampler_transforms[sr](torch.Tensor(y_stretch))
        torchaudio.save(os.path.join(target, file), audio, TARGET_SR)

def show_tempo_histogram(source:str)->None:
    """
    shows a histogram of the tempos of the dataset
    """
    files = os.listdir(source)
    tempos = []
    lengths = []
    print("Tabulating tempos over the dataset...")
    for file in tqdm(files):
        y, sr = librosa.load(os.path.join(source, file))
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
        secs = librosa.get_duration(y=y, sr=sr)
        tempos.append(tempo)
        lengths.append(secs)

    plt.title("Histogram of tempos in dataset")
    plt.hist(tempos, bins=100)
    plt.show()
    plt.title("Histogram of song lengths in the dataset.")
    plt.hist(lengths, bins=100)
    plt.show()

if __name__ == "__main__":
    # make the new directory if it doesn't exist
    # show_tempo_histogram(ROOT_DIR)
    os.system(f"mkdir -p {NEW_DIR}")
    resample_and_copy(ROOT_DIR, NEW_DIR)

