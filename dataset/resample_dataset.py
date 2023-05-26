import os
import torchaudio
from torchaudio import transforms as T
from tqdm import tqdm

ROOT_DIR = "/media/sinclair/datasets/lofi/train_splits"
TARGET_SR = 44100//2
NEW_DIR = "/media/sinclair/datasets/lofi-22k/train_splits"

def resample_and_copy(source:str, target:str)->None:
    """
    resamples a dataset, copying it, so that resampling only has to be done once
    """
    resampler_transforms = {}  # dictionary for different resampling transforms, created if required
    files = os.listdir(source)
    for file in tqdm(files):
        audio, sr = torchaudio.load(os.path.join(source, file))
        if sr not in resampler_transforms.keys():
            resampler_transforms[sr] = T.Resample(sr, TARGET_SR)
        audio = resampler_transforms[sr](audio)
        torchaudio.save(os.path.join(target, file), audio, TARGET_SR)

if __name__ == "__main__":
    # make the new directory if it doesn't exist
    os.system(f"mkdir -p {NEW_DIR}")
    resample_and_copy(ROOT_DIR, NEW_DIR)

