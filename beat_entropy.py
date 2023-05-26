"""
computes beat entropy, a measure of the regularity of the beat
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa, librosa.display

def beat_entropy(y, sr, bins=500, min_bpm: int = 1, max_bpm: int = 100, tightness: float =0.1):
    """
    computes beat entropy, a measure of the regularity of the beat
    """
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time', tightness=tightness)
    beat_diffs = np.diff(beats)

    beat_distribution, _ = np.histogram(beat_diffs, bins=bins, \
                                     range=(0, 2), density=False)  # 400 bpm vs 1 bpm
    beat_distribution = beat_distribution / np.sum(beat_distribution)
    be = scipy.stats.entropy(beat_distribution)
    return be

if __name__ == "__main__":
    y, sr = librosa.load("/media/sinclair/datasets/lofi/mrbrightside.wav")
    breakpoint()
    # y, sr = librosa.load("demo_00458001.wav")
    for tightness in [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 500, 1000, 10000]:
        be, = beat_entropy(y, sr, tightness=tightness)
        print(f"with {tightness} beat entropy is {be}.")
    # plt.figure(figsize=(14, 5))
    # librosa.display.waveshow(y, alpha=0.6)
    # plt.vlines(beats, -1, 1, color='r')
    # plt.ylim(-1, 1)
    # plt.show()

