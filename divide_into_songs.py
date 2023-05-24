from pydub import AudioSegment, silence
from tqdm import tqdm
import os

SONG_LENGTH = 10 # seconds

ROOT_DIR = "/media/sinclair/datasets/rainforest"

trainset_files = ["rainforest-0-97-100.wav"]

def export_songs(origin_files, split_dir: str) -> None:
    """
    takes large song files and divides it into smaller "songs"
    """
    song_counter = 0
    for file in origin_files:
        print(f"splitting up the file {file}")
        myaudio = AudioSegment.from_wav(os.path.join(ROOT_DIR, file))

        len_in_ms = len(myaudio)
        print(f"loaded a file that was {len_in_ms}ms long, {len_in_ms/(60 * 1000)} mins long, {len_in_ms/(60 * 60 * 1000)} hours long.")

        len_in_s = len_in_ms / 1000
        num_songs = int(len_in_s / SONG_LENGTH)  # a floor
        print(f"cutting audio into non-silent segments greater than {SONG_LENGTH} seconds.")
        print(f"expecting {num_songs} songs.")
        for x in range(num_songs):
            song = myaudio[x * SONG_LENGTH * 1000: (x+1) * SONG_LENGTH *1000]
            song.export(os.path.join(ROOT_DIR, split_dir, f"song_{song_counter}.wav"), format="wav")
            song_counter += 1
        print(f"exported {song_counter} songs.")

export_songs(trainset_files, "train_splits")

