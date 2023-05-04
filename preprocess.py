import os

import librosa
import numpy as np
import pandas as pd


def test():
    csv_path = 'Data/Train/data_list.csv'
    read_audio_files(csv_path, 22050, 11025)


def read_audio_files(csv_path: str, fs: int,
                     win_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Read all audio files and slice into same duration frames.
    Labels are considered to be the same for all frames from the same file.
    WARNING:
        Users are responsible for choosing a suitable `win_length`.

    Args:
        csv_path (str): csv file should contain information about audio files
        fs (int): sampling rate
        win_length (int): number of points per frame.

    Returns:
        x (np.ndarray): in the shape of (n_total_frames, win_length)
        y (np.ndarray): in the shape of (n_total_frames)
    """
    sound_dir = os.path.join(os.path.dirname(csv_path), 'raw')
    df = pd.read_csv(csv_path)

    # Get the duration of each file. Contents will not be loaded to memory.
    n_frames = np.zeros(df.shape[0], dtype=np.int_)
    for idx, ID in enumerate(df.ID):
        dur = librosa.get_duration(path=os.path.join(sound_dir, ID + '.wav'))
        n_frames[idx] = dur * fs / win_length

    # Now we can better initialize arrays sizes.
    y = np.zeros(n_frames.sum(), dtype=np.int_)
    x = np.zeros((n_frames.sum(), win_length), dtype=np.float_)

    # Load the contents of the audio file and slice into frames.
    frame_counter = 0
    for idx, ID in enumerate(df.ID):
        label = df.iloc[idx]['Disease category']
        audio, _ = librosa.load(os.path.join(sound_dir, ID + '.wav'), sr=fs)
        for j in range(n_frames[idx]):
            x[frame_counter] = audio[j * win_length: (j + 1) * win_length]
            y[frame_counter] = label
            frame_counter += 1
    return x, y


if __name__ == '__main__':
    test()
