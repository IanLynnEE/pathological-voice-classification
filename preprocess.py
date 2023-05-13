import os

import librosa
import numpy as np
import pandas as pd


def read_files(df: pd.DataFrame, audio_dir: str, fs: int, frame_length: int,
               drop_cols: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read all files and slice each audio file into same duration frames.
    Labels are considered to be the same for all frames from the same file.
    WARNING:
        Users are responsible for choosing a suitable `frame_length`.

    Args:
        df (pd.DataFrame): data to process
        audio_dir (str): directory where audio files are located
        fs (int): sampling rate
        frame_length (int): number of points per frame.

    Returns:
        x (np.ndarray): audio data in the same length
        c (np.ndarray): clinical data corresponding to the audio data
        y (np.ndarray): labels that corresponding to features above
        ids (np.ndarray): ID of the corresponding data
    """
    # Get the duration of each file. Contents will not be loaded to memory.
    n_frames = np.zeros(df.shape[0], dtype=np.int_)
    for idx, ID in enumerate(df.ID):
        dur = librosa.get_duration(path=os.path.join(audio_dir, ID + '.wav'))
        n_frames[idx] = dur * fs / frame_length

    # Now we can better initialize arrays sizes.
    y = np.zeros(n_frames.sum(), dtype=np.int_)
    x = np.zeros((n_frames.sum(), frame_length), dtype=np.float_)

    # Test data will not have answers.
    if 'Disease category' not in df.columns:
        df['Disease category'] = 0

    # Load the contents of the audio file and slice into frames.
    frame_counter = 0
    for idx, ID in enumerate(df.ID):
        label = df.iloc[idx]['Disease category']
        audio, _ = librosa.load(os.path.join(audio_dir, ID + '.wav'), sr=fs)
        for j in range(n_frames[idx]):
            x[frame_counter] = audio[j * frame_length: (j + 1) * frame_length]
            y[frame_counter] = label
            frame_counter += 1

    # Make the clinical data matches the audio data.
    c = np.repeat(df.drop(columns=drop_cols).fillna(0).to_numpy(), n_frames, axis=0)

    # Retain IDs so that majority vote can be applied to prediction.
    ids = np.repeat(df.ID.to_numpy(), n_frames, axis=0)
    return x, c, y, ids
