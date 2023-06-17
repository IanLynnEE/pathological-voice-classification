import argparse
from itertools import repeat
import multiprocessing
import os

import librosa
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from vta import vta_paper, vta_huang


def read_files(df: pd.DataFrame, audio_dir: str, fs: int, frame_length: int, drop_cols: list,
               binary_task: bool = False) -> tuple[np.ndarray | list, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read all files and slice each audio file into same duration frames.
    Labels are considered to be the same for all frames from the same file.
    WARNING:
        Users are responsible for choosing a suitable `frame_length`.

    Args:
        df (pd.DataFrame): data to process
        audio_dir (str): directory where audio files are located
        fs (int): sampling rate
        frame_length (int): number of points per frame
        binary_task (bool): binary classification or not

    Returns:
        x (np.ndarray | list): audio data
        c (np.ndarray): clinical data corresponding to the audio data
        y (np.ndarray): labels that corresponding to features above
        ids (np.ndarray): ID of the corresponding data
    """
    # Test data will not have answers.
    if 'Disease category' not in df.columns:
        df['Disease category'] = 0

    # Do not do any augmentation if frame_length is 0.
    if frame_length == 0:
        x = []
        for idx, ID in enumerate(df.ID):
            audio, _ = librosa.load(os.path.join(audio_dir, ID + '.wav'), sr=fs)
            x.append(audio)
        y = df['Disease category'].to_numpy()
        y = np.where(y == 5, 0, 1) if binary_task else y
        return x, df.drop(columns=drop_cols).fillna(0).to_numpy(), y, df.ID.to_numpy()

    # Get the duration of each file. Contents will not be loaded to memory.
    n_frames = np.zeros(df.shape[0], dtype=np.int_)
    for idx, ID in enumerate(df.ID):
        dur = librosa.get_duration(path=os.path.join(audio_dir, ID + '.wav'))
        n_frames[idx] = dur * fs / frame_length

    # Now we can better initialize arrays sizes.
    y = np.zeros(n_frames.sum(), dtype=np.int_)
    x = np.zeros((n_frames.sum(), frame_length), dtype=np.float_)

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

    # Binary classification.
    if binary_task:
        y = np.where(y == 5, 0, 1)
    return x, c, y, ids


def get_audio_features(audio: np.ndarray | list, args: argparse.Namespace) -> np.ndarray:
    """Generate audio features by MFCC or VTA, depending on the arguments.

    Args:
        audio (np.ndarray | list): raw audio data points (N, H_in)
        args (argparse.Namespace): arguments

    Returns:
        np.ndarray: extracted features (N, H_out, Frames)
    """
    n_samples = len(audio) if isinstance(audio, list) else audio.shape[0]
    if args.feature_extraction == 'mfcc':
        if isinstance(audio, list):
            raise NotImplementedError('Augmentation is required for MFCC.')
        try:
            # Load from cache if possible. MFCC is time-consuming.
            x = np.load(f'.cache/mfcc_{args.n_mfcc}_{n_samples}.npy', allow_pickle=True)
        except FileNotFoundError:
            mfcc = librosa.feature.mfcc(y=audio[0], sr=args.fs, n_mfcc=args.n_mfcc)
            x = np.zeros((n_samples, args.n_mfcc, mfcc.shape[1]))
            for i, row in tqdm(enumerate(audio), total=n_samples, postfix='MFCC'):
                x[i] = librosa.feature.mfcc(y=row, sr=args.fs, n_mfcc=args.n_mfcc)
            np.save(f'.cache/mfcc_{args.n_mfcc}_{n_samples}', x)
    elif args.feature_extraction == 'vta':
        if isinstance(audio, list):
            # Samples are in different lengths.
            # To get the same dimension output, the window length for each sample should be different.
            window_length = np.zeros(n_samples, dtype=np.int_)
            for i, sample in enumerate(audio):
                window_length[i] = np.ceil(len(sample) / args.n_frames).astype(np.int_)
            print(np.unique(window_length))
        else:
            window_length = np.ones(n_samples, dtype=np.int_) * args.vta_window_length
        # Use multiprocessing to speed up.
        zip_inputs = zip(audio, repeat(args.n_tube), window_length)
        with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pool:
            x = pool.starmap(vta_paper, tqdm(zip_inputs, total=n_samples, postfix='VTA'))
        # x is a list of (n_tubes, n_frames) arrays. To make the output agree with MFCC, transpose is needed.
        x = np.transpose(np.dstack(x), (2, 0, 1))   # (N, tubes, frames)
    elif args.feature_extraction == 'raw':
        x = audio
    else:
        x = np.empty((n_samples, 0, 0))
    return x


def get_1d_data(x: np.ndarray, axis: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=axis)
    var = np.var(x, axis=axis)
    skew = stats.skew(x, axis=axis)
    kurt = stats.kurtosis(x, axis=axis)
    diff = abs(np.diff(x, axis=axis)).sum(axis=axis)
    return mean, var, skew, kurt, diff, x.reshape(x.shape[0], -1)
