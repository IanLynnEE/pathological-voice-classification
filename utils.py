import argparse
import sys

from itertools import repeat
import multiprocessing
from collections import Counter

import librosa
import numpy as np
import pandas as pd
from scipy import stats
import torch
from tqdm import tqdm

from vta import vta_paper


def get_audio_features(audio: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Generate audio features by MFCC or VTA, depending on the arguments.

    Args:
        audio (np.ndarray): raw audio data points (N, H_in)
        args (argparse.Namespace): arguments

    Returns:
        np.ndarray: extracted features (N, H_out, Frames)
    """
    if args.feature_extraction == 'mfcc':
        try:
            x = np.load(f'cache/mfcc_{args.n_mfcc}_{args.seed}_{audio.shape[0]}.npy', allow_pickle=True)
        except FileNotFoundError:
            mfcc = librosa.feature.mfcc(y=audio[0], sr=args.fs, n_mfcc=args.n_mfcc)
            x = np.zeros((audio.shape[0], args.n_mfcc, mfcc.shape[1]))
            for i, row in tqdm(enumerate(audio), total=audio.shape[0], postfix='MFCC'):
                x[i] = librosa.feature.mfcc(y=row, sr=args.fs, n_mfcc=args.n_mfcc)
            np.save(f'cache/mfcc_{args.n_mfcc}_{args.seed}_{audio.shape[0]}', x)
    elif args.feature_extraction == 'vta':
        zip_inputs = zip(audio, repeat(args.n_tube), repeat(args.vta_window_length))
        with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pool:
            x = pool.starmap(vta_paper, tqdm(zip_inputs, total=audio.shape[0], postfix='VTA'))
        x = np.dstack(x).T                              # (N, tubes, frames)
    else:
        x = np.empty((audio.shape[0], 0, 0))
    return x


def get_1d_data(x: np.ndarray):
    mean = np.mean(x, axis=2)
    var = np.var(x, axis=2)
    skew = stats.skew(x, axis=2)
    kurt = stats.kurtosis(x, axis=2)
    diff = abs(np.diff(x, axis=2)).sum(axis=2)
    return mean, var, skew, kurt, diff, x.reshape(x.shape[0], -1)


def get_SMOTE(X, y, seed, SMOTE_strategy, categorical_features=None) -> tuple[np.ndarray, np.ndarray]:
    print(f'Original Dataset shape {Counter(y)}')
    print(f'Over sampling with {SMOTE_strategy.__name__}')
    if SMOTE_strategy.__name__ == 'SMOTENC':
        sm = SMOTE_strategy(random_state=seed, categorical_features=categorical_features)
    else:
        sm = SMOTE_strategy(random_state=seed)
    sm_X, sm_y = sm.fit_resample(X, y)
    print(f'Resampled Dataset shape {Counter(sm_y)}')
    return (sm_X, sm_y)


def summary(y_truth, y_prob, ids, tricky_vote=False, to_left=False):
    if isinstance(y_prob, tuple):
        prob_sum = None
        for prob in y_prob:
            a = np.c_[ids, prob]
            a = pd.DataFrame(a, columns=['ID', 1, 2, 3, 4, 5])
            if prob_sum is None:
                prob_sum = a.groupby('ID').agg(pd.Series.mean)
            else:
                prob_sum += a.groupby('ID').agg(pd.Series.mean)
        results = prob_sum.idxmax(axis='columns').to_frame(name='pred')
        if tricky_vote or to_left:
            raise NotImplementedError('y_prob is a tuple.')
    else:
        y_pred = np.argmax(y_prob, axis=1) + 1
        results = pd.DataFrame({'ID': ids, 'pred': y_pred})
        if tricky_vote and not to_left:
            results = results.groupby('ID').pred.agg(max).to_frame()
        elif tricky_vote and to_left:
            results = results.groupby('ID').pred.agg(min).to_frame()
        elif to_left:
            results = results.groupby('ID').pred.agg(lambda x: min(pd.Series.mode(x))).to_frame()
        else:
            results = results.groupby('ID').pred.agg(lambda x: max(pd.Series.mode(x))).to_frame()
    ground_truth = pd.DataFrame({'ID': ids, 'truth': y_truth})
    ground_truth = ground_truth.groupby('ID').truth.agg(pd.Series.mode).to_frame()
    return results.merge(ground_truth, how='inner', on='ID', validate='1:1')


def save_checkpoint(epoch, model, optimizer, scheduler=None):
    path = f'runs/{model.__class__.__name__}_{epoch}.pt'
    if scheduler is None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        return
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)
    return path


def merge_and_check():
    template_path = 'Data/Private/submission_template_public+private.csv'
    template = pd.read_csv(template_path, header=None, names=['ID', 'fake'])
    f1 = pd.read_csv(sys.argv[1], header=None, names=['ID', 'pred'])
    f2 = pd.read_csv(sys.argv[2], header=None, names=['ID', 'pred'])
    df = pd.concat([f1, f2], ignore_index=True, verify_integrity=True)
    print('Duplicate IDs:', df.duplicated(subset=['ID']).any())
    if df.ID.isin(template.ID).all() and template.ID.isin(df.ID).all():
        print('Union Checked.\nNumber of Samples:', df.shape[0])
        df.set_index('ID').to_csv(sys.argv[3], header=False)
        return
    print('Mismatched IDs!')


if __name__ == '__main__':
    merge_and_check()
