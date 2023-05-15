from itertools import repeat
import multiprocessing
from collections import Counter

import librosa
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from vta import vta_huang, vta_paper

def get_audio_features(audio, args) -> np.ndarray:
    if args.feature_extraction == 'mfcc':
        x = np.zeros((audio.shape[0], args.n_mfcc))
        for i, row in tqdm(enumerate(audio), postfix='MFCC'):
            mfcc = librosa.feature.mfcc(y=row, sr=args.fs, n_mfcc=args.n_mfcc)
            x[i] = mfcc.mean(axis=1)
    elif args.feature_extraction == 'vta':
        zip_inputs = zip(audio, repeat(args.n_tube), repeat(args.vta_window_length))
        with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pool:
            x = pool.starmap(vta_paper, tqdm(zip_inputs, total=audio.shape[0], postfix='VTA'))
        # (tubes, frames, N)
        Mean = np.mean(np.dstack(x), axis=1).T
        Var = np.var(np.dstack(x), axis=1).T
        Skew = stats.skew(np.dstack(x), axis=1).T
        Kurt = stats.kurtosis(np.dstack(x), axis=1).T
        diff = abs(np.diff(np.dstack(x), axis=1)).sum(axis=1).T
    elif args.feature_extraction == 'empty':
        x = np.empty((audio.shape[0], 0))
    else:
        raise ValueError('Invalid feature extraction method.')
    
    if args.temporal:
        return np.dstack(x)
    return (Mean, Var, Skew, Kurt, diff)


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


def majority_vote(y_truth, y_pred, ids):
    results = pd.DataFrame({'ID': ids, 'pred': y_pred})
    results = results.groupby('ID').pred.agg(max).to_frame()
    ground_truth = pd.DataFrame({'ID': ids, 'truth': y_truth})
    ground_truth = ground_truth.groupby('ID').truth.agg(pd.Series.mode).to_frame()
    return results.merge(ground_truth, how='inner', on='ID', validate='1:1')