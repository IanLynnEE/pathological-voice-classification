import argparse
from itertools import repeat
import multiprocessing

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from preprocess import read_files
from vta import vta_huang, vta_paper


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--csv_path', type=str, default='Data/Train/data_list.csv')
    parser.add_argument('--audio_dir', type=str, default='Data/Train/raw')
    parser.add_argument('--feature_extraction', type=str, default='vta', choices=['mfcc', 'vta', 'empty'])

    parser.add_argument('--fs', type=int, default=22050)
    parser.add_argument('--frame_length', type=int, default=3675)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--n_mfcc', type=int, default=13)

    parser.add_argument('--n_tube', type=int, default=21)
    parser.add_argument('--vta_window_length', type=int, default=147)

    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--max_features', type=str, default='sqrt')

    args = parser.parse_args()
    args.max_features = int(args.max_features) if args.max_features.isdecimal() else args.max_features
    return args


def main():
    args = config()
    manual_seed(args.seed)

    df = pd.read_csv(args.csv_path)
    train, valid = train_test_split(df, test_size=0.2, stratify=df['Disease category'])

    drop_cols = ['ID', 'Disease category', 'PPD', 'Voice handicap index - 10']

    audio, clinical, y, ids = read_files(train, args.audio_dir, args.fs, args.frame_length, drop_cols)
    audio_features = get_audio_features(audio, args)
    x = np.hstack((audio_features, clinical))

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=-2
    )
    model.fit(x, y)

    audio, clinical, y, ids = read_files(valid, args.audio_dir, args.fs, args.frame_length, drop_cols)
    audio_features = get_audio_features(audio, args)
    x = np.hstack((audio_features, clinical))

    y_pred = model.predict(x)
    results = majority_vote(y, y_pred, ids)

    print(classification_report(results.truth, results.pred, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    plt.savefig('confusion_matrix.png', dpi=300)
    return


def manual_seed(seed: int) -> None:
    np.random.seed(seed)
    return


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
        x = np.dstack(x).mean(axis=1).T
    elif args.feature_extraction == 'empty':
        x = np.empty((audio.shape[0], 0))
    else:
        raise ValueError('Invalid feature extraction method.')
    return x


def majority_vote(y_truth, y_pred, ids):
    results = pd.DataFrame({'ID': ids, 'pred': y_pred})
    results = results.groupby('ID').pred.agg(lambda x: pd.Series.mode(x)[0]).to_frame()
    ground_truth = pd.DataFrame({'ID': ids, 'truth': y_truth})
    ground_truth = ground_truth.groupby('ID').truth.agg(pd.Series.mode).to_frame()
    return results.merge(ground_truth, how='inner', on='ID', validate='1:1')


if __name__ == '__main__':
    main()
