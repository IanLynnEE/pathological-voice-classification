import argparse
import os

import librosa
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from preprocess import read_files
from vta import vta_huang, vta_paper


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--csv_path', type=str, default='Data/Train/data_list.csv')
    parser.add_argument('--sound_dir', type=str, default='Data/Train/raw')

    parser.add_argument('--fs', type=int, default=22050)
    parser.add_argument('--frame_length', type=int, default=3675)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--n_mfcc', type=int, default=13)

    parser.add_argument('--n_tube', type=int, default=16)

    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--max_features', type=str, default='sqrt')
    return parser.parse_args()


def main():
    args = config()
    manual_seed(args.seed)

    audio, y, n_repeat, df = read_files(args.csv_path, args.sound_dir, args.fs, args.frame_length)
    x = get_mfcc(audio, args)

    # Example of using VTA. Please note the `mean` is not a good choice here.
    # x = np.zeros((audio.shape[0], args.n_tube))
    # for i, row in tqdm(enumerate(audio), postfix='VTA'):
    #     vta = vta_paper(row, n_tube=args.n_tube)
    #     x[i] = vta.mean(axis=1)

    # Remove answers and unnecessary columns for clinical data.
    df.drop(columns=['ID', 'Disease category', 'PPD', 'Voice handicap index - 10'], inplace=True)

    # Clinical data needs to match the number of samples of audio.
    clinical = np.repeat(df.to_numpy(), repeats=n_repeat, axis=0)

    # Split into train and validation.
    x, x_val, y, y_val = train_test_split(np.hstack((clinical, x)), y, test_size=0.2, stratify=y)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=-2
    )
    model.fit(x, y)
    y_pred = model.predict(x_val)
    print(classification_report(y_val, y_pred))

    # Amazing results by using clinical data only, so we need to check what happened.
    # print(pd.DataFrame(model.feature_importances_, index=df.columns, columns=['importance']))
    return


def manual_seed(seed: int) -> None:
    np.random.seed(seed)
    return


def get_mfcc(audio, args) -> np.ndarray:
    """Extract features from audio files by MFCC. If the cache file exist, load features from the file.

    Args:
        audio (np.ndarray): audio data
        args (argparse.Namespace): arguments

    Returns:
        x (np.ndarray): for each sample, flattened mfcc features
    """
    cache = f'{args.frame_length}_{args.n_fft}_{args.n_mfcc}.csv'
    try:
        x = np.loadtxt(os.path.join(args.sound_dir, cache), delimiter=',')
    except FileNotFoundError:
        x = np.zeros((audio.shape[0], args.n_mfcc))
        for i, row in tqdm(enumerate(audio), postfix='MFCC'):
            mfcc = librosa.feature.mfcc(y=row, sr=args.fs, n_mfcc=args.n_mfcc)
            x[i] = mfcc.mean(axis=1)
        np.savetxt(os.path.join(args.sound_dir, cache), x, delimiter=',')
    return x


if __name__ == '__main__':
    main()
