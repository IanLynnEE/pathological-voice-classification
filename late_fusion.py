import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier

from config import get_config
from preprocess import read_files
from utils import get_audio_features, summary


def main():
    args = get_config()

    df = pd.read_csv(args.csv_path)
    train, valid = train_test_split(df, test_size=0.2, stratify=df['Disease category'], random_state=args.seed)
    drop_cols = ['ID', 'Disease category', 'PPD']

    if args.test_csv_path != 'None' and args.test_audio_dir != 'None':
        train = df
        valid = pd.read_csv(args.test_csv_path)

    model_a, model_c = train_two_model(train, args, drop_cols)

    audio, clinical, y, ids = read_files(valid, args.test_audio_dir, args.fs, args.frame_length, drop_cols)
    mean, var, skew, kurt, diff, all = get_audio_features(audio, args)

    y_prob_a = model_a.predict_proba(all)
    y_prob_c = model_c.predict_proba(clinical)
    results = summary(y, (y_prob_a, y_prob_c), ids)

    if args.test_csv_path != 'None' and args.test_audio_dir != 'None':
        results.drop(columns=['truth']).to_csv('test.csv', header=False)

    print(classification_report(results.truth, results.pred, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    plt.savefig('confusion_matrix.png', dpi=300)
    return


def train_two_model(train, args, drop_cols):
    audio, _, y_audio, _ = read_files(train, args.audio_dir, args.fs, args.frame_length, drop_cols)
    mean, var, skew, kurt, diff, all = get_audio_features(audio, args)

    clinical_feature = train.drop(columns=drop_cols).fillna(0).to_numpy()
    y_clinical = train['Disease category'].to_numpy()

    model_audio = BalancedRandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=-2
    )
    model_clinical = BalancedRandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=-2
    )
    model_audio.fit(all, y_audio)
    model_clinical.fit(clinical_feature, y_clinical)
    return model_audio, model_clinical


if __name__ == '__main__':
    main()
