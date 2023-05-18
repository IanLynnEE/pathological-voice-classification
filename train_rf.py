import joblib

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

    if args.test_csv_path != 'None':
        train = df
        valid = pd.read_csv(args.test_csv_path)

    x_audio_raw, x_clinical, y_audio, _ = read_files(train, args.audio_dir, args.fs, args.frame_length, drop_cols)
    x_audio = get_audio_features(x_audio_raw, args).reshape(x_audio_raw.shape[0], -1)

    xv_audio_raw, xv_clinical, yv, ids = read_files(valid, args.test_audio_dir, args.fs, args.frame_length, drop_cols)
    xv_audio = get_audio_features(xv_audio_raw, args).reshape(xv_audio_raw.shape[0], -1)

    if args.single_rf:
        x = np.hstack((x_audio, x_clinical))
        xv = np.hstack((xv_audio, xv_clinical))
        model = train_rf_model(args, x, y_audio)
        joblib.dump(model, 'runs/SingleRF')
        results = summary(yv, model.predict_proba(xv), ids)
    else:
        x_clinical = train.drop(columns=drop_cols).fillna(0).to_numpy()
        y_clinical = train['Disease category'].to_numpy()
        model_c = train_rf_model(args, x_clinical, y_clinical)
        joblib.dump(model_c, 'runs/ClinicalRF')
        y_prob_c = model_c.predict_proba(xv_clinical)

        if args.feature_extraction == 'clinical_only':
            results = summary(yv, y_prob_c, ids)
        else:
            model_a = train_rf_model(args, x_audio, y_audio)
            joblib.dump(model_a, f'runs/AudioRF_{args.feature_extraction}')
            y_prob_a = model_a.predict_proba(xv_audio)
            results = summary(yv, (y_prob_a, y_prob_c), ids)

    if args.test_csv_path != 'None':
        results.drop(columns=['truth']).to_csv(f'{args.prefix}_rf_{args.feature_extraction}.csv', header=False)

    print(classification_report(results.truth, results.pred, zero_division=0))
    display = ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    display.figure_.savefig(f'log/rf_{args.feature_extraction}_{args.rf_seed}.png', dpi=300)
    display.figure_.clf()
    return


def train_rf_model(args, x, y):
    model = BalancedRandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=-2,
        random_state=args.rf_seed,
    )
    model.fit(x, y)
    return model


if __name__ == '__main__':
    main()
