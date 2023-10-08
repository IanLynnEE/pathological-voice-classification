import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.ensemble import BalancedRandomForestClassifier

from config import get_config
from preprocess import read_files, get_audio_features, get_1d_data
from utils import summary


def main():
    args = get_config()

    # Check the model name.
    if isinstance(args.model, str):
        if args.model not in ['EarlyFusionRF', 'ClinicalRF', 'AudioRF']:
            raise ValueError('Invalid model name.')

    train = pd.read_csv(args.csv_path)
    valid = pd.read_csv(args.valid_csv_path)
    drop_cols = ['ID', 'Disease category', 'PPD']

    x_audio_raw, x_clinical, y_audio, _ = read_files(train, args.audio_dir, args.fs, args.frame_length,
                                                     drop_cols, args.binary_task)
    mean, var, skew, kurt, diff, all = get_1d_data(get_audio_features(x_audio_raw, args))
    x_audio = np.hstack((mean, var, skew, kurt, diff, all))

    xv_audio_raw, xv_clinical, yv, ids = read_files(valid, args.valid_audio_dir, args.fs, args.frame_length,
                                                    drop_cols, args.binary_task)
    mean, var, skew, kurt, diff, all = get_1d_data(get_audio_features(xv_audio_raw, args))
    xv_audio = np.hstack((mean, var, skew, kurt, diff, all))

    # Early fusion.
    if args.model == 'EarlyFusionRF':
        x = np.hstack((x_audio, x_clinical))
        xv = np.hstack((xv_audio, xv_clinical))
        model = train_rf_model(args, x, y_audio)
        joblib.dump(model, f'runs/EarlyFusionRF_{args.feature_extraction}.pkl')
        results = summary(yv, model.predict_proba(xv), ids)
        store_results(args, results)
        return

    # For late fusion, clinical part should be trained without augmentation.
    x_clinical = train.drop(columns=drop_cols).fillna(0).to_numpy()
    y_clinical = train['Disease category'].to_numpy()
    y_clinical = np.where(y_clinical == 5, 0, 1) if args.binary_task else y_clinical

    if 'ClinicalRF' in args.model:
        model_c = train_rf_model(args, x_clinical, y_clinical)
        joblib.dump(model_c, 'runs/ClinicalRF.pkl')
        y_prob_c = model_c.predict_proba(xv_clinical)

        # Clinical only.
        if isinstance(args.model, str):
            results = summary(yv, y_prob_c, ids)
            store_results(args, results)
            return

    if 'AudioRF' in args.model:
        model_a = train_rf_model(args, x_audio, y_audio)
        joblib.dump(model_a, f'runs/AudioRF_{args.feature_extraction}.pkl')
        y_prob_a = model_a.predict_proba(xv_audio)

        # Audio only.
        if isinstance(args.model, str):
            results = summary(yv, y_prob_a, ids)
            store_results(args, results)
            return

    # Late fusion.
    if 'ClinicalRF' in args.model and 'AudioRF' in args.model:
        results = summary(yv, (y_prob_a, y_prob_c), ids)
        store_results(args, results)
        return
    raise ValueError('Invalid model name.')


def train_rf_model(args, x, y):
    model = BalancedRandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=-2,
        random_state=args.rf_seed,
        sampling_strategy='not minority',
        replacement=False,      # True is actually better, but this is the original setting.
    )
    model.fit(x, y)
    return model


def store_results(args, results):
    results.drop(columns=['truth']).to_csv(f'{args.output}.csv', header=False)
    print(classification_report(results.truth, results.pred, zero_division=0))
    display = ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    display.figure_.savefig(f'runs/{args.output}.png', dpi=300)
    display.figure_.clf()
    return


if __name__ == '__main__':
    main()
