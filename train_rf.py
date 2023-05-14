import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier

from config import get_config
from utils import get_audio_features, get_SMOTE, majority_vote
from preprocess import read_files

def main():
    args = get_config()

    df = pd.read_csv(args.csv_path)
    train, valid = train_test_split(df, test_size=0.2, stratify=df['Disease category'], random_state=args.seed)
    if args.test_csv_path != 'None' and args.test_audio_dir != 'None':
        train = df
        valid = pd.read_csv(args.test_csv_path)

    drop_cols = ['ID', 'Disease category', 'PPD']

    audio, clinical, y, ids = read_files(train, args.audio_dir, args.fs, args.frame_length, drop_cols)
    audio_features = get_audio_features(audio, args)
    x = np.hstack((audio_features, clinical))

    categorical_features = range(audio_features.shape[1], x.shape[1])
    if args.do_smote:
        x, y = get_SMOTE(x, y, args.seed, SMOTE_strategy=eval(args.smote_strategy), categorical_features=categorical_features)

    model = BalancedRandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=-2
    )
    model.fit(x, y)

    audio, clinical, y, ids = read_files(valid, args.test_audio_dir, args.fs, args.frame_length, drop_cols)
    audio_features = get_audio_features(audio, args)
    x = np.hstack((audio_features, clinical))

    y_pred = model.predict(x)
    results = majority_vote(y, y_pred, ids)

    if args.test_csv_path != 'None' and args.test_audio_dir != 'None':
        results.drop(columns=['truth']).to_csv('test.csv', header=False)

    print(classification_report(results.truth, results.pred, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    plt.savefig('confusion_matrix.png', dpi=300)
    return


if __name__ == '__main__':
    main()
