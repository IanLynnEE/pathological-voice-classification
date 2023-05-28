import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier
import torch

from config import get_config
from models import EarlyFusionNN, LateFusionNN, LateFusionCNN, ClinicalNN, AudioNN, AudioCNN
from preprocess import read_files
from train_nn import get_dataloader, evaluate
from utils import get_audio_features, summary


def main():
    args = get_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prepare data.
    df = pd.read_csv(args.csv_path)
    _, valid = train_test_split(df, test_size=0.2, stratify=df['Disease category'], random_state=args.seed)
    drop_cols = ['ID', 'Disease category', 'PPD']

    if args.test_csv_path is not None:
        valid = pd.read_csv(args.test_csv_path)

    xv_audio_raw, xv_clinical, yv, ids = read_files(valid, args.test_audio_dir, args.fs, args.frame_length, drop_cols)
    xv_audio = get_audio_features(xv_audio_raw, args)

    # Check model name and decided dimensions of the audio features.
    if not isinstance(args.model, list):
        args.model = [args.model, None]
    if 'CNN' not in args.model[0]:
        if args.model[1] is None or 'CNN' not in args.model[1]:
            xv_audio = xv_audio.reshape(xv_audio.shape[0], -1)
    dataloader = get_dataloader(xv_audio, xv_clinical, yv, args.batch_size)

    # Get results. If args.model[1] is None, y_prob_2 is None.
    y_prob_1 = load_model_and_get_result(args.model[0], xv_audio, xv_clinical, dataloader, device)
    y_prob_2 = load_model_and_get_result(args.model[1], xv_audio, xv_clinical, dataloader, device)
    if y_prob_2 is None:
        y_prob_2 = y_prob_1

    results = summary(yv, (y_prob_1, y_prob_2), ids)

    # Store or log results
    filename = f'{args.prefix}_{args.model[0]}_{args.model[1]}_{args.feature_extraction}'
    if args.test_csv_path is not None:
        if args.output is not None:
            results.drop(columns=['truth']).to_csv(args.output, header=False)
            return
        results.drop(columns=['truth']).to_csv(f'{filename}.csv', header=False)

    print(classification_report(results.truth, results.pred, zero_division=0))
    display = ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    display.figure_.savefig(f'runs/{filename}.png', dpi=300)
    display.figure_.clf()
    return


def load_model_and_get_result(model_name, xv_audio, xv_clinical, dataloader, device):
    if model_name is None:
        return None
    if 'RF' in model_name:
        model: BalancedRandomForestClassifier = joblib.load(f'runs/{model_name}.pkl')
        if 'Audio' in model_name:
            return model.predict_proba(xv_audio)
        if 'Clinical' in model_name:
            return model.predict_proba(xv_clinical)
    if 'NN' in model_name:
        model_class_name = model_name.split('_')[0]
        model: torch.nn.Module = eval(model_class_name)(xv_audio.shape, xv_clinical.shape, 5)
        model.to(device)
        checkpoint = torch.load(f'runs/{model_name}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        _, y_prob = evaluate(device, model, None, dataloader, has_answers=False)
        return y_prob
    raise ValueError(f'Invalid model name: {model_name}')


if __name__ == '__main__':
    main()
