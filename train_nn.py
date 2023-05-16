import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config
from models import NN, LateFusionNN, AudioNN, ClinicalNN
from utils import get_audio_features, summary
from preprocess import read_files


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = get_config()

    df = pd.read_csv(args.csv_path)
    if args.test_csv_path != 'None':
        train = df
        valid = pd.read_csv(args.test_csv_path)
    else:
        train, valid = train_test_split(df, test_size=0.2, stratify=df['Disease category'], random_state=args.seed)

    drop_cols = ['ID', 'Disease category', 'PPD']

    # Train Data.
    audio, clinical_train, y, _ = read_files(train, args.audio_dir, args.fs, args.frame_length, drop_cols)
    mean, var, skew, kurt, diff, all = get_audio_features(audio, args)
    audio_train = all

    # Test Data.
    audio, clinical_test, yv, ids = read_files(valid, args.test_audio_dir, args.fs, args.frame_length, drop_cols)
    mean, var, skew, kurt, diff, all = get_audio_features(audio, args)
    audio_test = all  # np.hstack((mean, var, skew, kurt, diff))

    # Class Weights.
    _, counts = np.unique(y, return_counts=True)
    weights = torch.tensor(np.max(counts) / counts, device=device).float()

    # Data Loaders.
    train_loader = get_dataloader(audio_train, clinical_train, y, args.batch_size)
    valid_loader = get_dataloader(audio_test, clinical_test, yv, args.batch_size)

    model = NN(audio_train.shape[1], clinical_train.shape[1], 5)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    writer = SummaryWriter()
    for epoch in tqdm(range(args.epochs)):
        train_loss = train_one_epoch(device, model, criterion, optimizer, train_loader)
        writer.add_scalar('Train Loss', train_loss, epoch)
        if args.test_csv_path == 'None':
            valid_loss, _ = evaluate(device, model, criterion, valid_loader)
            writer.add_scalar('Valid Loss', valid_loss, epoch)

    _, y_prob = evaluate(device, model, criterion, valid_loader)
    results = summary(yv, y_prob, ids, tricky_vote=True)

    results.drop(columns=['truth']).to_csv('test.csv', header=False)
    print(classification_report(results.truth, results.pred, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    plt.savefig('confusion_matrix.png', dpi=300)
    return


def train_one_epoch(device, model, criterion, optimizer, train_data):
    model.train()
    loss_accum = 0.0
    for a, c, y in train_data:
        a = a.to(device)
        c = c.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(a, c)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_accum += loss.item()
    return loss_accum / len(train_data)


@torch.no_grad()
def evaluate(device, model, criterion, valid_data):
    model.eval()
    loss_accum = 0.0
    outputs = []
    for a, c, y in valid_data:
        a = a.to(device)
        c = c.to(device)
        y = y.to(device)
        out = model(a, c)
        loss = criterion(out, y)
        loss_accum += loss.item()
        outputs.append(out)
    return loss_accum / len(valid_data), torch.cat(outputs).detach().cpu().numpy()


def get_dataloader(audio_features, clinical_features, y, batch_size):
    dataset = TensorDataset(
        torch.tensor(audio_features).float(),
        torch.tensor(clinical_features).float(),
        torch.tensor(y - 1).long()
    )
    return DataLoader(dataset, batch_size, True, num_workers=4, pin_memory=False)


if __name__ == '__main__':
    main()
