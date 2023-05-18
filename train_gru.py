import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config
from models import GRUNet
from utils import get_audio_features, summary 
from preprocess import read_files

torch.manual_seed(2)

RNN_params = {
    "hidden_size": 64,
    "num_layers": 3,
    "dropout_rate": 0.1,
    "bidirectional": False,
}
NN_params = {
    "hidden_size": 32,
    "down_factor": 2,
    "activation": 'relu',
    "dropout_rate": 0.0,
}
fusion_params = {
    "down_factor": 2,
    "dropout_rate": 0.1,
}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = get_config()
    print(f"seed : {args.seed}")
    # same_seed(args.seed)

    df = pd.read_csv(args.csv_path)
    if args.test_csv_path != 'None' and args.test_audio_dir != 'None':
        train = df
        valid = pd.read_csv(args.test_csv_path)
    else:
        train, valid = train_test_split(df, test_size=0.2, stratify=df['Disease category'], random_state=args.seed)

    drop_cols = ['ID', 'Disease category', 'PPD']

    # Train Data.
    audio, clinical_train, y, _ = read_files(train, args.audio_dir, args.fs, args.frame_length, drop_cols)
    print(audio.shape)
    mean, var, skew, kurt, diff, all, temporal = get_audio_features(audio, args)
    audio_train = temporal.transpose((2, 1, 0))

    # Test Data.
    audio, clinical_test, yv, ids = read_files(valid, args.test_audio_dir, args.fs, args.frame_length, drop_cols)
    if args.private_csv_path != 'None':
        private = pd.read_csv(args.private_csv_path)
        p_audio, p_clinical_test, p_yv, p_ids = read_files(private, args.private_audio_dir, args.fs, args.frame_length, drop_cols)
        audio = np.concatenate((audio, p_audio), axis=0)
        clinical_test = np.concatenate((clinical_test, p_clinical_test), axis=0)
        yv = np.concatenate((yv, p_yv), axis=0)
        ids = np.concatenate((ids, p_ids), axis=0)
    mean, var, skew, kurt, diff, all, temporal = get_audio_features(audio, args)
    audio_test = temporal.transpose((2, 1, 0))

    # Class Weights.
    _, counts = np.unique(y, return_counts=True)
    weights = torch.tensor(np.max(counts) / counts, device=device).float()

    # Data Loaders.
    train_loader = get_dataloader(audio_train, clinical_train, y, args.batch_size, True)
    valid_loader = get_dataloader(audio_test, clinical_test, yv, args.batch_size)
    
    # dataset = AudioDataset(audio_features, clinical, y)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GRUNet(audio_train.shape[2], clinical_train.shape[1], 5, RNN_params, NN_params, fusion_params, device=device)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs,
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
        three_phase=args.three_phase,
    )
    writer = SummaryWriter()

    for epoch in tqdm(range(args.epochs)):
        train_loss = train_one_epoch(device, model, criterion, optimizer, scheduler, train_loader)
        writer.add_scalar('Train Loss', train_loss, epoch)
        if args.test_csv_path == 'None':
            valid_loss, _ = evaluate(device, model, criterion, valid_loader)
            writer.add_scalar('Valid Loss', valid_loss, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    
    if args.test_csv_path == 'None':
        _, y_prob = evaluate(device, model, criterion, valid_loader)
    else:
        _, y_prob = predict(device, model, criterion, valid_loader)
    results = summary(yv, y_prob, ids, tricky_vote=False)

    results.drop(columns=['truth']).to_csv('test.csv', header=False)
    print(classification_report(results.truth, results.pred, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    plt.savefig('confusion_matrix.png', dpi=300)
    return 

def train_one_epoch(device, model, criterion, optimizer, scheduler, train_data):
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
    scheduler.step()
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

@torch.no_grad()
def predict(device, model, criterion, valid_data):
    model.eval()
    outputs = []
    for a, c, y in valid_data:
        a = a.to(device)
        c = c.to(device)
        y = y.to(device)
        out = model(a, c)
        outputs.append(out)
    return 0, torch.cat(outputs).detach().cpu().numpy()

def get_dataloader(audio_features, clinical_features, y, batch_size, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(audio_features).float(),
        torch.tensor(clinical_features).float(),
        torch.tensor(y - 1).long()
    )
    return DataLoader(dataset, batch_size, shuffle, num_workers=4, pin_memory=False)


if __name__ == '__main__':
    main()