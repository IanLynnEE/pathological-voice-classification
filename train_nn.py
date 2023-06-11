import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, recall_score
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config
from models import EarlyFusionNN, LateFusionNN, LateFusionCNN, ClinicalNN, AudioNN, AudioCNN
from preprocess import read_files, get_audio_features
from utils import summary, save_checkpoint


def main():
    args = get_config()

    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed(args.torch_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train = pd.read_csv(args.csv_path)
    valid = pd.read_csv(args.valid_csv_path)
    drop_cols = ['ID', 'Disease category', 'PPD']

    # Train Data.
    x_audio_raw, x_clinical, y_audio, _ = read_files(train, args.audio_dir, args.fs, args.frame_length,
                                                     drop_cols, args.binary_task)
    x_audio = get_audio_features(x_audio_raw, args)

    # Test Data.
    xv_audio_raw, xv_clinical, yv, ids = read_files(valid, args.valid_audio_dir, args.fs, args.frame_length,
                                                    drop_cols, args.binary_task)
    xv_audio = get_audio_features(xv_audio_raw, args)

    # Class Weights.
    weights = compute_class_weight('balanced', classes=np.unique(y_audio), y=y_audio)
    weights = torch.tensor(weights, device=device, dtype=torch.float)

    if 'CNN' not in args.model:
        x_audio = x_audio.reshape(x_audio.shape[0], -1)
        xv_audio = xv_audio.reshape(xv_audio.shape[0], -1)

    # Data Loaders.
    train_loader = get_dataloader(x_audio, x_clinical, y_audio, args.batch_size, binary=args.binary_task, shuffle=True)
    valid_loader = get_dataloader(xv_audio, xv_clinical, yv, args.batch_size, binary=args.binary_task)

    # Model setup.
    model = eval(args.model)(x_audio.shape, x_clinical.shape, len(np.unique(y_audio)))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
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
    writer = SummaryWriter(comment=f'_{args.model}_{args.feature_extraction}_{args.lr}_{args.epochs}')

    # Training.
    best_score = args.best_score
    for epoch in tqdm(range(args.epochs)):
        train_loss = train_one_epoch(device, model, criterion, optimizer, scheduler, train_loader)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        # WARNING: This will fail if no answers are provided. Not a problem in our case, but be careful.
        valid_loss, y_prob = evaluate(device, model, criterion, valid_loader)
        score = recall_score(yv, np.argmax(y_prob, axis=1) + np.sign(y_prob.shape[1] - 2), average='macro')
        writer.add_scalar('Score/Recall', score, epoch)
        writer.add_scalar('Loss/Valid', valid_loss, epoch)
        if score > best_score:
            save_checkpoint(epoch, model, optimizer, scheduler)
            best_score = score
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    save_checkpoint(epoch, model, optimizer, scheduler)

    # Evaluating / Testing.
    _, y_prob = evaluate(device, model, criterion, valid_loader, has_answers=False)
    results = summary(yv, y_prob, ids, tricky_vote=False, to_left=True)

    results.drop(columns=['truth']).to_csv(f'{args.output}.csv', header=False)
    print(classification_report(results.truth, results.pred, zero_division=0))
    display = ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    display.figure_.savefig(f'runs/{args.output}.png', dpi=300)
    display.figure_.clf()
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
def evaluate(device, model, criterion, valid_data, has_answers=True):
    model.eval()
    loss_accum = 0.0
    outputs = []
    for a, c, y in valid_data:
        a = a.to(device)
        c = c.to(device)
        y = y.to(device)
        out = model(a, c)
        if has_answers:
            loss_accum += criterion(out, y).item()
        outputs.append(out)
    return loss_accum / len(valid_data), torch.cat(outputs).detach().cpu().numpy()


def get_dataloader(audio_features, clinical_features, y, batch_size, *, binary=False, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(audio_features).float(),
        torch.tensor(clinical_features).float(),
        torch.tensor(y - int(not binary)).long()
    )
    return DataLoader(dataset, batch_size, shuffle, num_workers=4, pin_memory=True)


if __name__ == '__main__':
    main()
