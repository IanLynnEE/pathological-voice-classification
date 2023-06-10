import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config
from models import EarlyFusionNN, LateFusionNN, LateFusionCNN, ClinicalNN, AudioNN, AudioCNN
from preprocess import read_files
from utils import get_audio_features, summary, save_checkpoint

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
    args = get_config()

    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed(args.torch_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Binary task mode: {args.binary_task}")

    train = pd.read_csv(args.train_csv_path)
    valid = pd.read_csv(args.valid_csv_path)
    test = pd.read_csv(args.test_csv_path)
    drop_cols = ['ID', 'Disease category', 'PPD']

    # Train Data.
    x_audio_raw, x_clinical, y_audio, _ = read_files(train, args.train_audio_dir, args.fs, args.frame_length, drop_cols, args.binary_task)
    x_audio = get_audio_features(x_audio_raw, args)

    # Valid Data.
    xv_audio_raw, xv_clinical, yv, ids = read_files(valid, args.valid_audio_dir, args.fs, args.frame_length, drop_cols, args.binary_task)
    xv_audio = get_audio_features(xv_audio_raw, args)

    # Test Data.
    xt_audio_raw, xt_clinical, yt, ids = read_files(test, args.test_audio_dir, args.fs, args.frame_length, drop_cols, args.binary_task)
    xt_audio = get_audio_features(xt_audio_raw, args)

    # Class Weights.
    weights = compute_class_weight('balanced', classes=np.unique(y_audio), y=y_audio)
    weights = torch.tensor(weights, device=device, dtype=torch.float)

    if 'CNN' not in args.model:
        if 'RNN' not in args.model:
            x_audio = x_audio.reshape(x_audio.shape[0], -1)
            xv_audio = xv_audio.reshape(xv_audio.shape[0], -1)
            xt_audio = xt_audio.reshape(xt_audio.shape[0], -1)
        else:
            x_audio = x_audio.transpose((0, 2, 1))
            xv_audio = xv_audio.transpose((0, 2, 1))
            xt_audio = xt_audio.transpose((0, 2, 1))


    # Data Loaders.
    train_loader = get_dataloader(x_audio, x_clinical, y_audio, args.batch_size, shuffle=True)
    valid_loader = get_dataloader(xv_audio, xv_clinical, yv, args.batch_size)
    test_loader = get_dataloader(xt_audio, xt_clinical, yt, args.batch_size)

    # Model setup.
    if 'RNN' not in args.model:
        model = eval(args.model)(
            x_audio.shape,
            x_clinical.shape,
            len(np.unique(y_audio)),
        )
    else:
        model = eval(args.model)(
            x_audio.shape,
            x_clinical.shape,
            len(np.unique(y_audio)),
            RNN_params,
            NN_params,
            fusion_params,
            device,
        )
    model.to(device)
    # criterion = torch.nn.BCELoss(weight=weights) if args.binary_task else torch.nn.CrossEntropyLoss(weight=weights)
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
    writer = SummaryWriter()

    # Training.
    best_score = args.best_score
    for epoch in tqdm(range(args.epochs)):
        train_loss = train_one_epoch(device, model, criterion, optimizer, scheduler, train_loader)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        if args.valid_csv_path is None:
            valid_loss, y_prob = evaluate(device, model, criterion, valid_loader)
            score = recall_score(yv - 1, np.argmax(y_prob, axis=1), average='macro')
            writer.add_scalar('Score/Recall', score, epoch)
            writer.add_scalar('Loss/Valid', valid_loss, epoch)
            if score > best_score:
                save_checkpoint(epoch, model, optimizer, scheduler)
                best_score = score
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

    # Testing.
    _, y_prob = evaluate(device, model, criterion, test_loader)
    results = summary(yt, y_prob, ids, tricky_vote=False, to_left=True)

    # if args.test_csv_path is not None:
        # if args.output is not None:
            # results.drop(columns=['truth']).to_csv(args.output, header=False)
            # return
        # results.drop(columns=['truth']).to_csv(f'{args.prefix}_{args.model}.csv', header=False)

    print(classification_report(results.truth, results.pred, zero_division=0))
    display = ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    display.figure_.savefig(f'runs/{args.prefix}_{args.model}.png', dpi=300)
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


def get_dataloader(audio_features, clinical_features, y, batch_size, shuffle=False):
    dataset = TensorDataset(
        torch.tensor(audio_features).float(),
        torch.tensor(clinical_features).float(),
        torch.tensor(y - 1).long()
    )
    return DataLoader(dataset, batch_size, shuffle, num_workers=4, pin_memory=True)


if __name__ == '__main__':
    main()
