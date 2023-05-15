import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, SMOTENC, SVMSMOTE, BorderlineSMOTE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader

from config import get_config
from gru import GRUNet
from utils import get_audio_features, get_SMOTE, majority_vote
from preprocess import read_files

RNN_params = {
    "hidden_size": 52,
    "num_layers": 1,
    "dropout_rate": 0.1,
}
NN_params = {
    "hidden_size": 32,
    "down_factor": 2,
    "activation": 'relu',
    "dropout_rate": 0.2,
}
fusion_params = {
    "down_factor": 2,
    "dropout_rate": 0.1,
}

def main():
    args = get_config()

    df = pd.read_csv(args.csv_path)
    if args.test_csv_path != 'None' and args.test_audio_dir != 'None':
        train = df
        valid = pd.read_csv(args.test_csv_path)
    else:
        train, valid = train_test_split(df, test_size=0.2, stratify=df['Disease category'], random_state=args.seed)

    drop_cols = ['ID', 'Disease category', 'PPD']

    audio, clinical, y, ids = read_files(train, args.audio_dir, args.fs, args.frame_length, drop_cols)
    audio_features = get_audio_features(audio, args)
    audio_features = audio_features.transpose((2, 1, 0))

    # categorical_features = range(audio_features.shape[1], x.shape[1])
    # if args.do_smote:
        # x, y = get_SMOTE(x, y, args.seed, SMOTE_strategy=eval(args.smote_strategy), categorical_features=categorical_features)
    
    dataset = AudioDataset(audio_features, clinical, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = GRUNet(audio_features.shape[2], clinical.shape[1], 5, RNN_params, NN_params, fusion_params)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model.train()
    for epoch in (range(args.num_epochs)):
        loss_record = []
        for input, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.detach().item())

        mean_loss = sum(loss_record) / len(loss_record)
        if epoch % (args.num_epochs // np.sqrt(args.num_epochs)) == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {mean_loss: .4f}")


    audio, clinical, y, ids = read_files(valid, args.test_audio_dir, args.fs, args.frame_length, drop_cols)
    audio_features = get_audio_features(audio, args)
    audio_features = audio_features.transpose((2, 1, 0))

    # dataset = AudioDataset(audio_features, clinical)
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model.eval()
    y_pred = model((torch.tensor(audio_features).float(), torch.tensor(clinical).float()))
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1) + 1
    results = majority_vote(y, y_pred, ids)

    if args.test_csv_path != 'None' and args.test_audio_dir != 'None':
        results.drop(columns=['truth']).to_csv('test.csv', header=False)

    print(classification_report(results.truth, results.pred, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    plt.savefig('confusion_matrix.png', dpi=300)
    return 

class AudioDataset(Dataset):
    def __init__(
        self,
        audio,
        clinical,
        y=None,
    ) -> None:
        super(AudioDataset, self).__init__()
        self.audio = audio
        self.clinical = clinical
        self.y = y-1
    
    def __len__(self):
        return len(self.clinical)
    
    def __getitem__(self, index):
        audio_data = self.audio[index,:,:]
        clinical_data = self.clinical[index,:]
        y = self.y[index]
        audio_data = torch.from_numpy(audio_data).float()
        clinical_data = torch.from_numpy(clinical_data).float()
        y = torch.tensor(y).long()
        return (audio_data, clinical_data), y


if __name__ == '__main__':
    main()