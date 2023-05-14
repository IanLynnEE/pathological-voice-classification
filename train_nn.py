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
from torch.utils.data import DataLoader

from config import get_config
from utils import get_audio_features, get_SMOTE, majority_vote
from preprocess import read_files

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
    # audio_features = get_audio_features(audio, args)
    mean, var, skew, kurt, diff = get_audio_features(audio, args)
    audio_features = np.hstack((mean, var, skew, kurt, diff))
    x = np.hstack((audio_features, clinical))

    categorical_features = range(audio_features.shape[1], x.shape[1])
    if args.do_smote:
        x, y = get_SMOTE(x, y, args.seed, SMOTE_strategy=eval(args.smote_strategy), categorical_features=categorical_features)
    
    dataset = TensorDataset(torch.tensor(x).float(), torch.tensor(y-1).long())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = NN(x.shape[1], 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    model.train()
    for epoch in (range(args.num_epochs)):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % (args.num_epochs // np.sqrt(args.num_epochs)) == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item(): .4f}")


    audio, clinical, y, ids = read_files(valid, args.test_audio_dir, args.fs, args.frame_length, drop_cols)
    # audio_features = get_audio_features(audio, args)
    mean, var, skew, kurt, diff = get_audio_features(audio, args)
    audio_features = np.hstack((mean, var, skew, kurt, diff))
    x = np.hstack((audio_features, clinical))

    model.eval()
    y_pred = model(torch.tensor(x).float())
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1) + 1
    results = majority_vote(y, y_pred, ids)

    if args.test_csv_path != 'None' and args.test_audio_dir != 'None':
        results.drop(columns=['truth']).to_csv('test.csv', header=False)

    print(classification_report(results.truth, results.pred, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(results.truth, results.pred)
    plt.savefig('confusion_matrix.png', dpi=300)
    return 

class NN(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs) -> None:
        super(NN, self).__init__(*args, **kwargs)

        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 64)
        # self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=0)

        return x

if __name__ == '__main__':
    main()