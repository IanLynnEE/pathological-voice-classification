import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self, audio_dim, clinical_dim, output_dim, *args, **kwargs) -> None:
        super(NN, self).__init__(*args, **kwargs)
        input_dim = audio_dim + clinical_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64, bias=False),
            nn.LeakyReLU(),
            nn.Linear(64, 32, bias=False),
            nn.LeakyReLU(),
            nn.Linear(32, output_dim, bias=False),
        )

    def forward(self, a, c):
        x = torch.cat((a, c), dim=1)
        return self.fc(x)


class LateFusionNN(nn.Module):
    def __init__(self, audio_dim, clinical_dim, output_dim, *args, **kwargs) -> None:
        super(LateFusionNN, self).__init__(*args, **kwargs)
        self.audio_fc = AudioNN(audio_dim, output_dim)
        self.clinical_fc = ClinicalNN(clinical_dim, output_dim)
        # self.merge = nn.Linear(output_dim * 2, output_dim, bias=False)

    def forward(self, a, b):
        out_a = self.audio_fc(a)
        out_c = self.clinical_fc(b)
        return out_a + out_c


class AudioNN(nn.Module):
    def __init__(self, audio_dim, output_dim, *args, **kwargs) -> None:
        super(AudioNN, self).__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32, bias=False),
            nn.LeakyReLU(),
            nn.Linear(32, output_dim, bias=False),
        )

    def forward(self, a):
        return self.fc(a)


class ClinicalNN(nn.Module):
    def __init__(self, clinical_dim, output_dim, *args, **kwargs) -> None:
        super(ClinicalNN, self).__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32, bias=False),
            nn.LeakyReLU(),
            nn.Linear(32, output_dim, bias=False),
        )

    def forward(self, x):
        return self.fc(x)
