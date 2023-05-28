import torch
import torch.nn as nn


class EarlyFusionNN(nn.Module):
    def __init__(self, audio_shape, clinical_shape, output_dim, *args, **kwargs) -> None:
        super(EarlyFusionNN, self).__init__(*args, **kwargs)
        input_dim = audio_shape[1] + clinical_shape[1]
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, output_dim, bias=False),
        )

    def forward(self, a, c):
        x = torch.cat((a, c), dim=1)
        return self.fc(x)


class LateFusionNN(nn.Module):
    def __init__(self, audio_shape, clinical_shape, output_dim, *args, **kwargs):
        super(LateFusionNN, self).__init__(*args, **kwargs)
        self.audio_fc = AudioNN(audio_shape, clinical_shape, output_dim)
        self.clinical_fc = ClinicalNN(audio_shape, clinical_shape, output_dim)

    def forward(self, a, c):
        out_a = self.audio_fc(a, c)
        out_c = self.clinical_fc(a, c)
        return out_a + out_c


class LateFusionCNN(nn.Module):
    def __init__(self, audio_shape, clinical_shape, output_dim, *args, **kwargs):
        super(LateFusionCNN, self).__init__(*args, **kwargs)
        self.audio_fc = AudioCNN(audio_shape, clinical_shape, output_dim)
        self.clinical_fc = ClinicalNN(audio_shape, clinical_shape, output_dim)

    def forward(self, a, c):
        out_a = self.audio_fc(a, c)
        out_c = self.clinical_fc(a, c)
        return out_a + out_c


class AudioNN(nn.Module):
    def __init__(self, audio_shape, clinical_shape, output_dim, *args, **kwargs):
        super(AudioNN, self).__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(audio_shape[1], 128, bias=False),
            nn.Linear(128, 32, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, output_dim, bias=False),
        )

    def forward(self, a, c):
        return self.fc(a)


class ClinicalNN(nn.Module):
    def __init__(self, audio_shape, clinical_shape, output_dim, *args, **kwargs):
        super(ClinicalNN, self).__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Linear(clinical_shape[1], 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 32, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, output_dim, bias=False),
        )

    def forward(self, a, c):
        return self.fc(c)


class AudioCNN(nn.Module):
    def __init__(self, audio_shape, clinical_shape, output_dim, *args, **kwargs):
        super(AudioCNN, self).__init__(*args, **kwargs)
        if audio_shape[1] == 21 and audio_shape[2] == 21:       # VTA
            self.conv = nn.Sequential(
                nn.Conv2d(1, 6, 5),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(6, 16, 3),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(3, 3)
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 5 * 5, 128, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(128, 64, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(64, output_dim, bias=False),
            )
        elif audio_shape[1] == 13 and audio_shape[2] == 8:      # MFCC
            self.conv = nn.Sequential(
                nn.Conv2d(1, 6, 3),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(6, 16, 3),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d((3, 2), (3, 2))
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 3 * 2, 128, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(128, 32, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Linear(32, output_dim, bias=False),
            )

    def forward(self, a, c):
        x = self.conv(torch.unsqueeze(a, dim=1))
        return self.fc(torch.flatten(x, start_dim=1, end_dim=-1))
