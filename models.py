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
        self.merge = nn.Linear(output_dim * 2, output_dim, bias=False)

    def forward(self, a, b):
        out_a = self.audio_fc(a)
        out_c = self.clinical_fc(b)
        return self.merge(torch.cat((out_a, out_c), dim=1))


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

class GRUNet(nn.Module):
    def __init__(
        self, 
        audio_dim,
        clinical_dim,
        output_dim,
        RNN_params: dict,
        NN_params: dict,
        fusion_params: dict,
        device,
        *args, 
        **kwargs
    ) -> None:
        super(GRUNet, self).__init__(*args, **kwargs)

        self.device = device
        self.rnn_hidden_size = RNN_params["hidden_size"]
        self.rnn_num_layers = RNN_params["num_layers"]
        # batch_norm = RNN_params["batch_norm"]
        dropout_rate = RNN_params["dropout_rate"]
        self.gru = nn.GRU(
            audio_dim,
            self.rnn_hidden_size,
            self.rnn_num_layers,
            batch_first=True,
            dropout=(0 if self.rnn_num_layers == 1 else dropout_rate),
        )

        self.nn_hidden_size = NN_params["hidden_size"]
        dropout_rate = NN_params["dropout_rate"]
        activation = NN_params["activation"]
        down_factor = NN_params["down_factor"]
        # batch_norm = NN_params["batch_norm"]
        # hidden_size = clinical_input_dim // down_factor
        if activation == "relu":
            act_fn = nn.LeakyReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        
        self.nn = nn.Sequential(
            nn.Linear(clinical_dim, self.nn_hidden_size),
            act_fn,
            nn.BatchNorm1d(self.nn_hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(self.nn_hidden_size, self.nn_hidden_size),
            act_fn,
            nn.BatchNorm1d(self.nn_hidden_size),
            # nn.Dropout(dropout_rate),
        )

        down_factor = fusion_params["down_factor"]
        dropout_rate = fusion_params["dropout_rate"]
        fusion_input_dim = self.rnn_hidden_size + self.nn_hidden_size
        hidden_size = fusion_input_dim // down_factor
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size),
            act_fn,
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, a, c):
        h = self.init_hidden(a.size(0))
        output, hidden = self.gru(a, h)
        h_gru = hidden[-1:]
        h_gru = h_gru.squeeze(0)
        output = self.nn(c)
        fusion_x = torch.cat([h_gru, output], 1)
        output = self.fusion(fusion_x)

        return output

    def init_hidden(self, batch_size):
        return torch.zeros(
            self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=self.device
        )