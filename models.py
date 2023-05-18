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

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None,):
        # shape: (N * seq_len, feat_dim) x (feat_dim, 1) = (N * seq_len, 1)
        eij = torch.mm(
            x.contiguous().view(-1, self.feature_dim), 
            self.weight
        ).view(-1, self.step_dim)

        # shape: (N, seq_len) + (*, seq_len) = (N, seq_len)
        if self.bias:
            eij = eij + self.b

        # shape: (N, seq_len)
        eij = torch.tanh(eij)
        a = torch.exp(eij)

        # shape: (N, seq_len) * (*, seq_len) = (N, seq_len)
        if mask is not None:
            a = a * mask

        # shape: (N, seq_len) / (N, seq_len) = (N, seq_len)
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        # shape: (N, seq_len, feat_dim) * (N, seq_len, *) = (N, seq_len, feat_dim)
        weighted_input = x * torch.unsqueeze(a, -1)
        # shape: (N, feat_dim)
        return torch.sum(weighted_input, 1)

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
        self.rnn_bidirectional = RNN_params["bidirectional"]
        dropout_rate = RNN_params["dropout_rate"]
        self.gru = nn.LSTM(
            audio_dim,
            self.rnn_hidden_size,
            self.rnn_num_layers,
            batch_first=True,
            dropout=(0 if self.rnn_num_layers == 1 else dropout_rate),
            bidirectional=self.rnn_bidirectional,
        )

        self.nn_hidden_size = NN_params["hidden_size"]
        dropout_rate = NN_params["dropout_rate"]
        activation = NN_params["activation"]
        down_factor = NN_params["down_factor"]
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
        )

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=21, nhead=1, dim_feedforward=64, dropout=0.1,
            activation=activation, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)

        down_factor = fusion_params["down_factor"]
        dropout_rate = fusion_params["dropout_rate"]
        fusion_input_dim = self.rnn_hidden_size * (1+int(self.rnn_bidirectional)) + self.nn_hidden_size
        # fusion_input_dim = 21 + self.nn_hidden_size
        # fusion_input_dim = self.rnn_hidden_size
        hidden_size = fusion_input_dim // down_factor
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_dim)
        )


    def forward(self, a, c):
        h = self.init_hidden(a.size(0))
        c0 = self.init_hidden(a.size(0))
        # output, hidden, _ = self.gru(a, h)
        output, (hidden, _) = self.gru(a, (h, c0))
        h_gru = hidden[-(1+int(self.rnn_bidirectional)):]
        # h_gru = output[:, -1, :]
        out_a = h_gru.squeeze(0)
        # out_a = self.attention_layer(output, )
        # out_a = self.transformer_encoder(a, src_key_padding_mask=None)
        # out_a = out_a[:, -1]
        out_c = self.nn(c)
        fusion_x = torch.cat([out_a, out_c], 1)
        output = self.fusion(fusion_x)
        return output

    def init_hidden(self, batch_size):
        return torch.zeros(
            self.rnn_num_layers * (1 + int(self.rnn_bidirectional)), batch_size, self.rnn_hidden_size, device=self.device
        )