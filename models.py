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
    ):
        super(GRUNet, self).__init__()

        self.device = device
        self.rnn_hidden_size = RNN_params.get("hidden_size", 128)
        self.rnn_num_layers = RNN_params.get("num_layers", 3)
        self.bidirectional = RNN_params.get("bidirectional", False)
        rnn_dropout_rate = RNN_params.get("dropout_rate", 0.1)

        self.gru = nn.GRU(
            audio_dim[2],
            self.rnn_hidden_size,
            self.rnn_num_layers,
            batch_first=True,
            dropout=rnn_dropout_rate if self.rnn_num_layers > 1 else 0,
            bidirectional=self.bidirectional,
        )

        self.nn_hidden_size = NN_params.get("hidden_size", 128)
        nn_dropout_rate = NN_params.get("dropout_rate", 0.1)
        activation = NN_params.get("activation", "relu")
        down_factor = NN_params.get("down_factor", 2)

        if activation == "relu":
            act_fn = nn.LeakyReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        
        self.nn = nn.Sequential(
            nn.Linear(clinical_dim[1], self.nn_hidden_size),
            act_fn,
            nn.BatchNorm1d(self.nn_hidden_size),
            nn.Dropout(nn_dropout_rate),
            nn.Linear(self.nn_hidden_size, self.nn_hidden_size),
        )

        down_factor = fusion_params.get("down_factor", 2)
        fusion_dropout_rate = fusion_params.get("dropout_rate", 0.1)
        fusion_input_dim = self.rnn_hidden_size * (1+int(self.bidirectional)) + self.nn_hidden_size
        hidden_size = fusion_input_dim // down_factor
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Dropout(fusion_dropout_rate),
            nn.Linear(hidden_size, output_dim)
        )


    def forward(self, a, c):
        h = self.init_hidden(a.size(0))
        output, hidden = self.gru(a, h)
        if self.bidirectional:
            out_a = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            out_a = hidden[-1]

        out_c = self.nn(c)
        fusion_x = torch.cat([out_a, out_c], 1)
        output = self.fusion(fusion_x)
        return output

    def init_hidden(self, batch_size):
        return torch.zeros(
            self.rnn_num_layers * (1 + int(self.bidirectional)), batch_size, self.rnn_hidden_size, device=self.device
        )


class LSTMNet(nn.Module):
    def __init__(
        self, 
        audio_dim,
        clinical_dim,
        output_dim,
        RNN_params: dict,
        NN_params: dict,
        fusion_params: dict,
        device,
    ) -> None:
        super(LSTMNet, self).__init__()

        self.device = device
        self.rnn_hidden_size = RNN_params.get("hidden_size", 128)
        self.rnn_num_layers = RNN_params.get("num_layers", 3)
        self.bidirectional = RNN_params.get("bidirectional", False)
        rnn_dropout_rate = RNN_params.get("dropout_rate", 0.1)

        self.lstm = nn.LSTM(
            audio_dim[2],
            self.rnn_hidden_size,
            self.rnn_num_layers,
            batch_first=True,
            dropout=rnn_dropout_rate if self.rnn_num_layers > 1 else 0,
            bidirectional=self.bidirectional,
        )

        self.nn_hidden_size = NN_params.get("hidden_size", 128)
        nn_dropout_rate = NN_params.get("dropout_rate", 0.1)
        activation = NN_params.get("activation", "relu")
        down_factor = NN_params.get("down_factor", 2)

        if activation == "relu":
            act_fn = nn.LeakyReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        
        self.nn = nn.Sequential(
            nn.Linear(clinical_dim[1], self.nn_hidden_size),
            act_fn,
            nn.BatchNorm1d(self.nn_hidden_size),
            nn.Dropout(nn_dropout_rate),
            nn.Linear(self.nn_hidden_size, self.nn_hidden_size),
        )

        down_factor = fusion_params.get("down_factor", 2)
        fusion_dropout_rate = fusion_params.get("dropout_rate", 0.1)
        fusion_input_dim = self.rnn_hidden_size * (1+int(self.bidirectional)) + self.nn_hidden_size
        hidden_size = fusion_input_dim // down_factor
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Dropout(fusion_dropout_rate),
            nn.Linear(hidden_size, output_dim)
        )


    def forward(self, a, c):
        h = self.init_hidden(a.size(0))
        c0 = self.init_hidden(a.size(0))
        output, (hidden, _) = self.lstm(a, (h, c0))

        if self.bidirectional:
            out_a = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            out_a = hidden[-1]

        out_c = self.nn(c)
        fusion_x = torch.cat([out_a, out_c], 1)
        output = self.fusion(fusion_x)
        return output

    def init_hidden(self, batch_size):
        return torch.zeros(
            self.rnn_num_layers * (1 + int(self.bidirectional)), batch_size, self.rnn_hidden_size, device=self.device
        )


class TransformerNet(nn.Module):
    def __init__(
        self,
        audio_dim,
        clinical_dim,
        output_dim,
        Transformer_params: dict,
        NN_params: dict,
        fusion_params: dict,
        device,
    ):
        super(TransformerNet, self).__init__()

        self.device = device

        self.d_model = Transformer_params.get("d_model", 128)
        self.nhead = Transformer_params.get("nhead", 8)
        self.num_layers = Transformer_params.get("num_layers", 6)
        self.dim_feedforward = Transformer_params.get("dim_feedforward", 512)
        self.dropout = Transformer_params.get("dropout", 0.1)
        activation = Transformer_params.get("activation", "relu")

        self.total_embedding = nn.Linear(audio_dim[2], self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )

        self.nn_hidden_size = NN_params.get("hidden_size", 128)
        nn_dropout_rate = NN_params.get("dropout_rate", 0.0)
        nn_activation = NN_params.get("activation", "relu")
        down_factor = NN_params.get("down_factor", 1)

        if nn_activation == "relu":
            nn_act_fn = nn.LeakyReLU()
        elif nn_activation == "gelu":
            nn_act_fn = nn.GELU()

        self.nn = nn.Sequential(
            nn.Linear(clinical_dim[1], self.nn_hidden_size),
            nn_act_fn,
            nn.BatchNorm1d(self.nn_hidden_size),
            nn.Dropout(nn_dropout_rate),
            nn.Linear(self.nn_hidden_size, self.nn_hidden_size),
        )

        fusion_input_dim = self.d_model + self.nn_hidden_size
        hidden_size = fusion_input_dim // down_factor
        fusion_dropout_rate = fusion_params.get("dropout_rate", 0.0)
        fusion_activation = fusion_params.get("activation", "relu")

        if fusion_activation == "relu":
            fusion_act_fn = nn.LeakyReLU()
        elif fusion_activation == "gelu":
            fusion_act_fn = nn.GELU()

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size),
            fusion_act_fn,
            nn.Linear(hidden_size, hidden_size),
            fusion_act_fn,
            nn.Dropout(fusion_dropout_rate),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, a, c, src_key_padding_mask=None):
        a_proj = self.total_embedding(a)

        output = self.transformer_encoder(
            a_proj,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch_size, seq_len, d_model)

        out_a = output.mean(dim=1)  # (batch_size, d_model)

        out_c = self.nn(c)  # (batch_size, nn_hidden_size)

        fusion_x = torch.cat([out_a, out_c], dim=1)  # (batch_size, fusion_input_dim)
        output = self.fusion(fusion_x)  # (batch_size, output_dim)
        return output