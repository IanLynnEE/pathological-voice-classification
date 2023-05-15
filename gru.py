import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUNet(nn.Module):
    def __init__(
        self, 
        audio_input_dim,
        clinical_input_dim,
        output_dim,
        RNN_params: dict,
        NN_params: dict,
        fusion_params: dict,
        device='cpu',
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
            audio_input_dim,
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
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        
        self.nn = nn.Sequential(
            nn.Linear(clinical_input_dim, self.nn_hidden_size),
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

    def forward(self, Xs):
        x_audio, x_clinical = Xs

        h = self.init_hidden(x_audio.size(0))
        output, hidden = self.gru(x_audio, h)
        h_gru = hidden[-1:]
        h_gru = h_gru.squeeze(0)
        output = self.nn(x_clinical)
        fusion_x = torch.cat([h_gru, output], 1)
        output = self.fusion(fusion_x)

        return output

    def init_hidden(self, batch_size):
        return torch.zeros(
            self.rnn_num_layers, batch_size, self.rnn_hidden_size, device=self.device
        )
