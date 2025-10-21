import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# originally from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
class Chomp1d(nn.Module):
    """
    Removes the last elements of a time series to ensure causal convolutions
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # 1st layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 2nd layer
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                               self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_features, num_channels, kernel_size=2, dropout=0.2):
        """
        num_features: # of input features (5 for OHLC+timestamp or 6 with volume)
        num_channels: list of integers; i-th element represents the number of channels in the i-th layer
        kernel_size: size of the convolving kernel
        dropout: dropout rate
        """
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        # self.linear = nn.Linear(num_channels[-1], 1)
        # self.init_weights()

    # def init_weights(self):
    #     self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        previous code

        # x input shape: [batch_size, sequence_length, num_features]
        # Reshape for TCN: [batch_size, num_features, sequence_length]
        # x = x.transpose(1, 2)
        # print(f"Input shape before permute: {x.shape}")

        y = self.network(x)
        # Use the last timestep's features for prediction
        out = self.linear(y[:, :, -1])
        return out
        """
        print(f"Input shape: {x.shape}")

        # new code
        return self.network(x)
        