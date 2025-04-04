import torch 
import torch.nn as nn
import numpy as np
class EncodeLinear(nn.Module):
    def __init__(self, dim_in, dim_out=32, head_name='lstm', device=None):
        super(EncodeLinear, self).__init__()

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.features = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(inplace=True)
        ).to(self.device)  # Đưa model lên device

        self.head_name = head_name
        self.feature_dim = dim_out
        self.train()

    def forward(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)

        x = inputs.to(self.device)  # Đưa input lên đúng device
        feature = self.features(x)
        return feature