import torch
import torch.nn as nn

class ClinicalNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(ClinicalNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
      
