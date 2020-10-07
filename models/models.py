import torch.nn as nn


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(in_features, in_features * 2, bias)
        self.fc2 = nn.Linear(in_features * 2, in_features)
        self.fc3 = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.drop(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
