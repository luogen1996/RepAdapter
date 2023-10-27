import torch
import torch.nn as nn


class RepAdapter(nn.Module):
    """ Pytorch Implemention of RepAdapter"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1.0
    ):
        super().__init__()
        r = hidden_dim
        self.conv_A = nn.Conv1d(in_features, r, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(r, in_features, 1, groups=groups, bias=True)
        self.dropout = nn.Dropout(0.1)

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

        self.groups = groups
        self.r = r
        self.scale = scale

    def forward(self, x):
        x = x.transpose(1,2)
        residual = x
        result = self.conv_A(x)
        result = self.dropout(result)
        result = self.conv_B(result)
        result = result*self.scale + residual
        result = result.transpose(1,2).contiguous()
        return result

if __name__ == "__main__":
    m = RepAdapter()
    print()