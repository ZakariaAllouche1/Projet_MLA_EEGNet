import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        s = x.mean(dim=(2, 3))
        z = F.relu(self.fc1(s))
        z = torch.sigmoid(self.fc2(z))
        z = z.view(b, c, 1, 1)
        return x * z


class EEGNetMSSE(nn.Module):
    def __init__(self, n_channels, n_times, n_classes,
                 F1_branch=4, D=2,
                 kernel_len_short=32, kernel_len_long=64,
                 dropout_p=0.5, se_reduction=4):
        super().__init__()
        F1_total = 2 * F1_branch
        F2 = D * F1_total

        self.conv_time_short = nn.Conv2d(
            in_channels=1, out_channels=F1_branch,
            kernel_size=(1, kernel_len_short),
            padding=(0, kernel_len_short // 2),
            bias=False
        )
        self.conv_time_long = nn.Conv2d(
            in_channels=1, out_channels=F1_branch,
            kernel_size=(1, kernel_len_long),
            padding=(0, kernel_len_long // 2),
            bias=False
        )
        self.bn_time = nn.BatchNorm2d(F1_total)

        self.conv_spatial = nn.Conv2d(
            in_channels=F1_total, out_channels=F1_total * D,
            kernel_size=(n_channels, 1),
            groups=F1_total,
            bias=False
        )
        self.bn_spatial = nn.BatchNorm2d(F1_total * D)
        self.se1 = SqueezeExcite(F1_total * D, reduction=se_reduction)

        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(dropout_p)

        self.conv_depthwise = nn.Conv2d(
            in_channels=F1_total * D, out_channels=F1_total * D,
            kernel_size=(1, 16),
            groups=F1_total * D,
            padding=(0, 16 // 2),
            bias=False
        )
        self.conv_pointwise = nn.Conv2d(
            in_channels=F1_total * D, out_channels=F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn_sep = nn.BatchNorm2d(F2)
        self.se2 = SqueezeExcite(F2, reduction=se_reduction)

        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(dropout_p)

        T_out = n_times // 32
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * T_out, n_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x_short = self.conv_time_short(x)
        x_long = self.conv_time_long(x)
        x = torch.cat([x_short, x_long], dim=1)
        x = self.bn_time(x)

        x = self.conv_spatial(x)
        x = self.bn_spatial(x)
        x = F.elu(x)
        x = self.se1(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv_depthwise(x)
        x = self.conv_pointwise(x)
        x = self.bn_sep(x)
        x = F.elu(x)
        x = self.se2(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        return self.classifier(x)