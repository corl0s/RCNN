import torch
import torch.nn as nn

lrn = nn.LocalResponseNorm(2)
signal_2d = torch.randn(16, 3, 448, 448)
signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
output_2d = lrn(signal_2d)
output_4d = lrn(signal_4d)

print(output_2d.shape)


conv = nn.Conv2d(in_channels=3,out_channels= 3, kernel_size=3,stride=2,padding=1, bias=False)
conv2 = nn.Conv2d(in_channels=3,out_channels= 3, kernel_size=3,stride=2,padding=1, bias=False)
z = conv(signal_2d)
x = conv2(z)
print(z+x)


