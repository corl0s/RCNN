import math
import torch
import torch.nn as nn


class RCL(nn.Module):
    def __init__(self, in_channels, out_channels, steps = 4, **kwargs):
        super(RCL, self).__init__()
        self.conv = nn.Conv2d(out_channels, out_channels, bias=False, **kwargs)
        # self.bn = nn.ModuleList([nn.BatchNorm2d(in_channels) for i in range(steps)])
        
        self.lrn = nn.LocalResponseNorm(2)
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.in_channels = in_channels

        self.shortcut = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)


        #init the parameter    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.LocalResponseNorm):
               pass 
    
    def forward(self, x):
        # print(self.in_channels)
        rx = x
        for i in range(self.steps):
            if i == 0:
                z = self.shortcut(x)
            else:
                # print("x_shape",x.shape)
                # print("rx shape", rx.shape)
                z = self.conv(x) + self.shortcut(rx)
            
            x = self.relu(z)
            x = self.lrn(x)
            # x = self.bn[i](x)
            # print(x.shape)
        return x