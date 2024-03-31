import torch
import torch.nn as nn
from rcl import RCL


architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # conv_out = self.conv(x)
        # rcl = conv_out + x
        return self.leakyrelu(self.batchnorm(self.conv(x)))

# class RCLUnit(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
#         super(RCLUnit, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

#     def forward(self, u, x_prev, w_f, w_r, b, **kwargs):
#         u_input = self.conv(u)
#         x_input = self.conv(x_prev)
#         net_input = torch.matmul(u_input.view(u.size(0), -1), w_f) + torch.matmul(x_input.view(x_prev.size(0), -1), w_r) + b
#         return net_input


# class RCLLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
#         super(RCLLayer, self).__init__()
#         self.rcl_unit = RCLUnit(in_channels, out_channels, kernel_size)

#     def forward(self, u, x_prev, w_f, w_r, b):
#         return self.rcl_unit(u, x_prev, w_f, w_r, b)




    
# rcl_layer = RCLLayer(in_channels, out_channels, kernel_size)


class Rolov1(nn.Module):
    def __init__(self, in_channels=3, steps=4,**kwargs):
        super(Rolov1, self).__init__()
        self.architecture = architecture_config
        self.steps = steps
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        steps = self.steps
        print("Creating Rolov1 instance with steps =", steps)

        for i,x in enumerate(architecture):
            # print(i)
            # print(x)
            if type(x) == tuple:
                if (i == 0) or (i == 15):
                    # print(x)
                    layers += [
                        # RCL(
                        #     in_channels, x[1],steps, kernel_size=x[0], stride=x[2], padding=x[3],
                        # )
                        CNNBlock(
                            in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                        )
                    ]
                else:
                    layers += [
                        RCL(
                            in_channels, x[1],steps, kernel_size=x[0], stride=x[2], padding=x[3],
                        )
                        # CNNBlock(
                        #     in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                        # )
                    ]
                    
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        RCL(
                            in_channels,
                            conv1[1],
                            steps,
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                        # CNNBlock(
                        #     in_channels,
                        #     conv1[1],
                        #     kernel_size=conv1[0],
                        #     stride=conv1[2],
                        #     padding=conv1[3],
                        # )
                    ]
                    layers += [
                        RCL(
                            conv1[1],
                            conv2[1],
                            steps,
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                        # CNNBlock(
                        #     conv1[1],
                        #     conv2[1],
                        #     kernel_size=conv2[0],
                        #     stride=conv2[2],
                        #     padding=conv2[3],
                        # )
                    ]
                    in_channels = conv2[1]
        print(layers)
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        
        # In the github page it is like below
        # nn.Linear(1024*S*S, 496),
        # nn.LeakyReLU(0.1),
        # nn.Linear(496, S*S*(B*5+C))

        # In original paper this should be
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)),
        )
        
# class CNNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(CNNBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
#         self.batchnorm = nn.BatchNorm2d(out_channels)
#         self.leakyrelu = nn.LeakyReLU(0.1)

#     def forward(self, x):
#         conv_out = self.conv(x)
#         rcl = conv_out + x
#         return self.leakyrelu(self.batchnorm(rcl))



# class Yolov1(nn.Module):
#     def __init__(self, in_channels=3, **kwargs):
#         super(Yolov1, self).__init__()
#         self.architecture = architecture_config
#         self.in_channels = in_channels
#         self.darknet = self._create_conv_layers(self.architecture)
#         self.fcs = self._create_fcs(**kwargs)

#     def forward(self, x):
#         x = self.darknet(x)
#         return self.fcs(torch.flatten(x, start_dim=1))

#     def _create_conv_layers(self, architecture):
#         layers = []
#         in_channels = self.in_channels

#         for x in architecture:
#             if type(x) == tuple:
#                 layers += [
#                     CNNBlock(
#                         in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
#                     )
#                 ]
#                 in_channels = x[1]

#             elif type(x) == str:
#                 layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

#             elif type(x) == list:
#                 conv1 = x[0]
#                 conv2 = x[1]
#                 num_repeats = x[2]

#                 for _ in range(num_repeats):
#                     layers += [
#                         CNNBlock(
#                             in_channels,
#                             conv1[1],
#                             kernel_size=conv1[0],
#                             stride=conv1[2],
#                             padding=conv1[3],
#                         )
#                     ]
#                     layers += [
#                         CNNBlock(
#                             conv1[1],
#                             conv2[1],
#                             kernel_size=conv2[0],
#                             stride=conv2[2],
#                             padding=conv2[3],
#                         )
#                     ]
#                     in_channels = conv2[1]

#         return nn.Sequential(*layers)

#     def _create_fcs(self, split_size, num_boxes, num_classes):
#         S, B, C = split_size, num_boxes, num_classes

        
#         # In the github page it is like below
#         # nn.Linear(1024*S*S, 496),
#         # nn.LeakyReLU(0.1),
#         # nn.Linear(496, S*S*(B*5+C))

#         # In original paper this should be
#         return nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(1024 * S * S, 4096),
#             nn.Dropout(0.0),
#             nn.LeakyReLU(0.1),
#             nn.Linear(4096, S * S * (C + B * 5)),
#         )
        
# def test(S=7,B=2,C=20):
#     model = Yolov1(split_size = S,num_boxes = B,num_classes=C)
#     x = torch.randn((2,3,448,448))
#     print(model(x).shape)

# test()



# class Rolov1(nn.Module):
#     def __init__(self, in_channels=3, **kwargs):
#         super(Rolov1, self).__init__()
#         self.architecture = architecture_config
#         self.in_channels = in_channels
#         self.darknet = self._create_conv_layers(self.architecture)
#         self.fcs = self._create_fcs(**kwargs)

#     def forward(self, x):
#         x = self.darknet(x)
#         return self.fcs(torch.flatten(x, start_dim=1))

#     def _create_conv_layers(self, architecture):
#         layers = []
#         in_channels = self.in_channels

#         for x in architecture:
#             if type(x) == tuple:
#                 layers += [
#                     RCLLayer(in_channels, x[1], x[0], stride=x[2], padding=x[3])
#                 ]
#                 in_channels = x[1]

#             elif type(x) == str:
#                 layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

#             elif type(x) == list:
#                 conv1 = x[0]
#                 conv2 = x[1]
#                 num_repeats = x[2]

#                 for _ in range(num_repeats):
#                     layers += [
#                         RCLLayer(in_channels, conv1[1], conv1[0], stride=conv1[2], padding=conv1[3])
#                         # CNNBlock(
#                         #     in_channels,
#                         #     conv1[1],
#                         #     kernel_size=conv1[0],
#                         #     stride=conv1[2],
#                         #     padding=conv1[3],
#                         # )
#                     ]
#                     layers += [
#                         RCLLayer(in_channels, conv2[1], conv2[0], stride=conv2[2], padding=conv2[3])
#                         # CNNBlock(
#                         #     conv1[1],
#                         #     conv2[1],
#                         #     kernel_size=conv2[0],
#                         #     stride=conv2[2],
#                         #     padding=conv2[3],
#                         # )
#                     ]
#                     in_channels = conv2[1]

#         return nn.Sequential(*layers)

#     def _create_fcs(self, split_size, num_boxes, num_classes):
#         S, B, C = split_size, num_boxes, num_classes

        
#         # In the github page it is like below
#         # nn.Linear(1024*S*S, 496),
#         # nn.LeakyReLU(0.1),
#         # nn.Linear(496, S*S*(B*5+C))

#         # In original paper this should be
#         return nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(1024 * S * S, 4096),
#             nn.Dropout(0.0),
#             nn.LeakyReLU(0.1),
#             nn.Linear(4096, S * S * (C + B * 5)),
#         )
        