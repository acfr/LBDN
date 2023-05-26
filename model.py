import torch 
import torch.nn as nn 
from layer import *

def getModel(config):
    models = {
        'DNN': DNN,
        'KWL': KWL,
        'Resnet': Resnet,
        'Toy': Toy
    }[config.model]
    return models(config)

class DNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.LLN = config.LLN
        self.layer = config.layer
        self.scale = config.scale
        g = self.gamma ** 0.5
        if self.img_size == 32:
            n = 32 // 8
            layer = [
                SandwichConv(self.in_channels, 64, 3, scale=g),
                SandwichConv(64, 64, 3, stride=2),
                SandwichConv(64, 128, 3),
                SandwichConv(128, 128, 3, stride=2),
                SandwichConv(128, 256, 3),
                SandwichConv(256, 256, 3, stride=2),
                nn.Flatten(),
                SandwichFc(256 * n * n, 2048),
                SandwichFc(2048, 2048)
            ]
        elif self.img_size == 64:
            n = 64 // 16
            layer = [
                SandwichConv(self.in_channels, 64, 3, scale=g),
                SandwichConv(64, 64, 3, stride=2),
                SandwichConv(64, 128, 3),
                SandwichConv(128, 128, 3, stride=2),
                SandwichConv(128, 256, 3),
                SandwichConv(256, 256, 3, stride=2),
                SandwichConv(256, 512, 3),
                SandwichConv(512, 512, 3, stride=2),
                nn.Flatten(),
                SandwichFc(512 * n * n, 2048),
                SandwichFc(2048, 2048)
            ]
        if self.LLN:
            layer.append(SandwichFc(2048, 1024, scale=g))
            layer.append(LinearNormalized(1024, self.num_classes))
        else:
            layer.append(SandwichFc(2048, 1024))
            layer.append(SandwichLin(1024, self.num_classes, scale=g))

        self.model = nn.Sequential(*layer)

    def forward(self, x):
        return self.model(x)


#------------------------------------------------------------
class KWL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.LLN = config.LLN
        self.width = config.width
        self.layer = config.layer

        w = self.width
        if self.gamma is not None:
            g = self.gamma ** (1.0 / 2)
        n = self.img_size // 4

        if self.layer == 'Plain':
            self.model = nn.Sequential(
                PlainConv(self.in_channels, 32 * w, 3), nn.ReLU(),
                PlainConv(32 * w, 32 * w, 3, stride=2), nn.ReLU(),
                PlainConv(32 * w, 64 * w, 3), nn.ReLU(),
                PlainConv(64 * w, 64 * w, 3, stride=2), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * n * n * w, 640 * w), nn.ReLU(),
                nn.Linear(640 * w, 512 * w), nn.ReLU(),
                nn.Linear(512 * w, self.num_classes)
            )
        elif self.layer == 'Sandwich':
            self.model = nn.Sequential(
                SandwichConv(self.in_channels, 32 * w, 3, scale=g), 
                SandwichConv(32 * w, 32 * w, 3, stride=2), 
                SandwichConv(32 * w, 64 * w, 3), 
                SandwichConv(64 * w, 64 * w, 3, stride=2), 
                nn.Flatten(),
                SandwichFc(64 * n * n * w, 512 * w), 
                SandwichFc(512 * w, 512 * w), 
                SandwichLin(512 * w, self.num_classes, scale=g)
            )
        elif self.layer == 'Orthogon':
            self.model = nn.Sequential(
                    OrthogonConv(self.in_channels, 32 * w, 3, scale=g), 
                    OrthogonConv(32 * w, 32 * w, 3, stride=2), 
                    OrthogonConv(32 * w, 64 * w, 3), 
                    OrthogonConv(64 * w, 64 * w, 3, stride=2), 
                    nn.Flatten(),
                    OrthogonFc(64 * n * n * w, 640 * w), 
                    OrthogonFc(640 * w, 512 * w), 
                    OrthogonLin(512 * w, self.num_classes, scale=g)
                ) 
        elif self.layer == 'Aol':
            self.model = nn.Sequential(
                AolConv(self.in_channels, 32 * w, 3, scale=g), 
                AolConv(32 * w, 32 * w, 3), 
                AolConv(32 * w, 64 * w, 3), 
                AolConv(64 * w, 64 * w, 3), 
                nn.AvgPool2d(4, divisor_override=4),
                nn.Flatten(),
                AolFc(64 * n * n * w, 640 * w), 
                AolFc(640 * w, 512 * w), 
                AolLin(512 * w, self.num_classes, scale=g)
            )  

    def forward(self, x):
        return self.model(x)
    
# #---------------------------------------------------------------------------
class Resnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gamma = config.gamma
        self.depth_conv = config.depth_conv
        self.n_channels = config.n_channels
        self.depth_linear = config.depth_linear
        self.n_features = config.n_features
        self.conv_size = config.conv_size
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.LLN = config.LLN

        if self.gamma is None: 
            g = 1.0
        else:
            g = self.gamma ** (1.0/2)

        layers = []
        self.conv1 = PaddingChannels(self.in_channels, self.n_channels, scale=g)   
        for _ in range(self.depth_conv):
            layers.append(SLLBlockConv(self.n_channels, self.conv_size, 3))
        layers.append(nn.AvgPool2d(4, divisor_override=4))
            
        self.stable_block = nn.Sequential(*layers)

        layers_linear = [nn.Flatten()]

        in_features = self.n_channels * ((self.img_size // 4) ** 2)

        for _ in range(self.depth_linear):
            layers_linear.append(SLLBlockFc(in_features, self.n_features))
        self.layers_linear = nn.Sequential(*layers_linear)

        if self.LLN:
            self.last_last = LinearNormalized(in_features, self.num_classes, scale=g)
        else:
            self.last_last = FirstChannel(self.num_classes, scale=g)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stable_block(x)
        x = self.layers_linear(x)
        x = self.last_last(x)
        return x
    
class Toy(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma 
        self.layer = config.layer
        g = self.gamma ** 0.5

        if self.layer == 'Plain':
            self.gamma = None
            w = 128
            self.model = nn.Sequential(
                nn.Linear(self.in_channels, w), nn.ReLU(),
                nn.Linear(w, w), nn.ReLU(),
                nn.Linear(w, w), nn.ReLU(),
                nn.Linear(w, w), nn.ReLU(),
                nn.Linear(w, w), nn.ReLU(),
                nn.Linear(w, w), nn.ReLU(),
                nn.Linear(w, w), nn.ReLU(),
                nn.Linear(w, w), nn.ReLU(),
                nn.Linear(w, w), nn.ReLU(),
                nn.Linear(w, self.num_classes)
            )
        elif self.layer == 'Sandwich':
            w = 86
            self.model = nn.Sequential(
                SandwichFc(self.in_channels, w, scale=g),
                SandwichFc(w, w),
                SandwichFc(w, w),
                SandwichFc(w, w),
                SandwichFc(w, w),
                SandwichFc(w, w),
                SandwichFc(w, w),
                SandwichFc(w, w),
                SandwichFc(w, w),
                SandwichLin(w, self.num_classes, scale=g)
            )
        elif self.layer == 'Orthogon':
            w = 128
            self.model = nn.Sequential(
                OrthogonFc(self.in_channels, w, scale=g),
                OrthogonFc(w, w),
                OrthogonFc(w, w),
                OrthogonFc(w, w),
                OrthogonFc(w, w),
                OrthogonFc(w, w),
                OrthogonFc(w, w),
                OrthogonFc(w, w),
                OrthogonFc(w, w),
                OrthogonLin(w, self.num_classes, scale=g)
            )
        elif self.layer == 'Aol':
            w = 128
            self.model = nn.Sequential(
                AolFc(self.in_channels, w, scale=g),
                AolFc(w, w),
                AolFc(w, w),
                AolFc(w, w),
                AolFc(w, w),
                AolFc(w, w),
                AolFc(w, w),
                AolFc(w, w),
                AolFc(w, w),
                AolLin(w, self.num_classes, scale=g)
            )
        elif self.layer == 'SLL':
            w = 128
            self.model = nn.Sequential(
                PaddingFeatures(self.in_channels, w, scale=g),
                SLLBlockFc(w, w),
                SLLBlockFc(w, w),
                SLLBlockFc(w, w),
                SLLBlockFc(w, w),
                SLLBlockFc(w, w),
                SLLBlockFc(w, w),
                SLLBlockFc(w, w),
                SLLBlockFc(w, w),
                FirstChannel(self.num_classes, scale=g)
            )

    def forward(self, x):
        return self.model(x)