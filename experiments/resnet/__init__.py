import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from collections import OrderedDict
from .flatten_sequential import flatten_sequential

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.layers = nn.ModuleList()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers.append(self.conv1)
        self.layers.append(self.bn1)
        self.layers.append(self.relu)
        self.layers.append(self.maxpool)

        self.layers.append(self._make_layer(block, 64, layers[0]))
        self.layers.append(self._make_layer(block, 128, layers[1], stride=2))
        self.layers.append(self._make_layer(block, 256, layers[2], stride=2))
        self.layers.append(self._make_layer(block, 512, layers[3], stride=2))

        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(512 * block.expansion, num_classes))

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def build_resnet(
        layers: list[int], # [3, 4, 6, 3]
        num_classes: int, 
        inplace: bool=False
    ) -> nn.Sequential:
    # model = ResNet(ResidualBlock, [3, 4, 6, 3])
    in_channels = 64
    
    def _make_layer(
        out_channels, 
        num_blocks: int, 
        stride: int=1,
        inplace: bool=False,
    ):
        block = ResidualBlock
        nonlocal in_channels

        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU()),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ('layer1', _make_layer(64, layers[0], inplace=inplace)),
        ('layer2', _make_layer(128, layers[1], stride=2, inplace=inplace)),
        ('layer3', _make_layer(256, layers[2], stride=2, inplace=inplace)),
        ('layer4', _make_layer(512, layers[3], stride=2, inplace=inplace)),

        ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
        ('flat', nn.Flatten()),
        ('fc', nn.Linear(512 * ResidualBlock.expansion, num_classes)),
    ]))
    model = flatten_sequential(model)

     # Initialize weights for Conv2d and BatchNorm2d layers.
    def init_weight(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            return

        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            return
    
    model.apply(init_weight)
    return model

def resnet50(num_classes: int=10, inplace: bool=False):
    return build_resnet([3, 4, 6, 3], num_classes, inplace)

def resnet100(num_classes: int=10, inplace: bool=False):
    return build_resnet([3, 13, 30, 3], num_classes, inplace)


def build_train_stuffs(model: nn.Module, batch_size: int, data_dir: str):
    # 1. define network
    # 2. define dataloader
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True,
    )

    return train_loader, criterion, optimizer