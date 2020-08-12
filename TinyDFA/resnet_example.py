# https://github.com/FrancescoSaverioZuppichini/ResNet/blob/master/ResNet.ipynb

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import json

from tinydfa import DFA, DFALayer, FeedbackPointsHandling

import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter  # somehow causes a dependency issue.

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


from collections import OrderedDict

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

from collections import OrderedDict
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({
            'conv': conv(in_channels, out_channels, *args, **kwargs), 
            'bn': nn.BatchNorm2d(out_channels)
        }))


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation(),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock, dfas=None, *args, **kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])

        self.dfas = dfas
        # import pdb; pdb.set_trace()
        
    def forward(self, x):
        x = self.gate(x)
        for index, block in enumerate(self.blocks):
            x = block(x)
            if self.dfas:
                x = self.dfas[index](x)
        return x

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.training_method = None
        if 'training_method' in kwargs:
            self.training_method = kwargs['training_method']
        self.use_dfa = self.training_method in ['DFA', 'SHALLOW']
        if self.use_dfa:
            dfas = [DFALayer() for _ in range(len(kwargs['deepths']))]  # should I grab these from the EncoderLayers instead?
            self.dfa = DFA(dfas, feedback_points_handling=FeedbackPointsHandling.LAST,
                           no_training=(self.training_method == 'SHALLOW'))
        else:
            dfas = None

        self.encoder = ResNetEncoder(in_channels, dfas=dfas, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        if self.use_dfa:
            x = self.dfa(x)
        x = self.decoder(x)
        return x


def resnet18(in_channels, n_classes, **kwargs):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2], **kwargs)

def resnet34(in_channels, n_classes, **kwargs):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3], **kwargs)

def resnet50(in_channels, n_classes, **kwargs):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3], **kwargs)

def resnet101(in_channels, n_classes, **kwargs):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 23, 3], **kwargs)

def resnet152(in_channels, n_classes, **kwargs):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 8, 36, 3], **kwargs)


def train(args, train_loader, model, optimizer, device, epoch, writer):
    model.train()
    correct = 0
    n_seen = 0
    for b, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        n_seen += data.shape[0]
        loss.backward()
        optimizer.step()
        if b % 10 == 0:
            global_step = epoch * len(train_loader) + b
            # print(f"Training loss at batch {b}: {loss.item():.4f}", end='\r')
            accuracy = correct / n_seen * 100
            print(f"Training loss at batch {b}: {loss.item():.4f}, accuracy {accuracy:.2f}%.")
            if writer:
                writer.add_scalar('train/epoch', epoch, global_step)
                writer.add_scalar('train/step_in_epoch', b * len(train_loader), global_step)
                # writer.add_scalar('train/avg_accuracy_pct', accuracy_pct, global_step)  # TODO add
                writer.add_scalar('train/loss', loss.item(), global_step)


def test(args, test_loader, model, device, epoch, writer):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for b, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset) * 100
    print(f"Epoch {epoch}: test loss {test_loss:.4f}, accuracy {accuracy:.2f}%.")
    if writer:
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/accuracy', accuracy, epoch)


def load_cifar(args, use_cuda, do_grayscale=False, no_normalize=False):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # 1. Loading and normalizing CIFAR{10,100}

    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].

    mytransformations = [
        transforms.ToTensor(),
        transforms.Normalize(
            # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            (0.49139765, 0.48215759, 0.44653141),
            (0.24703199, 0.24348481, 0.26158789),
            # from https://github.com/kuangliu/pytorch-cifar/issues/8
            # WARNING: Values for CIFAR10 only!!!
        ),
    ]
    if no_normalize:
        mytransformations.pop()
    if do_grayscale:
        mytransformations = [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5), (0.5, 0.5)),
        ]
    transform = transforms.Compose(mytransformations)

    if args.dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100
        classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                   'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
    else:
        raise ValueError('bad dataset...')
    trainset = dataset(root='./data', train=True,
                       download=True, transform=transform)
    testset = dataset(root='./data', train=False,
                      download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return trainloader, testloader, classes


def main(args):
    # writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'), flush_secs=30)
    writer = None

    use_gpu = not args.no_gpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu_id}" if use_gpu else "cpu")
    torch.manual_seed(args.seed)

    gpu_args = {'num_workers': 12, 'pin_memory': True} if use_gpu else {}
    # MNIST loaders...
    # mnist_transform = transforms.Compose([transforms.ToTensor(),
    #                                                 transforms.Normalize((0.1307,), (0.3081,))])
    # train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.dataset_path, train=True, download=True,
    #                                                                       transform=mnist_transform),
    #                                            batch_size=args.batch_size, shuffle=True, **gpu_args)
    # test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.dataset_path, train=False,
    #                                                                      transform=mnist_transform),
    #                                           batch_size=args.test_batch_size, shuffle=True, **gpu_args)
    # in_channels = 1
    # classes = list(range(10))

    train_loader, test_loader, classes = load_cifar(
        args, use_gpu, do_grayscale=False, no_normalize=False)
    in_channels = 3

    model = resnet18(in_channels=in_channels, n_classes=len(classes), training_method=args.training_method).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, model, optimizer,
              device, epoch, writer=writer)
        test(args, test_loader, model, device, epoch, writer=writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tiny-DFA MNIST Example')
    parser.add_argument('-t', '--training-method', type=str, choices=['BP', 'DFA', 'SHALLOW'], default='DFA',
                        metavar='T', help='training method to use, choose from backpropagation (BP), direct feedback '
                                          'alignment (DFA), or only topmost layer (SHALLOW) (default: DFA)')

    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='B',
                        help='training batch size (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='B',
                        help='testing batch size (default: 1000)')

    parser.add_argument('--hidden-size', type=int, default=256, metavar='H',
                        help='hidden layer size (default: 256)')

    parser.add_argument('-e', '--epochs', type=int, default=15, metavar='E',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01, metavar='LR',
                        help='SGD learning rate (default: 0.01)')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables GPU training')
    parser.add_argument('-g', '--gpu-id', type=int, default=0, metavar='i',
                        help='id of the gpu to use (default: 0)')
    parser.add_argument('-s', '--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 0)')
    parser.add_argument('-p', '--dataset-path', type=str, default='/data', metavar='P',
                        help='path to dataset (default: /data)')
    parser.add_argument("--output_dir",
                        default="[PT_OUTPUT_DIR]",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--dataset', default='cifar10',
                        help='which dataset to use (default cifar10). [cifar10 | cifar100] ')
    args = parser.parse_args()
    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', 'tmp'))
    os.makedirs(args.output_dir, exist_ok=True)
    print(args)
    # dump args to args.json
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    main(args)

