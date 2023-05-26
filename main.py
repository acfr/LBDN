import os
import warnings
from argparse import ArgumentParser
from train import *
from evaluate import *
warnings.filterwarnings("ignore")

def main(args):

    config = args 

    config.lip_batch_size = 64
    config.print_freq = 10
    config.save_freq = 5
    
    if config.dataset == 'cifar10':
        config.in_channels = 3
        config.img_size = 32
        config.num_classes = 10
    elif config.dataset == 'cifar100':
        config.in_channels = 3
        config.img_size = 32
        config.num_classes = 100
    elif config.dataset == 'tiny_imagenet':
        config.in_channels = 3
        config.img_size = 64
        config.num_classes = 200
    elif config.dataset == 'square_wave':
        config.in_channels = 1
        config.img_size = 0
        config.num_classes = 1
        config.train_batch_size = 50
        config.test_batch_size = 200
        config.num_workers = 0

    if config.model == 'DNN':
        config.layer = 'Sandwich'
        config.scale = 'small'
        config.LLN = True
        config.normalized = False
        config.loss = 'xent'
    elif config.model == 'KWL':
        if config.layer is None:
            config.layer = 'Plain'
        if config.scale is None:
            config.scale = 'small'
        config.width = {
            'small': 1,
            'medium': 2,
            'large': 4
        }[config.scale]
        if config.layer == 'Plain':
            config.gamma = None
    elif config.model == 'Resnet':
        config.layer = 'SLL'
        if config.scale is None:
            config.scale = 'small'
        if config.scale == 'small':
            config.depth_conv   = 20
            config.n_channels   = 45
            config.conv_size    = 5
            config.depth_linear = 7
            config.n_features   = 2048
        elif config.scale == 'medium':
            config.depth_conv   = 30
            config.n_channels   = 60
            config.conv_size    = 5
            config.depth_linear = 10
            config.n_features   = 2048
        elif config.scale == 'large':
            config.depth_conv   = 90
            config.n_channels   = 60
            config.conv_size    = 5
            config.depth_linear = 15
            config.n_features   = 2048
        elif config.scale == 'xlarge':
            config.depth_conv   = 120
            config.n_channels   = 70
            config.conv_size    = 5
            config.depth_linear = 15
            config.n_features   = 4096
    elif config.model == 'Toy':
        config.loss = 'mse'
        if config.scale is None:
            config.scale = 'small'
        if config.layer is None:
            config.layer = 'Plain'

    if config.loss == 'xent':
        config.offset = 1.5

    if config.gamma is None:
        config.train_dir = f"{config.root_dir}_seed{config.seed}/{config.dataset}/{config.model}-{config.layer}-{config.scale}"
    elif config.LLN:
        config.train_dir = f"{config.root_dir}_seed{config.seed}/{config.dataset}/{config.model}-{config.layer}-{config.scale}-LLN-gamma{config.gamma:.1f}"
    else:
        config.train_dir = f"{config.root_dir}_seed{config.seed}/{config.dataset}/{config.model}-{config.layer}-{config.scale}-gamma{config.gamma:.1f}"

    os.makedirs("./data", exist_ok=True)
    os.makedirs(config.train_dir, exist_ok=True)
    if config.mode == 'train':
        if config.model == 'Toy':
            train_toy(config)
        else:
            train(config)
    elif config.mode == 'eval':
        if config.model == 'Toy':
            evaluate_toy(config)
        else:
            evaluate(config)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('-m', '--model', type=str, default='Resnet',
                        help="[DNN, KWL, Resnet, Toy]")
    parser.add_argument('-d', '--dataset', type=str, default='tiny_imagenet',
                        help="dataset [cifar10, cifar100, tiny_imagenet, square_wave]")
    parser.add_argument('-g', '--gamma', type=float, default=1.0,
                        help="Network Lipschitz bound")
    parser.add_argument('-s', '--seed', type=int, default=123)
    parser.add_argument('-e','--epochs', type=int, default=100)

    parser.add_argument('--layer', type=str, default='SLL')
    parser.add_argument('--scale', type=str, default='xlarge')
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--loss', type=str, default='xent')
    parser.add_argument('--root_dir', type=str, default='./saved_models')
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--LLN', action='store_true')
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--cert_acc', action='store_true')
    
    args = parser.parse_args()

    main(args)