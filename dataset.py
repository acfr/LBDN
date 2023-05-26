import os
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg
from utils import load_obj

def getDataLoader(config):
    loaders = {
        'cifar10': cifar10_loaders,
        'cifar100': cifar100_loaders,
        'tiny_imagenet': tiny_imagenet_loaders,
        'square_wave': square_wave_loaders
    }[config.dataset]
    return loaders(config)

## square-wave ----------------------------------------------------------
def square_wave_loaders(config):
    fpath = "./data/square-wave.pt"
    train_size, test_size = 300, 200
    if os.path.isfile(fpath):
        data = torch.load(fpath)
    else:
        x, y = square_wave(train_size)
        xt, yt = square_wave(test_size, xrand=False)
        data = {"xt":xt, "yt":yt, "x":x, "y":y}
        torch.save(data, fpath)

    train_dataset = Curve(data['x'], data['y'])
    test_dataset  = Curve(data['xt'], data['yt'])
        
    trainLoader = DataLoader(train_dataset,batch_size=config.train_batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)

    testLoader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, pin_memory=True, num_workers=config.num_workers)

    return trainLoader, testLoader

def square_wave(size, xrand=True):
        if xrand:
            x = 2*(2*torch.rand(size,1) - 1)
        else:
            x = torch.linspace(-2.0, 2.0, size).reshape((size,1))
        y = torch.zeros((size,1))
        for i in range(size):
            if x[i, 0] <= -1.0:
                y[i, 0] = 1.0
            if x[i, 0] > 0.0 and x[i, 0] <= 1.0:
                y[i, 0] = 1.0
        return x, y

class Curve(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = x.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def tiny_imagenet_loaders(config):
    
    mean = [0.485, 0.456, 0.406]
    if config.normalized:
        std = [0.229, 0.224, 0.225]
    else:
        std = [1.0, 1.0, 1.0]

    normalize = transforms.Normalize(mean=mean, std=std)
    hue = 0.02
    saturation = (.3, 2.)
    brightness = 0.1
    contrast = (.5, 2.)

    transforms_list = [transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=brightness, contrast=contrast, 
                    saturation=saturation, hue=hue),
                transforms.ToTensor(),
                normalize]

    train_dset = TinyImageNet('data',
                              split='train',
                              download=True,
                              transform=transforms.Compose(transforms_list))
    test_dset = TinyImageNet('data',
                             split='val',
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize
                             ]))

    trainLoader = DataLoader(train_dset, batch_size=config.train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=config.num_workers)

    testLoader = DataLoader(test_dset, batch_size=config.test_batch_size, shuffle=False, pin_memory=True,num_workers=config.num_workers)

    return trainLoader, testLoader

##------------------------------------------------------------------------------------------
## from https://github.com/araujoalexandre/lipschitz-sll-networks
class TinyImageNet(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_integrity():
            # print('Files already downloaded and verified.')
            pass
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        if not os.path.isdir(self.dataset_path):
            print('Extracting...')
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx)

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)
    
def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images
##------------------------------------------------------------------------------------------

def cifar100_loaders(config):
    
    mean = [0.5071, 0.4865, 0.4409]
    if config.normalized:
        std = [0.2675, 0.2565, 0.2761]
    else:
        std = [1.0, 1.0, 1.0] 

    normalize = transforms.Normalize(mean=mean, std=std)
    hue = 0.02
    saturation = (.3, 2.)
    brightness = 0.1
    contrast = (.5, 2.)

    transforms_list = [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=brightness, contrast=contrast, 
                saturation=saturation, hue=hue),
            transforms.ToTensor(),
            normalize]

    train_dset = CIFAR100('data',train=True,download=True, transform=transforms.Compose(transforms_list))
    test_dset = CIFAR100('data', train=False,transform=transforms.Compose([transforms.ToTensor(),normalize]))

    trainLoader = DataLoader(train_dset, batch_size=config.train_batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=config.num_workers)

    testLoader = DataLoader(test_dset, batch_size=config.test_batch_size, shuffle=False, pin_memory=False,num_workers=config.num_workers)

    return trainLoader, testLoader

def cifar10_loaders(config):
    
    mean = [0.4914, 0.4822, 0.4465]
    if config.normalized:
        std  = [0.2470, 0.2435, 0.2616]
    else:
        std = [1.0, 1.0, 1.0]

    normalize = transforms.Normalize(mean=mean, std=std)
    hue = 0.02
    saturation = (.3, 2.)
    brightness = 0.1
    contrast = (.5, 2.)

    transforms_list = [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=brightness, contrast=contrast, 
                saturation=saturation, hue=hue),
            transforms.ToTensor(),
            normalize]

    train_dset = CIFAR10('data',train=True,download=True,transform=transforms.Compose(transforms_list))
    test_dset = CIFAR10('data',train=False,transform=transforms.Compose([transforms.ToTensor(),normalize]))

    trainLoader = DataLoader(train_dset, batch_size=config.train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=config.num_workers)

    testLoader = DataLoader(test_dset, batch_size=config.test_batch_size, shuffle=False, pin_memory=True, num_workers=config.num_workers)

    return trainLoader, testLoader

def mnist_loaders(config):

    mean = (0.1307,)
    if config.normalized:
        std = (0.3081,)
    else:
        std = (1.0, )

    trainLoader = DataLoader(MNIST('data',train=True,download=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                        ])),
                    batch_size=config.train_batch_size,
                    shuffle=True, pin_memory=True, num_workers=config.num_workers)

    testLoader = DataLoader(MNIST('data',
                    train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])),
                batch_size=config.test_batch_size,
                shuffle=False, pin_memory=True, num_workers=config.num_workers)
    
    return trainLoader, testLoader
    

