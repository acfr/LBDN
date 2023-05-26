import torch
import torch.linalg as la
import numpy as np
from torchvision.transforms import Normalize
from autoattack import AutoAttack
from model import getModel
from dataset import getDataLoader
from utils import *

def evaluate_toy(config):
    seed_everything(config.seed)
    model = getModel(config).cuda() 
    txtlog = TxtLogger(config)
    xshape = (config.lip_batch_size, config.in_channels)
    x = (torch.rand(xshape) + 0.3*torch.randn(xshape)).cuda()
    model(x)
    model_state = torch.load(f"{config.train_dir}/model.ckpt")
    model.load_state_dict(model_state)
    layer_lip = np.zeros(10)
    layer_norm= np.zeros(10)
    if model.layer == 'Sandwich':
        psi0 = torch.tensor([-np.log(2)],dtype=torch.float32).to(x.device)
        At0  = torch.eye(1,dtype=torch.float32).to(x.device)
    for k in range(10):
        g = empirical_lipschitz(model.model[k], x)
        layer_lip[k] = g
        x = torch.randn_like(model.model[k](x))
        if model.layer == 'Sandwich':
            if k < 9:
                psi=model.model[k].psi
            else:
                psi=torch.tensor([np.log(2)],dtype=torch.float32).to(x.device)
            f=psi.shape[0]
            Q=model.model[k].Q
            At, B = Q[:,:f].T, Q[:,f:]
            W = 2*torch.exp(-psi).diag() @ B @ At0 @ torch.exp(psi0).diag()
            At0, psi0 = At, psi
        elif model.layer == 'Orthogon':
            W = model.model[k].Q
        elif model.layer == 'Aol':
            Weight = model.model[k].weights
            T = 1/torch.sqrt(torch.abs(Weight.T @ Weight).sum(1))
            W = model.model[k].scale * Weight * T

        _, S, _ = la.svd(W)
        w = S[0].item()
        layer_norm[k] = w
        txtlog(f"Layer: {k}, Lip: {g:.4f}, Norm: {w:4f}")

    np.savetxt(f"{config.train_dir}/layer_lip.csv", layer_lip)
    np.savetxt(f"{config.train_dir}/layer_norm.csv", layer_norm)

class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model 
        self.normalize = Normalize(mean, std)

    def forward(self, x):
        return self.model(self.normalize(x))

def evaluate(config):
    seed_everything(config.seed)
    model = getModel(config).cuda() 
    _, testLoader = getDataLoader(config)
    txtlog = TxtLogger(config)
    xshape = (1, config.in_channels, config.img_size, config.img_size)
    x = (torch.rand(xshape) + 0.3*torch.randn(xshape)).cuda()
    model(x) # allocate memory for Q param
    model_state = torch.load(f"{config.train_dir}/model.ckpt")
    model.load_state_dict(model_state)
    model(x) # update Q param
    mean = {
        'cifar10': [0.4914, 0.4822, 0.4465],
        'cifar100': [0.5071, 0.4865, 0.4409],
        'tiny_imagenet': [0.485, 0.456, 0.406]
    }[config.dataset]

    if config.normalized:
        std = {
            'cifar10': [0.2470, 0.2435, 0.2616],
            'cifar100': [0.2675, 0.2565, 0.2761],
            'tiny_imagenet': [0.229, 0.224, 0.225]
        }[config.dataset]
    else:
        std = [1.0, 1.0, 1.0]

    mu = torch.Tensor(mean)[:, None, None].cuda()
    sg = torch.Tensor(std)[:, None, None].cuda()
    aa_model = NormalizedModel(model, mean, std).cuda()

    aa_model.eval()
    for eps in [36, 72, 108]:
        attack = AutoAttack(aa_model, norm='L2', eps=eps/255.0, version='standard')
        attack.perturb = lambda x, y: attack.run_standard_evaluation(x, y, bs=config.test_batch_size)
        n, acc = 0.0, 0.0
        nbatch = len(testLoader)
        stats = np.zeros((nbatch,2))
        for batch_idx, batch in enumerate(testLoader):
            x, y = batch[0].cuda(), batch[1].cuda()
            xg = x.clone()
            xg.mul_(sg).add_(mu) # the original data
            stats[batch_idx, 0] = y.size(0)
            xd = attack.perturb(xg, y)
            yd = aa_model(xd).detach()
            correct = (yd.max(1)[1] == y)
            stats[batch_idx, 1] = correct.sum().item()  
            n += stats[batch_idx, 0]
            acc += stats[batch_idx, 1]
            txtlog(f"eps {eps:3d} | batch: {batch_idx+1:3d}/{nbatch:3d}, acc: {100*acc/n:.1f}")
        np.savetxt(f"{config.train_dir}/autoattack{eps}.csv", stats)

        
         