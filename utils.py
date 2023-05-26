import torch 
import torch.nn as nn
import torch.nn.functional as F
import importlib 
import logging
import numpy as np

class TxtLogger():
    def __init__(self, config):
        format_ = "[%(asctime)s] %(message)s"
        filename = f'{config.train_dir}/log.txt'
        f = open(filename, "a")
        logging.basicConfig(filename=filename, level=20, format=format_, datefmt='%H:%M:%S')

    def __call__(self, msg):
        print(msg)
        logging.info(msg)

    
def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_obj(obj_path: str, default_obj_path: str = ''):
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)

def getLoss(config):
    if config.loss == 'mse':
        crition = F.mse_loss
    elif config.loss == 'xent':
        crition = Xent(config.num_classes, offset=config.offset)
    elif config.loss == 'multimargin':
        crition = MultiMargin()
    return crition

class MultiMargin(nn.Module):

    def __init__(self, margin = 0.5):
        super().__init__()
        self.margin = margin 

    def __call__(self, outputs, labels):
        return F.multi_margin_loss(outputs, labels, margin=self.margin)
    
## from https://github.com/araujoalexandre/lipschitz-sll-networks
class Xent(nn.Module):

  def __init__(self, num_classes, offset=3.0/2):
    super().__init__()
    self.criterion = nn.CrossEntropyLoss()
    self.offset = (2 ** 0.5) * offset
    self.temperature = 0.25 
    self.num_classes = num_classes

  def __call__(self, outputs, labels):
    one_hot_labels = F.one_hot(labels, num_classes=self.num_classes)
    offset_outputs = outputs - self.offset * one_hot_labels
    offset_outputs /= self.temperature
    loss = self.criterion(offset_outputs, labels) * self.temperature
    return loss

def empirical_lipschitz(model, x, eps=0.05):

    norms = lambda X: X.view(X.shape[0], -1).norm(dim=1) ** 2
    gam = 0.0
    for r in range(10):
        dx = torch.zeros_like(x)
        dx.uniform_(-eps,eps)
        x.requires_grad = True
        dx.requires_grad = True
        optimizer = torch.optim.Adam([x, dx], lr=1E-1)
        iter, j = 0, 0
        LipMax = 0.0
        while j < 50:
            LipMax_1 = LipMax
            optimizer.zero_grad()
            dy = model(x + dx) - model(x)
            Lip = norms(dy) / (norms(dx) + 1e-6)
            Obj = -Lip.sum()
            Obj.backward()
            optimizer.step()
            LipMax = Lip.max().item()
            iter += 1
            j += 1
            if j >= 5:
                if LipMax < LipMax_1 + 1E-3:  
                    optimizer.param_groups[0]["lr"] /= 10.0
                    j = 0

                if optimizer.param_groups[0]["lr"] <= 1E-5:
                    break
        
        gam = max(gam, np.sqrt(LipMax))

    return gam 