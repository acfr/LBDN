import torch 
import numpy as np
import time
from model import getModel
from dataset import getDataLoader
from utils import *

def train(config):
    
    seed_everything(config.seed)
    trainLoader, testLoader = getDataLoader(config)
    model = getModel(config).cuda()
    criterion = getLoss(config)

    txtlog = TxtLogger(config)
    txtlog(vars(config))

    txtlog(f"Set global seed to {config.seed:d}")
    
    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])

    if nparams >= 1000000:
        txtlog(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-6*nparams:.1f}M")
    else:
        txtlog(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-3*nparams:.1f}K")
    
    Epochs = config.epochs
    Lr = config.lr
    steps_per_epoch = len(trainLoader)
    PrintFreq = config.print_freq
    LLN = config.LLN
    gamma = config.gamma

    opt = torch.optim.Adam(model.parameters(), lr=Lr, weight_decay=0)
    lr_schedule = lambda t: np.interp([t], [0, Epochs*2//5, Epochs*4//5, Epochs], [0, Lr, Lr/20.0, 0])[0]

    tloss_step, tacc_step, lr_step = [], [], []
    ttime_epoch,vtime_epoch,tloss_epoch,vloss_epoch,tacc_epoch, vacc_epoch = [],[],[],[],[],[]
    vacc_epoch,acc36_epoch,acc72_epoch,acc108_epoch = [], [], [], []
    for epoch in range(Epochs):
        ## train_step
        start = time.time()
        n, Loss, Acc = 0, 0.0, 0.0
        model.train()
        for batch_idx, batch in enumerate(trainLoader):
            x, y = batch[0].cuda(), batch[1].cuda()
            lr = lr_schedule(epoch + (batch_idx+1)/steps_per_epoch)
            opt.param_groups[0].update(lr=lr)

            yh = model(x)
            J = criterion(yh, y)

            opt.zero_grad()
            J.backward()
            opt.step()

            loss = J.item()
            n += y.size(0)
            Loss += loss * y.size(0)
            acc = (yh.max(1)[1] == y).sum().item()
            acc = 100*acc / y.size(0)
            Acc += acc * y.size(0)
            
            tloss_step.append(loss)
            tacc_step.append(acc)
            lr_step.append(lr)

            if (batch_idx+1) % PrintFreq == 0:
                print(f"Epoch: {epoch+1:3d} | {batch_idx+1:3d}/{steps_per_epoch}, acc: {Acc/n:.1f}, loss: {Loss/n:.2f}, lr: {100*lr:.3f}", end='\r', flush=True)
        
        train_time = time.time()-start 
        train_loss = Loss/n 
        train_acc = Acc/n

        ## dummy call to flush the new model parameter in the last batch
        model(torch.rand((1,x.shape[1], x.shape[2], x.shape[3])).to(x.device)) 

        n, Loss, Acc = 0, 0.0, 0.0
        Acc36, Acc72, Acc108 = 0.0, 0.0, 0.0
        model.eval()
        if LLN:
            last_weight = model.model[-1].weight
            normalized_weight = torch.nn.functional.normalize(last_weight, p=2, dim=1)

        start = time.time()
        with torch.no_grad():
            for batch_idx, batch in enumerate(testLoader):
                x, y = batch[0].cuda(), batch[1].cuda()
                yh = model(x)
                Loss += criterion(yh, y).item() * y.size(0)
                n += y.size(0)
                correct = yh.max(1)[1] == y
                acc = correct.sum().item()
                Acc += acc
                

                if config.cert_acc:
                    margins, indices = torch.sort(yh, 1)
                    if LLN:
                        margins = margins[:, -1][:, None] - margins[: , 0:-1]
                        for idx in range(margins.shape[0]):
                            margins[idx] /= torch.norm(
                                normalized_weight[indices[idx, -1]] - normalized_weight[indices[idx, 0:-1]], dim=1, p=2)
                        margins, _ = torch.sort(margins, 1)
                        cert36 = margins[:, 0] > 36.0/255 * gamma
                        cert72 = margins[:, 0] > 72.0/255 * gamma
                        cert108= margins[:, 0] > 108.0/255 * gamma
                    else:
                        cert36 = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * gamma * 36/255.0
                        cert72 = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * gamma * 72/255.0
                        cert108= (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * gamma *108/255.0

                    Acc36 += torch.sum(correct & cert36).item()
                    Acc72 += torch.sum(correct & cert72).item()
                    Acc108+= torch.sum(correct & cert108).item()

        test_time = time.time()-start 
        test_loss = Loss/n
        test_acc = 100*Acc/n
        Acc36 = 100.0*Acc36/n 
        Acc72 = 100.0*Acc72/n
        Acc108= 100.0*Acc108/n

        ttime_epoch.append(train_time)
        vtime_epoch.append(test_time)
        tloss_epoch.append(train_loss)
        vloss_epoch.append(test_loss)
        tacc_epoch.append(train_acc)
        vacc_epoch.append(test_acc)

        if config.cert_acc:
            acc36_epoch.append(Acc36)
            acc72_epoch.append(Acc72)
            acc108_epoch.append(Acc108)

            txtlog(f"Epoch: {epoch+1:3d} | time: {train_time:.1f}/{test_time:.1f}, loss: {train_loss:.2f}/{test_loss:.2f}, acc: {train_acc:.1f}/{test_acc:.1f}, cert: {Acc36:.1f}/{Acc72:.1f}/{Acc108:.1f}, 100lr: {100*lr:.3f}")
            
        else:

            txtlog(f"Epoch: {epoch+1:3d} | time: {train_time:.1f}/{test_time:.1f}, loss: {train_loss:.2f}/{test_loss:.2f}, acc: {train_acc:.1f}/{test_acc:.1f}, 100lr: {100*lr:.3f}")

        if epoch % config.save_freq == 0 or epoch + 1 == Epochs:
            torch.save(model.state_dict(), f"{config.train_dir}/model.ckpt")
    
    # after training
    np.savetxt(f'{config.train_dir}/tloss_step.csv',np.array(tloss_step))
    np.savetxt(f'{config.train_dir}/tacc_step.csv',np.array(tacc_step))
    np.savetxt(f'{config.train_dir}/lr_step.csv',np.array(lr_step))
    np.savetxt(f'{config.train_dir}/ttime_epoch.csv',np.array(ttime_epoch))
    np.savetxt(f'{config.train_dir}/vtime_epoch.csv',np.array(vtime_epoch))
    np.savetxt(f'{config.train_dir}/tloss_epoch.csv',np.array(tloss_epoch))
    np.savetxt(f'{config.train_dir}/vloss_epoch.csv',np.array(vloss_epoch))
    np.savetxt(f'{config.train_dir}/tacc_epoch.csv',np.array(tacc_epoch))
    np.savetxt(f'{config.train_dir}/vacc_epoch.csv',np.array(vacc_epoch))
    if config.cert_acc:
        np.savetxt(f'{config.train_dir}/acc36_epoch.csv',np.array(acc36_epoch))
        np.savetxt(f'{config.train_dir}/acc72_epoch.csv',np.array(acc72_epoch))
        np.savetxt(f'{config.train_dir}/vacc_epoch.csv',np.array(vacc_epoch))

    xshape = (config.lip_batch_size, config.in_channels, config.img_size, config.img_size)
    x = (torch.rand(xshape) + 0.3*torch.randn(xshape)).cuda()
    gam = empirical_lipschitz(model, x)
    if model.gamma is None:
        txtlog(f"Lipschitz: {gam:.2f}/--")
    else:
        txtlog(f"Lipschitz capcity: {gam:.4f}/{gamma:.2f}, {100*gam/gamma:.2f}")

def train_toy(config):
    
    seed_everything(config.seed)
    trainLoader, testLoader = getDataLoader(config)
    model = getModel(config).cuda()
    criterion = getLoss(config)

    txtlog = TxtLogger(config)
    # wanlog = WandbLogger(config)

    txtlog(f"Set global seed to {config.seed:d}")
    
    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])

    if nparams >= 1000000:
        txtlog(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-6*nparams:.1f}M")
    else:
        txtlog(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-3*nparams:.1f}K")
    
    Epochs = config.epochs
    Lr = config.lr
    steps_per_epoch = len(trainLoader)
    gamma = config.gamma

    opt = torch.optim.Adam(model.parameters(), lr=Lr, weight_decay=0)
    lr_schedule = lambda t: np.interp([t], [0, Epochs*2//5, Epochs*4//5, Epochs], [0, Lr, Lr/20.0, 0])[0]

    for epoch in range(Epochs):
        ## train_step
        n, Loss = 0, 0.0
        model.train()
        for batch_idx, batch in enumerate(trainLoader):
            x, y = batch[0].cuda(), batch[1].cuda()
            lr = lr_schedule(epoch + (batch_idx+1)/steps_per_epoch)
            opt.param_groups[0].update(lr=lr)

            yh = model(x)
            J = criterion(yh, y)

            opt.zero_grad()
            J.backward()
            opt.step()

            loss = J.item()
            n += y.size(0)
            Loss += loss * y.size(0)
        
        train_loss = Loss/n 

        ## dummy call to flush the new model parameter in the last batch
        model(torch.rand((1,x.shape[1])).to(x.device)) 

        n, Loss = 0, 0.0, 
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(testLoader):
                x, y = batch[0].cuda(), batch[1].cuda()
                yh = model(x)
                Loss += criterion(yh, y).item() * y.size(0)
                n += y.size(0)

        test_loss = Loss/n

        txtlog(f"Epoch: {epoch+1:3d} | loss: {train_loss:.2f}/{test_loss:.2f}, 100lr: {100*lr:.3f}")
            
        if epoch % config.save_freq == 0 or epoch + 1 == Epochs:
            torch.save(model.state_dict(), f"{config.train_dir}/model.ckpt")
    
    # after training
    xshape = (config.lip_batch_size, config.in_channels)
    x = (torch.rand(xshape) + 0.3*torch.randn(xshape)).cuda()
    gam = empirical_lipschitz(model, x)
    if model.gamma is None:
        txtlog(f"Lipschitz: {gam:.2f}/--")
    else:
        txtlog(f"Lipschitz capcity: {gam:.4f}/{gamma:.2f}, {100*gam/gamma:.2f}")



