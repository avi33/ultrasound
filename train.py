import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
from utils.helper_funcs import add_weight_decay
import utils.logger as logger
import datasets.fda as fda
from kornia.metrics import psnr
# from clearml import Task

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--dataset", default="kaggle", type=str)
    '''net'''    
    '''optimizer'''
    parser.add_argument("--max_lr", default=3e-4, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument('--ema', default=0.995, type=float)
    parser.add_argument("--amp", action='store_true', default=False)        
    '''debug'''
    parser.add_argument("--save_path", default='outputs/tmp', type=Path)
    parser.add_argument("--load_path", default=None, type=Path)
    parser.add_argument("--save_interval", default=100, type=int)    
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--use_fda", default=False, action="store_true")
    
    args = parser.parse_args()
    return args

def train():
    args = parse_args()

    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None    
    root.mkdir(parents=True, exist_ok=True)
    # task = Task.init(project_name='classification', task_name='transformer')
    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))
        
    ####################################
    # Data #
    ####################################
    from datasets.data_utils import create_dataset
    train_set, test_set = create_dataset(args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=False)
    
    if args.use_fda:
        import datasets.fda as fda

    ####################################
    # Network #
    ####################################
    from modules.models import Net
    net = Net(emb_dim=128, nf=16, factors=[2, 2, 2])        
    net.to(device)
    
    '''optimizer'''
    if args.amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(init_scale=2**10)
        eps = 1e-4
    else:
        scaler = None
        eps = 1e-8
    
    parameters = add_weight_decay(net, weight_decay=args.wd, skip_list=())

    opt = optim.AdamW(parameters, lr=args.max_lr, betas=(0.9, 0.99), eps=eps, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                       max_lr=args.max_lr,
                                                       steps_per_epoch=len(train_loader),
                                                       epochs=args.n_epochs,
                                                       pct_start=0.1,                                                       
                                                    )    
    if args.ema is not None:
        from modules.ema import ModelEma as EMA
        ema = EMA(net, decay_per_epoch=args.ema)
        epochs_from_last_reset = 0
        decay_per_epoch_orig = args.ema

    '''loss'''    
    c_rec = nn.L1Loss(reduction="sum").to(device)
    from kornia.losses import TotalVariation, SSIMLoss
    c_tv = TotalVariation().to(device)
    c_ssim = SSIMLoss(reduction="sum", window_size=5).to(device)

    if load_root and load_root.exists():
        checkpoint = torch.load(load_root / "chkpnt.pt")
        net.load_state_dict(checkpoint['model_dict'])
        opt.load_state_dict(checkpoint['opt_dict'])
        steps = checkpoint['resume_step'] if 'resume_step' in checkpoint.keys() else 0
        del checkpoint
        print('checkpoints loaded')        

    
    torch.backends.cudnn.benchmark = True    
    steps = 0        
    skip_scheduler = False    

    n_test_samples = min(4, args.batch_size)
    disp_samples = []
    idx_disp = np.random.permutation(len(test_set))[:n_test_samples]

    for i in range(n_test_samples):
        disp_samples.append(test_set[idx_disp[i]])

    grid = torchvision.utils.make_grid(disp_samples, nrow=2)
    for i in range(n_test_samples): 
        writer.add_image('images/orig', grid, global_step=i)

    for epoch in range(1, args.n_epochs + 1):
        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", logger.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"
        
        if args.ema is not None:
            if epochs_from_last_reset <= 1:  # two first epochs do ultra short-term ema
                ema.decay_per_epoch = 0.01
            else:
                ema.decay_per_epoch = decay_per_epoch_orig
            epochs_from_last_reset += 1
            # set 'decay_per_step' for the eooch
            ema.set_decay_per_step(len(train_loader))        
        
        for iterno, x in  enumerate(metric_logger.log_every(train_loader, args.log_interval, header)):        
            net.zero_grad(set_to_none=True)
            x = x.to(device)

            if args.use_fda:
                x = fda.fda(x)
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):                
                x_est = net(x)
                loss_rec = c_rec(x_est, x) / x.shape[0]
                loss_ssim = c_ssim(x_est, x) / x.shape[0]
                loss_tv = c_tv(x_est).sum(-1).mean(0)
                loss = loss_rec + 0.1*loss_ssim + loss_tv
                
            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
                scaler.step(opt)
                amp_scale = scaler.get_scale()
                scaler.update()
                skip_scheduler = amp_scale != scaler.get_scale()
            else:
                loss.backward()
                opt.step()

            if args.ema is not None:
                ema.update(net, steps)

            if not skip_scheduler:
                lr_scheduler.step()

            '''metrics'''            
            
            metric_logger.update(loss=loss.item())            
            metric_logger.update(loss_rec=loss_rec.item())
            metric_logger.update(loss_ssim=loss_ssim.item())
            metric_logger.update(loss_tv=loss_tv.item())
            metric_logger.update(lr=opt.param_groups[0]["lr"])
            ######################
            # Update tensorboard #
            ######################               
            steps += 1                        
            if steps % args.save_interval != 0:
                writer.add_scalar(f"train/loss_rec", loss.item(), steps)
                writer.add_scalar(f"train/loss_ssim", loss_ssim.item(), steps)
                writer.add_scalar(f"train/loss_tv", loss_tv.item(), steps)
                writer.add_scalar(f"train/loss", loss.item(), steps)
                writer.add_scalar(f"train/psnr", psnr(x_est, x, max_val=1).mean(0), steps)
                writer.add_scalar(f"lr", lr_scheduler.get_last_lr()[0], steps)
            else:
                loss_test = 0  
                psnr_test = 0              
                net.eval()
                with torch.no_grad():                                        
                    for i, x in enumerate(test_loader):
                        x = x.to(device)
                        x_est = net(x)
                        loss_test += c_rec(x_est, x).item()                                                
                        psnr_test += psnr(x_est, x, max_val=1).mean(0)
                                                
                loss_test /= len(test_loader)
                psnr_test /= len(test_loader)

                writer.add_scalar("loss/test", loss_test, steps)
                writer.add_scalar("loss/psnr", psnr_test, steps)

                with torch.no_grad():
                    disp_samples_est = []
                    for k, d in enumerate(disp_samples):
                        d = d.unsqueeze(0).to(device)
                        d_est = net(d)
                        disp_samples_est.append(d_est.squeeze(0))                                                

                grid = torchvision.utils.make_grid(disp_samples_est, nrow=2)
                for k in range(n_test_samples): 
                    writer.add_image('images/est', grid, global_step=k)

                metric_logger.update(loss_test=loss_test)

                net.train()                
                                
                chkpnt = {
                    'model_dict': net.state_dict(),
                    'opt_dict': opt.state_dict(),
                    'step': steps,
                }
                torch.save(chkpnt, root / "chkpnt.pt")

if __name__ == "__main__":
    train()