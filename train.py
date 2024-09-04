import argparse
import os
import random
import time
import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

from torch.utils.tensorboard import SummaryWriter
import core.datasets as datasets
from core.models import *
from evaluate import validate_things_v2, validate_kitti_v2

"""

python train.py --log_wandb --num_steps 100 --val_freq 20 --batch_size 8

# finetune on kitti
python train.py \
    --restore_ckpt checkpoints/2024-08-14_22-22-40/checkpoint_035000.pth \
    --log_wandb --num_steps 5000 --val_freq 100 --batch_size 8 \
    --train_datasets kitti \
    --lr 1e-5 \
    --image_size 320 1000 \
    --save_path 2024-08-14_22-22-40_finetune_kitti

"""


def parse_config():
    """
    Parse the configuration for the training script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_disp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--model', default='basic',
                        help='select model')
    parser.add_argument('--data_path', default='data/SceneFlowData/',
                        help='data path')
    parser.add_argument('--num_steps', type=int, default=100000,
                        help='number of steps')
    parser.add_argument('--val_freq', type=int, default=1000,
                        help='validation frequency')
    parser.add_argument('--restore_ckpt', default= None,
                        help='load model')
    parser.add_argument('--save_path', default=None,
                        help='save model')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=310, metavar='S',
                        help='random seed (default: 310)')
    parser.add_argument('--log_wandb', action='store_true', default=False,
                        help='log to wandb')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8, 
                        help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], 
                        help="training datasets.")
    parser.add_argument('--val_dataset', default='things',
                        help="validation dataset.")
    parser.add_argument('--lr', type=float, default=0.001, 
                        help="max learning rate.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 512], 
                        help="size of the random image crops used during training.")
    parser.add_argument('--weight_decay', type=float, default=.00001, 
                        help="Weight decay in optimizer.")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    if args.save_path is None:
        # Generate a save path if not provided based on the current time
        args.save_path = 'checkpoints/' + time.strftime('%Y-%m-%d_%H-%M-%S')
    else:
        # Use the provided save path
        args.save_path = 'checkpoints/' + args.save_path

    # Create the save path if it does not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print(f"Experiment will be saved in: {args.save_path}")

    return args



def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='runs')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()



def train():
    args = parse_config()

    if args.log_wandb:
        wandb.login()
        wandb.init(project="simple-stereo", sync_tensorboard=True)

    # Load the data
    train_loader = datasets.fetch_dataloader(args)

    # Create the model
    assert args.model == 'basic', 'Only basic model is supported for now'

    model = basic(args.max_disp)
    model = nn.DataParallel(model)
        
    if args.restore_ckpt is not None:
        logging.info(f'Restoring model from {args.restore_ckpt}')
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint)
        logging.info('Model restored successfully')
    
    model.cuda()

    logging.info(f'Model: {args.model}')
    logging.info(f'Number of model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    # Set up the optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        total_steps=args.num_steps + 100,
        pct_start=0.01, 
        cycle_momentum=False, 
        anneal_strategy='linear'
    )

    logger = Logger(model, scheduler)

    should_keep_training = True
    global_batch_num = 0
    # Training loop
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]
            valid = valid.bool()

            # Negative disparity to positive disparity
            flow = torch.abs(flow)

            assert model.training
            depth_predictions = model(image1, image2)
            assert model.training

            # Squeeze out the channel dimension (B, 1, H, W) -> (B, H, W)
            depth_predictions = depth_predictions.squeeze(1)
            flow = flow.squeeze(1)

            loss = F.smooth_l1_loss(depth_predictions[valid], flow[valid], reduction='mean')
            
            if torch.isnan(loss) or torch.isinf(loss):
                print('NaN loss encountered. Skipping this batch.')
                continue
            
            global_batch_num += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            logger.writer.add_scalar('train_loss', loss.item(), global_batch_num)
            logger.writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_batch_num)

            if global_batch_num % args.val_freq == 0:
                # save the model with filename containing the epoch number 3 digits wide
                current_save_path = os.path.join(args.save_path, f'checkpoint_{global_batch_num:06d}.pth')
                logging.info(f'Saving model to {current_save_path}')
                torch.save(model.state_dict(), current_save_path)

                if args.val_dataset == 'things':
                    results = validate_things_v2(model.module)
                elif args.val_dataset == 'kitti':
                    results = validate_kitti_v2(model.module)
                else:
                    raise ValueError(f'Unknown validation dataset: {args.val_dataset}')
                logger.write_dict(results)

                model.train()

        if global_batch_num >= args.num_steps:
            should_keep_training = False
            break

    print("FINISHED TRAINING")
    if args.log_wandb:
        wandb.finish()
    

if __name__ == '__main__':
   train()
    
