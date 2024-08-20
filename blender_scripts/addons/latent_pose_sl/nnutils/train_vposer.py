# -*- coding: utf-8 -*-
import os
import argparse
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import tqdm
import torch.cuda.amp as amp  # For mixed precision

from vposer import VPoser
from skeleton import Skeleton
from geoutils import *

class VPoserTrainer:

    def __init__(self, work_dir, skeleton_path, checkpoint_dir='checkpoints', checkpoint_file=None):
        from dataloader import AnimationDS

        self.batch_size = 512  # Adjusted batch size for better GPU utilization
        self.pt_dtype = torch.float32
        self.comp_device = torch.device("cuda")

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = checkpoint_file

        # DataLoader setup
        ds_train = AnimationDS(work_dir + "_train.pt")
        self.ds_train = DataLoader(ds_train, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

        ds_val = AnimationDS(work_dir + "_val.pt")
        self.ds_val = DataLoader(ds_val, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

        ds_test = AnimationDS(work_dir + "_test.pt")
        self.ds_test = DataLoader(ds_test, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

        print('Train dataset size %.2f M' % (len(self.ds_train.dataset) * 1e-6))
        print('Validation dataset size %d' % len(self.ds_val.dataset))
        print('Test dataset size %d' % len(self.ds_test.dataset))

        data_shape = list(ds_val[0]['pose_aa'].shape)
        self.latentD = 32
        self.vposer_model = VPoser(num_neurons=512, latentD=self.latentD, data_shape=data_shape, use_cont_repr=True)
        self.vposer_model.to(self.comp_device)

        varlist = [var[1] for var in self.vposer_model.named_parameters()]
        self.optimizer = optim.AdamW(varlist, lr=1e-2, weight_decay=0.0001)  # Using AdamW

        self.ske = Skeleton(skeleton_path=skeleton_path)
        self.ske.to(self.comp_device)
        self.default_trans = torch.zeros(3).view(1, 3).to(self.comp_device)

        self.best_loss_total = np.inf
        self.epochs_completed = 0

        # Load checkpoint if provided
        if checkpoint_file:
            self.load_checkpoint(checkpoint_file)

    def save_checkpoint(self, epoch, best_loss_total):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.vposer_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss_total': best_loss_total,
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=self.comp_device)
        self.vposer_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss_total = checkpoint['best_loss_total']
        self.epochs_completed = checkpoint['epoch']
        print(f"Resumed from checkpoint {checkpoint_file} at epoch {self.epochs_completed}")

    def train(self):
        self.vposer_model.train()
        save_every_it = len(self.ds_train) / 4
        train_loss_dict = {}

        scaler = amp.GradScaler()

        for it, dorig in enumerate(self.ds_train):
            dorig = {k: dorig[k].to(self.comp_device) for k in dorig.keys()}
            
            self.optimizer.zero_grad()

            with amp.autocast():  # Enable mixed precision
                drec = self.vposer_model(dorig['pose_aa'], output_type='aa')
                loss_total, cur_loss_dict = self.compute_loss(dorig, drec)

            scaler.scale(loss_total).backward()
            scaler.step(self.optimizer)
            scaler.update()

            train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}
            if it % (save_every_it + 1) == 0:
                cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                print("Training Iteration: {}, Loss: {}".format(it, cur_train_loss_dict))

        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}
        return train_loss_dict

    def evaluate(self, split_name='vald'):
        self.vposer_model.eval()
        eval_loss_dict = {}
        data = self.ds_val if split_name == 'vald' else self.ds_test
        with torch.no_grad():
            for dorig in data:
                dorig = {k: dorig[k].to(self.comp_device) for k in dorig.keys()}
                drec = self.vposer_model(dorig['pose_aa'], output_type='aa')
                _, cur_loss_dict = self.compute_loss(dorig, drec)
                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in cur_loss_dict.items()}

        eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}
        return eval_loss_dict

    def compute_loss(self, dorig, drec):
        q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])

        prec = drec['pose_aa'].to(self.comp_device)
        porig = dorig['pose_aa'].to(self.comp_device)

        batchnum = prec.shape[0]
        trans = self.default_trans.repeat(batchnum, 1)

        joint_rec = self.ske(prec, trans)
        joint_rig = self.ske(porig, trans)

        loss_joint_rec = (1. - 5e-3) * torch.mean(torch.abs(joint_rec - joint_rig))

        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros([self.batch_size, self.latentD], requires_grad=False).to(prec.device),
            scale=torch.ones([self.batch_size, self.latentD], requires_grad=False).to(prec.device))
        loss_kl = 5e-3 * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))

        loss_dict = {'loss_kl': loss_kl, 'loss_joint_rec': loss_joint_rec}

        if self.vposer_model.training and self.epochs_completed < 10:
            loss_dict['loss_pose_rec'] = (1. - 5e-3) * torch.mean(torch.sum(torch.pow(porig - prec, 2), dim=[1, 2, 3]))

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def perform_training(self, num_epochs=None, message=None):
        starttime = datetime.now().replace(microsecond=0)
        if num_epochs is None:
            num_epochs = 500

        print(
            'Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), num_epochs))

        prev_lr = np.inf
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(num_epochs // 3), gamma=0.5)
        self.best_loss_total = np.inf
        loop = tqdm.tqdm(range(self.epochs_completed + 1, num_epochs + 1))
        for epoch_num in loop:
            print("Started Training Epoch {}".format(epoch_num))
            cur_lr = self.optimizer.param_groups[0]['lr']
            if cur_lr != prev_lr:
                print('--- Optimizer learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr
            self.epochs_completed = epoch_num
            train_loss_dict = self.train()
            eval_loss_dict = self.evaluate()
            scheduler.step()

            with torch.no_grad():
                print("Eval Training Epoch {}".format(epoch_num))
                if eval_loss_dict['loss_total'] < self.best_loss_total:
                    self.best_model_fname = os.path.join('snapshots', 'E%03d.pt' % (self.epochs_completed))
                    
                    output_dir = os.path.dirname(self.best_model_fname)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        
                    self.best_loss_total = eval_loss_dict['loss_total']
                    torch.save(self.vposer_model.state_dict(), self.best_model_fname)
                    print("Loss {} is less, save model to {}".format(self.best_loss_total, self.best_model_fname))
                else:
                    print("Loss {} is larger, skip".format(eval_loss_dict['loss_total']))

                # Save a checkpoint after each epoch
                self.save_checkpoint(epoch_num, self.best_loss_total)

        endtime = datetime.now().replace(microsecond=0)

        print('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        print('Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_loss_total))
        print('Best model path: %s\n' % self.best_model_fname)

def run_vposer_trainer(datapath, bodymodel_path, checkpoint_dir='checkpoints', checkpoint_file=None):
    vp_trainer = VPoserTrainer(datapath, bodymodel_path, checkpoint_dir=checkpoint_dir, checkpoint_file=checkpoint_file)
    vp_trainer.perform_training()

    test_loss_dict = vp_trainer.evaluate(split_name='test')

    print('Final loss on test set is %s' % (' | '.join(['%s = %.2e' % (k, v) for k, v in test_loss_dict.items()])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VPoser model')
    parser.add_argument('--datapath', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--skeletonpath', type=str, required=True, help='Path to the skeleton file')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_file', type=str, default=None, help='Checkpoint file to resume training from')
    
    args = parser.parse_args()

    run_vposer_trainer(args.datapath, args.skeletonpath, checkpoint_dir=args.checkpoint_dir, checkpoint_file=args.checkpoint_file)
