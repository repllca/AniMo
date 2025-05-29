import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt

import time
import numpy as np
from collections import OrderedDict, defaultdict
from utils.eval_t2m import evaluation_vqvae
from utils.utils import print_current_loss

import os
import sys

def def_value():
    return 0.0


D_hdim = 512
INPUT_DIM = 22976
LAMBDA_ADV = 0.002


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, D_hdim),
            nn.LeakyReLU(0.2),
            nn.Linear(D_hdim, D_hdim),
            nn.LeakyReLU(0.2),
            nn.Linear(D_hdim, D_hdim),
            nn.LeakyReLU(0.2),
            nn.Linear(D_hdim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, -1)
        return self.model(x)

class RVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device

        self.discriminator = Discriminator(INPUT_DIM).to(self.device)
        self.adversarial_loss = nn.BCELoss()
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr)
        self.vq_model_optimizer = torch.optim.Adam(self.vq_model.parameters(), lr=args.lr)
        self.lambda_adv = LAMBDA_ADV

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()
        
        self.l2_criterion = torch.nn.MSELoss()
        
        

    def forward(self, batch_data):
        
       
        species, gender, motions = batch_data  
       
        motions = motions.to(self.device).float()
        pred_motion, loss_commit, perplexity = self.vq_model(species,gender,motions)


        
        self.motions = motions
        self.pred_motion = pred_motion

        loss_rec = self.l1_criterion(pred_motion, motions)
        pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
        loss_explicit = self.l1_criterion(pred_local_pos, local_pos)

        gen_loss =  loss_rec + self.opt.loss_vel * loss_explicit + self.opt.commit * loss_commit 
        real_validity = self.discriminator(motions)
        fake_validity = self.discriminator(pred_motion.detach())
        real_label = torch.ones_like(real_validity)
        fake_label = torch.zeros_like(fake_validity)
        d_real_loss = self.adversarial_loss(real_validity, real_label)
        d_fake_loss = self.adversarial_loss(fake_validity, fake_label)
        d_loss = d_real_loss + d_fake_loss
        g_loss = self.adversarial_loss(self.discriminator(pred_motion), real_label)
        gen_loss += self.lambda_adv * g_loss

        return (gen_loss.mean(),

                d_loss.mean(),
                g_loss.mean(),

                loss_rec.mean(),
                loss_explicit.mean(),
                loss_commit.mean(),
                perplexity.mean(),

                pred_motion,
                motions
                )

    def train_gan(self, batch_data, epoch):
        self.discriminator_optimizer.zero_grad()
        gen_loss_d, d_loss, _, loss_rec_d, loss_explicit_d, _, perplexity_d, pred_motion_d, motions_d = self.forward(batch_data)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=10)
        self.discriminator_optimizer.step()
        self.vq_model_optimizer.zero_grad()
        gen_loss, _, g_loss, loss_rec, loss_explicit, loss_commit, perplexity, pred_motion, motions = self.forward(batch_data)
        (gen_loss + loss_commit).backward()
        self.vq_model_optimizer.step()

        return gen_loss.item(), g_loss.item(), d_loss.item(), loss_rec.item(), loss_explicit.item(), loss_commit.item(), perplexity.item()

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.vq_model_optimizer.param_groups:
            param_group["lr"] = current_lr
        for param_group in self.discriminator_optimizer.param_groups:
            # param_group["lr"] = current_lr * 2
            param_group["lr"] = current_lr * 3
            # param_group["lr"] = current_lr * 4
        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.vq_model_optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.vq_model_optimizer.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        
        self.vq_model.to(self.device)
        self.vq_model_optimizer = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.vq_model_optimizer, milestones=self.opt.milestones, gamma=self.opt.gamma)
        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print(f"Load model epoch:{epoch} iterations:{it}")

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print(f'Iters Per Epoch, Training: {len(train_loader):04d}, Validation: {len(eval_val_loader):03d}')
        logs = defaultdict(def_value, OrderedDict())
        
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_vqvae(
            self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=1000,
            best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100,
            eval_wrapper=eval_wrapper, save=False)

        while epoch < self.opt.max_epoch:
            self.vq_model.train()
            self.discriminator.train()
            
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                else:
                    current_lr = self.vq_model_optimizer.param_groups[0]['lr']

                gen_loss, g_loss, d_loss_item, loss_rec, loss_explicit, loss_commit, perplexity = self.train_gan(batch_data, epoch)

                logs['gen_loss'] += gen_loss
                logs['g_loss'] += g_loss
                logs['d_loss'] += d_loss_item
                logs['loss_rec'] += loss_rec
                logs['loss_explicit'] += loss_explicit
                logs['loss_commit'] += loss_commit
                logs['perplexity'] += perplexity
                logs['lr'] += current_lr
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s' % tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')
            self.vq_model.eval()
            self.discriminator.eval()
            
            val_gen_loss = []
            val_g_loss = []
            val_d_loss = []
            val_loss_rec = []
            val_loss_explicit = []
            val_loss_commit = []
            val_perplexity = []
            
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    gen_loss, d_loss, g_loss, loss_rec, loss_explicit, loss_commit, perplexity, pred_motion, motions = self.forward(batch_data)
                    
                    val_gen_loss.append(gen_loss.item())
                    val_g_loss.append(g_loss.item())
                    val_d_loss.append(d_loss.item())
                    val_loss_rec.append(loss_rec.item())
                    val_loss_explicit.append(loss_explicit.item())
                    val_loss_commit.append(loss_commit.item())
                    val_perplexity.append(perplexity.item())

            print(f"Validation - Gen Loss: {np.mean(val_gen_loss):.3f}, G Loss: {np.mean(val_g_loss):.3f}, "
                  f"D Loss: {np.mean(val_d_loss):.3f}, Rec Loss: {np.mean(val_loss_rec):.3f}, "
                  f"Explicit Loss: {np.mean(val_loss_explicit):.3f}, Commit Loss: {np.mean(val_loss_commit):.3f}, "
                  f"Perplexity: {np.mean(val_perplexity):.3f}")
            
            self.logger.add_scalar('Val/gen_loss', np.mean(val_gen_loss), epoch)
            self.logger.add_scalar('Val/g_loss', np.mean(val_g_loss), epoch)
            self.logger.add_scalar('Val/d_loss', np.mean(val_d_loss), epoch)
            self.logger.add_scalar('Val/loss_rec', np.mean(val_loss_rec), epoch)
            self.logger.add_scalar('Val/loss_explicit', np.mean(val_loss_explicit), epoch)
            self.logger.add_scalar('Val/loss_commit', np.mean(val_loss_commit), epoch)
            self.logger.add_scalar('Val/perplexity', np.mean(val_perplexity), epoch)

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_vqvae(
                self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3, 
                best_matching=best_matching, eval_wrapper=eval_wrapper)

            epoch += 1
            
            if epoch % self.opt.eval_every_e == 0:
                data = torch.cat([self.motions[:4], self.pred_motion[:4]], dim=0).detach().cpu().numpy()
                save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                os.makedirs(save_dir, exist_ok=True)
                plot_eval(data, save_dir)