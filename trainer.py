import os
import csv
import json
import random
import argparse

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from network import *

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='ae')
    parser.add_argument('--data_name', type=str, default='creditcard')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--val_iter', type=int, default=1)

    args = parser.parse_args()
    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

def load_csv(f_name, header=False, encoding='utf-8', **kwargs):
    with open(f_name, newline='', encoding=encoding) as csvfile:
        data_reader = csv.reader(csvfile, **kwargs)
        data = [row for row in data_reader]
    if header:
        return data
    else:
        return data[1:]

class NoveltyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.input_size = self.data.shape[1] - 1
        self.len = self.data.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.data[index,:-1], dtype=torch.float32)
        y = torch.tensor(self.data[index,-1], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.len

def get_dataset(args):
    if args.data_name == 'creditcard':
        data = load_csv('./data/creditcard.csv')
        data = np.array(data, dtype=np.float32)
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        data = data[:, 1:]
    
    print('Data Name : {}'.format(args.data_name))
    trains, tests = train_test_split(data, train_size=0.9, random_state=args.seed)
    trains, vals = train_test_split(trains, train_size=0.8, random_state=args.seed)
    print('Train / Validation / Test split')
    print('train / val / tests : {} / {} / {}'.format(trains.shape[0], vals.shape[0], tests.shape[0]))
    print('Train : ')
    print('normal : {}'.format((trains[:,-1]==0.).sum()))
    print('novelty : {}'.format((trains[:,-1]==1.).sum()))
    print('Valid : ')
    print('normal : {}'.format((vals[:,-1]==0.).sum()))
    print('novelty : {}'.format((vals[:,-1]==1.).sum()))
    
    return NoveltyDataset(trains), NoveltyDataset(vals), NoveltyDataset(tests)

def get_model(args, input_size):
    if args.model == 'ae':
        model = AE(input_size=input_size).to(args.device)
    elif args.model == 'vae':
        model = VAE(input_size=input_size).to(args.device)
    elif args.model == 'aae':
        model = AAE(input_size=input_size).to(args.device)
    return model

def get_optimizer(args, model):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    return optimizer

def get_loss_fn(args):
    if args.model == 'ae':
        criterion = nn.MSELoss(reduction='mean')
    return criterion

class Trainer(object):
    def __init__(self, args):
        self.device = args.device
        self.args = args
        self.trainset, self.valset, self.testset = get_dataset(self.args)
        self.input_size = self.trainset.input_size
        self.train_loader = DataLoader(self.trainset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0)
        self.val_loader = DataLoader(self.valset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0)
        self.test_loader = DataLoader(self.testset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0)
        self.model = get_model(args, self.input_size)
        self.optimizer = get_optimizer(args, self.model)
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True)
        self.criterion = get_loss_fn(args)

    def get_recon_err(self, inputs, outputs):
        diff = inputs - outputs
        diff = diff.numpy()
        diff = np.power(diff, 2).mean(axis=1)
        return diff
    
    def run_batch(self, batch, train):
        x, y = batch
        inputs = x.to(self.device)
        outputs = self.model(inputs)
        if train:
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, inputs)
            loss.backward()
            self.optimizer.step()
        else:
            loss = torch.mean(self.criterion(outputs, inputs))
        loss = loss.item()
        batch_size = y.size(0)
        recon_err = self.get_recon_err(inputs.detach().cpu(), outputs.detach().cpu()) # numpy array
        label = y.tolist()
        return loss * batch_size, recon_err.tolist(), label, batch_size

    def train(self, epoch=None):
        self.model.train()
        total_loss = 0
        total_cnt = 0
        total_recon_err = []
        total_label = []
        auc_roc = 0
        ap = 0

        tqdm_batch_iterator = tqdm(self.train_loader)
        for batch_idx, batch in enumerate(tqdm_batch_iterator):
            
            batch_status = self.run_batch(batch, train=True)
            loss, recon_err, label, batch_size = batch_status
            
            total_loss += loss
            total_recon_err += recon_err
            total_label += label
            total_cnt += batch_size

            description = "[Epoch: {:3d}][Loss: {:6f}][lr: {:7f}]".format(epoch,
                    loss, 
                    self.optimizer.param_groups[0]['lr'])
            tqdm_batch_iterator.set_description(description)
        auc_roc = metrics.roc_auc_score(total_label, total_recon_err)
        ap = metrics.average_precision_score(total_label, total_recon_err)
        status = {'train_loss':total_loss/total_cnt,
                'train_auc_roc':auc_roc,
                'train_ap':ap}
        return status

    def valid(self, epoch=None):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_recon_err = []
            total_label = []
            total_cnt = 0

            for batch_idx, batch in enumerate(self.val_loader):
                
                batch_status = self.run_batch(batch, train=False)
                loss, recon_err, label, batch_size = batch_status
                
                total_loss += loss
                total_recon_err += recon_err
                total_label += label
                total_cnt += batch_size
        
        auc_roc = metrics.roc_auc_score(total_label, total_recon_err)
        ap = metrics.average_precision_score(total_label, total_recon_err)
        status = {'valid_loss':total_loss/total_cnt,
                'valid_auc_roc':auc_roc,
                'valid_ap':ap}
        return status

    def run(self):
        train_report = 'train loss : {:.4f}, AUC-ROC : {:.4f}, AUC-PR : {:.4f}'
        val_report = 'valid loss : {:.4f}, AUC-ROC : {:.4f}, AUC-PR : {:.4f}'
        for epoch in range(1, self.args.epochs + 1):
            epoch_status = self.train(epoch)
            print(train_report.format(
                epoch_status['train_loss'],
                epoch_status['train_auc_roc'],
                epoch_status['train_ap']
                ))
            if epoch % self.args.val_iter == 0:
                valid_status = self.valid(epoch)
                print(val_report.format(
                    valid_status['valid_loss'],
                    valid_status['valid_auc_roc'],
                    valid_status['valid_ap']
                    ))
                self.scheduler.step(valid_status['valid_loss'])

if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    trainer = Trainer(args)
    trainer.run()
