# -*- coding: utf-8 -*-
import os
import sys
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class NoveltyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index,1:-1], dtype=torch.float32)
        y = torch.tensor(self.data[index,-1], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.len

def load_csv(f_name, header=False, encoding='utf-8', **kwargs):
    with open(f_name, newline='', encoding=encoding) as csvfile:
        data_reader = csv.reader(csvfile, **kwargs)
        data = [row for row in data_reader]
    if header:
        return data
    else:
        return data[1:]

def get_dataset(args):
    if args.data_name == 'creditcard':
        data = csv_reader('./data/creditcard')
        data = np.array(data, dtype=np.float32)
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = sclaer.transform(data)

    
