#Here we collect BACE dataset from raw files (файлы уже давно мной подготовлены, я потом как-то выложу и подготовку самих этих файлов)
#В BACE именно первая цифра означает номер environment'а, а вторая - порядковый номер графа, а метка это последний атрибут в файликах attrs

import os
import collections
from os import listdir
import pandas as pd
import torch
from torch_geometric.data import Data
import random
import itertools

class DataPreparing:
    '''
    args
    name - name of a dataset
    '''
    def __init__(self, name: str):
        self.dataset = []
        self.name = name
        path_initial = './datasets/' + name
        self.indices = self.indices_exctracting(path_initial)
        self.dataset = self.run(name, self.indices)
        self.N = len(self.dataset)


        super().__init__()

    def run(self, name, indices):
        dataset = [ ]
        y_max = 0
        for i in list(indices['first']):
            for j in range(indices[indices['first'] == i]['second'].item()):
                G = self.data_load(name, i, j)
                dataset.append(G)
                if G.y.item() > y_max:
                    y_max = G.y.item()
        self.num_classes = int(y_max+1)
        return dataset

    def indices_exctracting(self, path_initial): # нужно узнать максимальное число графов и максимальное число окружений (первый индекс)
        names_datasets = listdir(path_initial)
        df_indices = pd.DataFrame(columns=['first', 'second'])
        for name in names_datasets:
            split = name.split('_')
            if len(split) == 3:
                df_indices = pd.concat([df_indices, pd.DataFrame({'first': int(split[1]), 'second': int(split[2].split('.')[0])}, index=[0])])
        indices = df_indices.groupby(by=['first']).max().reset_index()
        return indices  # второй индекс - это не число графов с соотв окружением а МАКСИМАЛЬНЫЙ индекс, на самом деле графов на 1 больше

    def data_load(self, name, i, j):
        path = './datasets/'

        with open(path + str(name) + '/edge_list_' + str(i) + '_' + str(j) + '_' + '.txt', "r") as f:
            l = f.readlines()

        edge_list = []
        for line in l:
            f = (line.split(','))
            edge_list.append([int(f[0]), int(f[1])])
        edge_index = (torch.tensor(edge_list).T)

        with open(path + str(name) + '/attrs_' + str(i) + '_' + str(j) + '.txt', "r") as f:
          l = f.readlines()

        attrs = []
        for line in l:
            f = (line.split(','))
            attr = []
            for symb in (f[:len(f) - 1]):
                attr.append(float(symb))
            attrs.append(attr)

        y = torch.tensor(int(float(l[0].split(',')[(len(f) - 1)])))
        x = torch.tensor(attrs)
        G = Data(edge_index=edge_index, x=x,y=y,env = i,num_nodes_features = x.shape[1])
        return G

    def split_random(self,p):
        shuffled_dataset = random.sample(self.dataset, len(self.dataset))
        train_data = shuffled_dataset[:int(self.N*p)]
        test_data = shuffled_dataset[int(self.N*p):]
        return train_data, test_data

    def split_env(self, p):

        #здесь проценты не будут сохраняться
        #сначала надо отобрать какие именно контексты подходят для train_percent деления
        #TODO сделать автоматическое разделение на трейн тест, причем чтоб были всевозможные варианты и он каждый раз рандомно выбирал
        # look in junk.ipynb for a sample with combinations()
        if p == 0.7:
            max_env = 82
        elif p == 0.8:
            max_env = 112
        elif p == 0.9:
            max_env = 142


        train_split = sum(self.indices['second'].iloc[:max_env]) + max_env

        train_data = self.dataset[:train_split+1]
        test_data = self.dataset[train_split+1:]

        return train_data, test_data




