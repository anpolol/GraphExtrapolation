#Here we collect BACE dataset from raw files (файлы уже давно мной подготовлены, я потом как-то выложу и подготовку самих этих файлов)
#В BACE именно первая цифра означает номер environment'а, а вторая - порядковый номер графа, а метка это последний атрибут в файликах attrs

import os
import collections
from os import listdir

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import random

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
        self.dataset, self.n_min = self.run(name, self.indices)
        self.N = len(self.dataset)
        super().__init__()

    def run(self, name, indices):
        dataset = []
        y_max = 0
        n_min = np.inf
        for i in list(indices['first']):
            for j in range(indices[indices['first'] == i]['second'].item()):
                G = self.data_load(name, i, j)
                dataset.append(G)
                if G.num_nodes <= n_min:
                    n_min = G.num_nodes
                if G.y.item() > y_max:
                    y_max = G.y.item()
        self.num_classes = int(y_max+1)
        return dataset, n_min

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

    def split_random(self, p):
        shuffled_dataset = random.sample(self.dataset, len(self.dataset))
        train_data = shuffled_dataset[:int(self.N*p)]
        test_data = shuffled_dataset[int(self.N*p):]
        return train_data, test_data

    def split_env(self, p):

        # здесь проценты не будут сохраняться прям точь в точь
        # жадный алгоритм

        train_data = []
        test_data = []
        train_split = int(len(self.dataset)*p)

        env_list = self.indices['first'].tolist()

        while len(train_data) < train_split:
            random.shuffle(env_list)
            k = (env_list.pop())
            number_of_graphs = int(self.indices[self.indices['first'] == k]['second']) + 1
            train_indices = sum(self.indices[self.indices['first'] < k]['second']) + k
            train_data += self.dataset[train_indices:train_indices+number_of_graphs]

        for k in env_list:
            number_of_graphs = int(self.indices[self.indices['first'] == k]['second']) + 1
            test_indices = sum(self.indices[self.indices['first'] < k]['second']) + k

            test_data += self.dataset[test_indices:test_indices + number_of_graphs]
        print(len(self.dataset), len(train_data),len(test_data))

        return train_data, test_data




