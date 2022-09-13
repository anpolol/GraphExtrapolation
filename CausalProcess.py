import pandas as pd
import bamt
import bamt.Networks as Nets
from sklearn import preprocessing
from bamt.Preprocessors import Preprocessor
from pgmpy.estimators import K2Score
from torch_geometric.utils import get_laplacian, to_dense_adj, dense_to_sparse
import torch
import math
import numpy as np

def func(x):
    if x[1] == 'y' and len(x[0]) > 1:

        number = int(x[0][5:])
    elif x[0] == 'y' and len(x[1]) > 1:
        number = int(x[1][5:])
    else:
        number = np.nan
    return number


class CausalProcess:
    '''
    preprocessing with causality
    '''

    def __init__(self, train_dataset: list, test_dataset: list, score_func: str, init_edges: bool,
                 remove_init_edges: bool, white_list):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.score_func = score_func
        self.init_edges = init_edges
        self.remove_init_edges = remove_init_edges
        self.white_list = white_list

        if self.score_func == 'K2':
            self.score = K2Score
        elif self.score_func == 'MI':
            self.score = None
        else:
            raise Exception('there is no ', self.score_func, 'score function. Choose one of: MI, K2')

        super().__init__()

    def run(self):
        data_bamt = self.data_eigen_exctractor(self.train_dataset)
        bn = self.bn_build(data_bamt)
        lis = list(map(lambda x: func(x), bn.edges)) #мы берем только те веришны, которые исходят из y или входят в у
        left_vertices = sorted(list(filter(lambda x: not np.isnan(x), lis)))
        left_edges = list(filter(lambda x: x[0] == 'y' or x[1] == 'y', bn.edges))
        left_edges = sorted(left_edges, key=lambda x: int(x[0][5:] if x[1]=='y' else int(x[1][5:])))
        ll = list(map(lambda x: bn.weights[tuple(x)], left_edges))
        N = len(ll) #TODO подумать: мб тут было бы логичнее взять N = число переменных из которых строилась bn
        weights_preprocessed = list(map(lambda x: x * N / sum(ll), ll))
        train_dataset = self.convolve(self.train_dataset, weights_preprocessed, left_vertices)
        test_dataset = self.convolve(self.test_dataset, weights_preprocessed, left_vertices)
        return train_dataset, test_dataset

    def data_eigen_exctractor(self, dataset):
        columns_list = list(map(lambda x: 'eigen' + str(x), range(10)))
        data_bamt = pd.DataFrame(columns=columns_list + ['y'])
        for gr in dataset:
            A = to_dense_adj(gr.edge_index)
            eig = torch.eig(A.reshape(A.shape[1], A.shape[2]))[0].T[0].T
            ordered, indices = torch.sort(eig[:gr.num_nodes], descending=True)

            to_append = pd.Series(ordered[:10].tolist() + [gr.y], index=data_bamt.columns)
            data_bamt = data_bamt.append(to_append, ignore_index=True)

        return data_bamt

    def bn_build(self, data_bamt):
        # поиск весов для bamt
        for col in data_bamt.columns[:len(data_bamt.columns)]:
            data_bamt[col] = data_bamt[col].astype(float)
        data_bamt['y'] = data_bamt['y'].astype(int)

        bn = Nets.HybridBN(has_logit=True)
        discretizer = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        p = Preprocessor([('discretizer', discretizer)])
        discretized_data, est = p.apply(data_bamt)

        bn.add_nodes(p.info)

        params = dict()
        params['remove_init_edges'] = self.remove_init_edges
        if self.init_edges:
            params['init_edges'] = [('eigen0', 'y'), ('eigen1', 'y'), ('eigen2', 'y'), ('eigen3', 'y'), ('eigen4', 'y'),
                                    ('eigen5', 'y'), ('eigen6', 'y'), ('eigen7', 'y'), ('eigen8', 'y'), ('eigen9', 'y')]
        if self.white_list:
            params['white_list'] = [('eigen0', 'y'), ('eigen1', 'y'), ('eigen2', 'y'), ('eigen3', 'y'), ('eigen4', 'y'),
                                    ('eigen5', 'y'), ('eigen6', 'y'), ('eigen7', 'y'), ('eigen8', 'y'), ('eigen9', 'y')]
        #   params = {'init_edges': [('eigen0', 'y'), ('eigen1', 'y'), ('eigen2', 'y'), ('eigen3', 'y'), ('eigen4', 'y'),
        #                           ('eigen5', 'y'), ('eigen6', 'y'), ('eigen7', 'y'), ('eigen8', 'y'), ('eigen9', 'y')],
        #           'remove_init_edges': False,
        #          'white_list': [('eigen0', 'y'), ('eigen1', 'y'), ('eigen2', 'y'), ('eigen3', 'y'), ('eigen4', 'y'),
        #                        ('eigen5', 'y'), ('eigen6', 'y'), ('eigen7', 'y'), ('eigen8', 'y'), ('eigen9', 'y')]}

        bn.add_edges(discretized_data, scoring_function=(self.score_func, self.score), params=params)

        bn.calculate_weights(discretized_data)
        bn.plot('BN1.html')
        return bn

    def convolve(self, dataset, weights, left_vertices):
        new_Data = []
        for graph in dataset:
            A = to_dense_adj(graph.edge_index)
            eigs = torch.eig(A.reshape(A.shape[1], A.shape[2]), True)
            eigenvectors = eigs[1]
            eig = eigs[0].T[0].T
            ordered, indices = torch.sort(eig[:graph.num_nodes], descending=True)
            lef = indices[left_vertices]
            zeroed = torch.tensor(list(set(range(len(eig))) - set(lef.tolist())))
            eig[zeroed] = 0

            for e, d in enumerate(lef):
                eig[d] = eig[d] * weights[e]

            eigenvalues = torch.diag(eig)
            convolved = torch.matmul(torch.matmul(eigenvectors, eigenvalues), eigenvectors.T)
            new_A = convolved.type(torch.DoubleTensor)
            graph.edge_index, graph.edge_weight = dense_to_sparse(convolved)
            graph.edge_weight = graph.edge_weight  # .type(torch.FloatTensor)
            graph.edge_index = graph.edge_index.type(torch.LongTensor)
            # print((graph.edge_weight).dtype)
            new_Data.append(graph)
        return new_Data
