
from Model import GCN, TrainingModel, TrainOptuna

from datetime import datetime


if __name__ == "__main__":
    dt = datetime.now()
    causal = True
    number_of_trials = 100
    name = 'BACE'
    epoch = 50
    device = 'cuda'
    white_list = True
    init_edges = True
    remove_init_edges = False

#    MO = TrainOptuna(name=name, causal=causal, epoch=epoch, device=device, score_func='MI', white_list=white_list, init_edges=init_edges, remove_init_edges=remove_init_edges)
 #   best_values = MO.run(number_of_trials=number_of_trials)

    best_values={'hidden_layer': 128, 'dropout': 0.4, 'size of network, number of convs': 2, 'lr': 0.008402517735246102}
    M = TrainingModel(name=name, causal=causal, epoch=epoch, device=device, score_func='K2', white_list=white_list, init_edges=init_edges, remove_init_edges=remove_init_edges)
    M.run(best_values)
    print(datetime.now()-dt)