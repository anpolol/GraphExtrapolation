
from Model import GCN, TrainingModel, TrainOptuna

from datetime import datetime


if __name__ == "__main__":
    dt = datetime.now()
    causal = True
    number_of_trials = 100
    name = 'BACE'
    epoch = 50
    device = 'cuda'
    MO = TrainOptuna(name=name, causal=causal, epoch=epoch, device=device, score_func='MI',white_list=True,init_edges=True,remove_init_edges=False)
    best_values = MO.run(number_of_trials=number_of_trials)

    M = TrainingModel(name=name, causal=causal, epoch=epoch, device=device, score_func='K2',white_list=False,init_edges=False,remove_init_edges=True)
    M.run(best_values)
    print(datetime.now()-dt)