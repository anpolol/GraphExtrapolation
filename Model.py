from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
from DataPreparing import DataPreparing
from torch_geometric.loader import DataLoader
import torch
import matplotlib.pyplot as plt
from CausalProcess import CausalProcess
import optuna

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes, dropout, size, device= 'cuda'):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.dropout = dropout
        self.size = size
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))
        self.device = device
        for s in range(size-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < self.size-1:
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x


class TrainingModel():
    def __init__(self, name: str, causal: bool, epoch: int, device: str, init_edges: bool, remove_init_edges: bool, white_list: bool, score_func: str = None, ):
        self.epoch = epoch
        self.dataobj = DataPreparing(name)
        self.train_dataset, self.test_dataset = self.dataobj.split_env(0.8)
        self.device = device
        if causal:
            causality = CausalProcess(self.train_dataset, self.test_dataset, score_func=score_func, init_edges=init_edges, remove_init_edges=remove_init_edges, white_list=white_list)
            self.train_dataset, self.test_dataset = causality.run()

        #self.train_dataset, self.test_dataset = self.train_dataset.to(self.device), self.test_dataset.to(self.device)
        super().__init__()

    def train(self, model, train_loader, criterion, optimizer):
        model.train()
        total_loss = 0
        for data in train_loader:  # Iterate in batches over the training dataset.
            data=data.to(self.device)
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            total_loss += loss
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        return total_loss / len(train_loader)

    def test(self, model, loader):
        model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data=data.to(self.device)
            out = model(data.x, data.edge_index, data.edge_weight, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    def plot(self, losses, accs_train, accs_test):
        print('loss at the last epoch', losses[len(losses) - 1])
        plt.plot(losses)
        plt.title('loss')
        plt.show()

        print('train acc at the last epoch',accs_train[len(accs_train) - 1])
        plt.plot(accs_train)
        plt.title('train acc')
        plt.show()

        print('test acc at the last epoch', accs_test[len(accs_test) - 1])
        plt.plot(accs_test)
        plt.title('test acc')
        plt.show()

    def run(self, params):
        hidden_layer = params['hidden_layer']
        dp = params['dropout']
        size = params['size of network, number of convs']
        learning_rate = params['lr']

        model = GCN(hidden_channels=hidden_layer, num_node_features=self.train_dataset[0].num_node_features,num_classes=self.dataobj.num_classes, dropout=dp,size=size)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=True)
        losses = []
        accs_train = []
        accs_test = []
        for epoch in range(self.epoch):
            loss = self.train(model, train_loader, criterion,optimizer)
            train_acc = self.test(model, train_loader)
            test_acc = self.test(model, test_loader)
            losses.append(loss.item())
            accs_train.append(train_acc)
            accs_test.append(test_acc)
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        self.plot(losses,accs_train,accs_test)


class TrainOptuna(TrainingModel):
    def objective(self,trial):
        # варьируем параметры

        train_dataset, val_dataset = self.train_dataset[:int(len(self.train_dataset)*0.7)], self.train_dataset[int(len(self.train_dataset)*0.7):]

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        hidden_layer = trial.suggest_categorical("hidden_layer", [32, 64, 128, 256])
        dp = trial.suggest_float("dropout", 0.0, 0.5, step=0.1)
        size = trial.suggest_categorical("size of network, number of convs", [1, 2, 3])
        learning_rate = trial.suggest_float("lr", 5e-3, 1e-2)

        model = GCN(hidden_channels=hidden_layer, num_node_features=self.train_dataset[0].num_node_features, num_classes=self.dataobj.num_classes, dropout=dp,size=size)
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.epoch):
            loss = self.train(model,train_loader,criterion,optimizer)

        val_acc = self.test(model,val_loader)

        return val_acc


    def run(self, number_of_trials):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=number_of_trials)
        trial = study.best_trial
        print(" Value: ", trial.value)
        return trial.params