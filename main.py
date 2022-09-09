from DataPreparing import DataPreparing
from torch_geometric.loader import DataLoader
from IPython.display import Javascript
from Model import GCN
import torch
import matplotlib.pyplot as plt

dataobj = DataPreparing('BACE')
dataset = dataobj.dataset
train_dataset, test_dataset = dataobj.split_env(0.8)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = GCN(hidden_channels=64, num_node_features=dataset[0].num_node_features,num_classes = dataobj.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         total_loss+=loss
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
    return total_loss/len(train_loader)


def test(loader):
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


losses = []
accs_train = []
accs_test = []
for epoch in range(50):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    losses.append(loss.item())
    accs_train.append(train_acc)
    accs_test.append(test_acc)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

print(losses[len(losses)-1])
plt.plot(losses)
plt.title('loss')
plt.show()

print(accs_train[len(accs_train)-1])
plt.plot(accs_train)
plt.title('train acc')
plt.show()

print(accs_test[len(accs_test)-1])
plt.plot(accs_test)
plt.title('test acc')
plt.show()