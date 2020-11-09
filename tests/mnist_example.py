# Importing all required Libraries Below
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sqlite_db import SQLDb
# Below we set the device where PyTorch would be running
# This below snippet ensures that if a GPU is present, it will use the GPU for training the model
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Declaring a batch size
batch_size = 32
# Get a training dataset by passing train=True and next get validation data by passing
# train=False. Next we have to convert them to a tensor so that it can be loaded into a
# torch DataLoader

train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())
validation_dataset = datasets.MNIST('./data',
                                    train=False,
                                    transform=transforms.ToTensor())
# Torch data Loader is used to load the data in batches according to the passed batch size
# the dataloader object can be used in for-loops to loop through data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)


def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU
        data = data.to(device)
        target = target.to(device)
        # Before the backward pass, use the optimizer object to update (which are the learnable
        # weights of the model) to zero all the gradients for the variables
        optimizer.zero_grad()
        # Pass data through the network
        output = model(data)
        # Calculate loss
        loss = criterion(output, target)
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        if batch_idx % log_interval == 0:
            # After certain intervals printout the current loss
            print(
                f"""Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.data.item()}""")


def validate(loss_vector, accuracy_vector):
    # In this function you pass a loss_vector list and accuracy_vector
    # This function will use latest model and test it on validation data
    # evaluate the loss and accuracy and append it appropriate list
    # which will be used for further plotting
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)
    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    print(f'\nValidation set: Average loss: {val_loss}, Accuracy: {correct}/{len(validation_loader.dataset)} {accuracy}%\n')
    # Start saving data for T-SNE
    # usually the penultimate layer is taken as input for T-SNE
    # I am going to do the same here
    global view_output
    def hook_fn(module, input, output):
        global view_output
        view_output = output
    hook = model.fc1.register_forward_hook(hook_fn)
    all_outputs = []
    all_labels = []
    all_preds = []
    all_ids = []
    id = 1
    for data, target in validation_loader:
        data = data.to(device)
        output = model(data)
        pred = output.data.max(1)[1]
        all_preds.extend(pred.to('cpu').numpy())
        all_outputs.extend(view_output.to('cpu').detach().numpy())
        all_labels.extend(target.to('cpu').numpy())
        all_ids.append(id)
        id += 1
    X_numpy = np.array(all_outputs, dtype=np.float32)
    X_embedded = TSNE(n_components=3).fit_transform(X_numpy)
    all_labels = [str(each_label) for each_label in all_labels]
    all_preds = [str(each_pred) for each_pred in all_preds]
    all_ids = [str(each_id) for each_id in all_ids]
    log_db.insert(epoch, X_embedded, all_labels, all_preds, all_ids)
    hook.remove()


class CNN(nn.Module):
    def __init__(self):
        # In this model rather than using an MLP we use a CNN
        # CNN are known to better perform on image classification
        # tasks. Here we add 3 convolution layers and at last we have a
        # layer consisting of 10 neurons with log softmax activation function.
        # Same as how it's in MLP
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3*3*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
# Initialize the model
model = CNN().to(device)
# from the PyTorch optim package we use an Optimizer that will update the weights of
# the model for us. Here we will use SGD; the package contains many other
# optimization algorithms. The first argument to the SGD constructor tells the
# optimizer which Tensors it should update.
print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
# The nn package also contains definitions of popular loss functions; in our
# case we will use Cross Entropy Loss as our loss function.
criterion = nn.CrossEntropyLoss()
# Print out the model
print(model)

epochs = 10
# below is the main train loop
# here we decide how many epochs we want the model to train
# After training for each epoch, we also validate the model
# and printout loss and accuracy statistics
# we save all of them in below lossv_cnn and accv_cnn lists
lossv_cnn, accv_cnn = [], []
log_db = SQLDb(table_name="mnist")
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv_cnn, accv_cnn)
log_db.close_connection()



