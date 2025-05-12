# Importing all required Libraries Below
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sqlite_db import SQLDb
from PIL import Image

# Set the device for PyTorch execution
# This ensures that GPU acceleration is used if available, falling back to CPU otherwise
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define batch size for training and evaluation
batch_size = 32

# Load the MNIST dataset
# - train_dataset: 60,000 training examples with labels
# - validation_dataset: 10,000 test examples with labels
# Both are transformed to PyTorch tensors with pixel values normalized to [0,1]
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
validation_dataset = datasets.MNIST("./data", train=False, transform=transforms.ToTensor())

# Create DataLoader objects to efficiently batch and shuffle the data
# - train_loader: Shuffles data to improve training convergence
# - validation_loader: No need to shuffle validation data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)


def train(epoch, log_interval=200):
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number for logging
        log_interval (int): How often to print progress (in batches)
    """
    # Set model to training mode (enables dropout, batch normalization adjustments, etc.)
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to the appropriate device (GPU/CPU)
        data = data.to(device)
        target = target.to(device)

        # Reset gradients for this batch iteration
        # This is necessary because gradients accumulate by default
        optimizer.zero_grad()

        # Forward pass: compute model predictions
        output = model(data)

        # Calculate loss between predictions and true labels
        loss = criterion(output, target)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update model parameters based on gradients and optimizer settings
        optimizer.step()

        # Occasionally print training progress
        if batch_idx % log_interval == 0:
            print(
                f"""Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.data.item()}"""
            )


def validate(loss_vector, accuracy_vector):
    """
    Evaluate the model on validation data and capture intermediate activations for TSNE visualization.

    This function:
    1. Evaluates model performance (loss and accuracy)
    2. Collects penultimate layer activations for dimensionality reduction
    3. Performs TSNE to convert high-dimensional activations to 3D points
    4. Stores results in SQLite database for later visualization

    Args:
        loss_vector (list): List to append validation loss for plotting
        accuracy_vector (list): List to append validation accuracy for plotting
    """
    # Set model to evaluation mode (disables dropout, freezes batch norm statistics)
    model.eval()

    # Initialize metrics
    val_loss, correct = 0, 0

    # Evaluate model on validation data without computing gradients (saves memory and computation)
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()

        # Get predicted class (index of maximum logit)
        pred = output.data.max(1)[1]

        # Count correct predictions
        correct += pred.eq(target.data).cpu().sum()

    # Calculate average validation loss
    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    # Calculate accuracy percentage
    accuracy = 100.0 * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    # Print validation metrics
    print(
        f"\nValidation set: Average loss: {val_loss}, Accuracy: {correct}/{len(validation_loader.dataset)} {accuracy}%\n"
    )

    # ============= TSNE Visualization Section =============
    # Access the penultimate layer (fc1) activations using a forward hook
    global view_output

    def hook_fn(module, input, output):
        """
        Forward hook function to capture intermediate layer outputs.
        Will store fc1 layer outputs in the global view_output variable.
        """
        global view_output
        view_output = output

    # Register the hook to capture fc1 layer outputs
    hook = model.fc1.register_forward_hook(hook_fn)

    # Initialize storage for the TSNE visualization data
    all_outputs = []  # fc1 layer activations
    all_labels = []  # true labels
    all_preds = []  # predicted labels
    all_ids = []  # unique identifier for each datapoint
    id = 1

    # Collect intermediate activations, true labels, and predictions
    for data, target in validation_loader:
        data = data.to(device)

        # Forward pass through the model
        output = model(data)

        # Get predicted class
        pred = output.data.max(1)[1]

        # Store predictions, activations, and true labels
        all_preds.extend(pred.to("cpu").numpy())
        all_outputs.extend(view_output.to("cpu").detach().numpy())
        all_labels.extend(target.to("cpu").numpy())

        # Assign unique IDs to each datapoint for tracking in visualization
        for _ in data:
            all_ids.append(id)
            id += 1

    # Convert activations to numpy array
    X_numpy = np.array(all_outputs, dtype=np.float32)

    # Apply TSNE dimensionality reduction to get 3D coordinates
    # This converts high-dimensional fc1 layer outputs (256-dimensional)
    # to 3D points that can be visualized in the Manifolk interface
    X_embedded = TSNE(n_components=3).fit_transform(X_numpy)

    # Convert all data to strings for database storage
    all_labels = [str(each_label) for each_label in all_labels]
    all_preds = [str(each_pred) for each_pred in all_preds]
    all_ids = [str(each_id) for each_id in all_ids]

    # Store epoch, TSNE coordinates, labels, predictions, and IDs in the database
    # This will be used by the Manifolk visualization dashboard
    log_db.insert(epoch, X_embedded, all_labels, all_preds, all_ids)

    # Remove the hook after data collection
    hook.remove()


class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.

    Architecture:
    - 3 convolutional layers with ReLU activations and pooling
    - 2 fully-connected layers
    - Dropout regularization to prevent overfitting
    - Log softmax output for 10 digit classes
    """

    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)

        # Second convolutional layer: 32 input channels, 32 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)

        # Third convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)

        # First fully-connected layer: 3*3*64 = 576 input features (after convolutions and pooling)
        # to 256 hidden units
        self.fc1 = nn.Linear(3 * 3 * 64, 256)

        # Output layer: 256 input features to 10 classes (one per digit)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]
               (MNIST images are 28x28 pixels with 1 channel)

        Returns:
            Log probabilities for each of the 10 digit classes
        """
        # First conv layer with ReLU activation
        x = F.relu(self.conv1(x))

        # Second conv layer with ReLU activation and 2x2 max pooling (reduces spatial dimensions)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # Dropout to prevent overfitting
        x = F.dropout(x, p=0.5, training=self.training)

        # Third conv layer with ReLU activation and 2x2 max pooling
        x = F.relu(F.max_pool2d(self.conv3(x), 2))

        # Dropout again
        x = F.dropout(x, p=0.5, training=self.training)

        # Flatten the tensor for the fully-connected layers
        # Input shape after convolutions and pooling is [batch_size, 64, 3, 3]
        # After flattening: [batch_size, 3*3*64]
        x = x.view(-1, 3 * 3 * 64)

        # First fully-connected layer with ReLU
        x = F.relu(self.fc1(x))

        # Final dropout
        x = F.dropout(x, training=self.training)

        # Output layer
        x = self.fc2(x)

        # Log softmax activation for classification
        return F.log_softmax(x, dim=1)


# Initialize the CNN model and move it to the appropriate device (GPU/CPU)
model = CNN().to(device)

# Print model architecture
print(model)

# Setup the optimizer with Stochastic Gradient Descent
# Learning rate of 0.01 and momentum of 0.5 for faster convergence
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Define the loss function (Cross Entropy Loss is standard for classification tasks)
criterion = nn.CrossEntropyLoss()

# Print model architecture again (redundant but kept for consistency)
print(model)

# Set number of training epochs
epochs = 10

# Save validation images for reference
# This creates image files from the MNIST validation set
save_id = 1
for data, target in validation_loader:
    for each_data in data:
        im = ToPILImage()(each_data)
        im.save(f"validation_images/{save_id}.jpg")
        save_id += 1

# Initialize tracking variables
id = 1  # Counter for datapoint IDs
lossv_cnn, accv_cnn = [], []  # Storage for loss and accuracy history

# Initialize the SQLite database connection through the Manifolk SQLDb class
# This will create a table with a timestamp-based name for storing TSNE visualization data
log_db = SQLDb(table_name="mnist")

# Main training loop
for epoch in range(1, epochs + 10):
    # Train for one epoch
    train(epoch)

    # Validate model and store TSNE visualization data
    validate(lossv_cnn, accv_cnn)

# Close the database connection when finished
log_db.close_connection()
