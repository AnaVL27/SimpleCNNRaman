import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRamanCNN(nn.Module):
    """
    Convolutional Neural Network 1D for bacteria classification and treatment assignment
    This network processes Raman Spectres with 10000 points of intensity
    to identify one of the 30 bacteria species in Stanford dataset (Ho et al., 2019).
    
    Attributes:
       conv1 (nn.Conv1d): First conoluvtional layer to detect simple peaks
       conv2 (nn.Conv1d): Second convolutional layer to detect patterns in complex peaks
       pool (nn.MaxPool1d): Downsampling pooling to reduce spatial dimensions to the half
       fc1 (nn.Linear): Abstraction step to reduce the original data information (32000) to 128 abstract concepts      representing bacteria features
       fc2 (nn.Linear): Output layer to give one of the 30 bacteria species
    """

    def __init__(self):
        """
        Initialices the CNN layers for 1D signal
        1000 points -> conv/pooling -> 500 points -> classification
        """
        super(SimpleRamanCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2) 
        self.fc1 = nn.Linear(32000, 128) #64 filters * 500 points
        self.fc2 = nn.Linear(128, 30) 

    def forward(self, x):
        """
        Defines the data flow
        Args:
            x (torch.Tensor): Input tensor with dimensions (Batch, 1, 1000)
        Returns:
            torch.Tensor: Log-probabilities for each of the 30 classes    
        """
        x = F.relu(self.conv1(x)) #First conv and ReLU activation
        x = self.pool(F.relu(self.conv2(x))) #Seconf conv, ReLU and pooling
        x = torch.flatten(x, 1) #flatting the resulting tensor
        x = F.relu(self.fc1(x)) #final classification
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) #Log-softmax for number stability

