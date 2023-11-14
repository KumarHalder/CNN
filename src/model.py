import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.conv1 = nn.Conv2d(3, 32, 5)  #224x224
        self.batchNorm1 = nn.BatchNorm2d(32) 
        
        self.conv2 = nn.Conv2d(32, 64, 5) #110x110
        self.batchNorm2 = nn.BatchNorm2d(64) 
        
        self.conv3 = nn.Conv2d(64, 128, 5) #53x53
        self.batchNorm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3) #24x24
        self.batchNorm4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 256, 3) #11x11
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        self.fc1 = nn.Linear(256 * 9 * 9, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.pool(self.relu(self.batchNorm1(self.conv1(x))))
        x = self.pool(self.relu(self.batchNorm2(self.conv2(x))))
        x = self.pool(self.relu(self.batchNorm3(self.conv3(x))))
        x = self.pool(self.relu(self.batchNorm4(self.conv4(x))))
        x = self.conv5(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.relu(self.dropout(self.fc3(x)))
        x = self.fc4(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()
    print(model)
    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
