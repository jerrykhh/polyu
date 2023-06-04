import torch.nn as nn

CNNs = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(in_features=3*3*32, out_features=64),
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=10)
)

# # convolutional layer 1
# conv_layer1 = nn.Sequential(
#     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
#     nn.ReLU(),
# )

# # convolutional layer 2
# conv_layer2 = nn.Sequential(
#     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
#     nn.ReLU(),
# )

# # fully connected layer 1
# fc_layer1 = nn.Sequential(
#     nn.Linear(in_features=16*5*5, out_features=120),
#     nn.ReLU(),
# )

# # fully connected layer 2
# fc_layer2 = nn.Sequential(
#     nn.Linear(in_features=120, out_features=64),
#     nn.ReLU(),
# )

# # fully connected layer 3
# fc_layer3 = nn.Linear(in_features=84, out_features=10)

# LeNet5 = nn.Sequential(
#     conv_layer1,
#     nn.MaxPool2d(kernel_size=2),
#     conv_layer2,
#     nn.MaxPool2d(kernel_size=2),
#     nn.Flatten(), # flatten
#     fc_layer1,
#     fc_layer2,
#     fc_layer3
# )
