import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from src.run_model import run_model, _train
import time
from data.read_fr_dataset import M2

# data transformation to normalize data
# normalizing data helps your optimization algorithm converge more quickly
transform = transforms.Compose(
    [
        # transforms.Resize((1, 28 * 28)),
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
# training set split

training_dataset_size = [500, 1000, 1500, 2000
                         ]

running_time_list = []
accuracy_list = []

for i in training_dataset_size:
    trainset = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          download=True,
                                          transform=transform)
    trainset_updated = M2(trainset, i)
    # testing set split, this portion is similar to your assignment
    # but instead of passing in trainset you need to pass in an instance
    # of a custom data object which is provided in the starter code

    trainloader = torch.utils.data.DataLoader(trainset_updated,
                                              batch_size=10,
                                              shuffle=True,
                                              num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=10,
                                             shuffle=False,
                                             num_workers=2)
    # labeling the classes of images just like the digits of MNIST, or the
    # the breeds of dogs from DogNet

    classes = ['0 - zero', '1 - one', '2 - two', '3 - three',
               '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    # def imshow(img):
    #     img = img / 2 + 0.5  # have to unnormalize the images
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    #
    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    #
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            size_layer1 = 128
            size_layer2 = 64
            self.hidden1 = nn.Linear(28 * 28, size_layer1)
            self.hidden2 = nn.Linear(size_layer1, size_layer2)
            self.out = nn.Linear(size_layer2, 10)

        def forward(self, x):
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))
            x = self.out(x)
            x = F.log_softmax(x, dim=1)
            return x

    # def train_model(model, criterion, optimizer, trainloader, epochs=100, devloader=None, print_info=True):
    #     epochs_loss = []
    #     epoch_acc = []
    #     running_loss = 0.0
    #     # the length of the train loader will give you the number of mini-batches
    #     # not the cleanest solution but for now it generalizes well and avoids
    #     # computation that we don't need
    #     minibatches = len(trainloader)
    #     # moving the network onto the gpu/cpu
    #     model = model.to(device)
    #     for epoch in range(epochs):
    #         epoch_loss = 0.0
    #         for batch, labels in trainloader:
    #             # batch is a tensor with m elements, where each element is
    #             # a training example
    #
    #             # moving batch/labels onto the gpu/cpu
    #             batch, labels = batch.to(device), labels.to(device)
    #
    #             # zeroing the parameters of the model
    #             # becuase we want to optimize them
    #             optimizer.zero_grad()
    #
    #             # forward pass
    #             # getting the predictions from our model by passing in a mini-batch
    #             # the ouput will have shape (mini-batch-size, number-of-classes)
    #             # where each element of output is the probabliity of that example being
    #             # the classification correspoding to the index of the value
    #             output = model(batch)
    #             loss = criterion(output, labels)
    #
    #             # backward pass
    #             loss.backward()
    #
    #             # optimize the parameters
    #             optimizer.step()
    #
    #             # add the loss of a mini-batch to the list of epoch loss
    #             epoch_loss += loss.item()
    #
    #         #  after each epoch we need to average out the loss across all minibatches
    #         epochs_loss.append(epoch_loss / minibatches)
    #         # printing some info
    #         if print_info:
    #             print(f'Epoch: {epoch} Loss: {epoch_loss / minibatches}')
    #     return model, epoch_loss

    def test_model(model, testloader):
        # variables to keep count of correct labels and the total labels in a mini batch
        correct = 0
        total = 0
        # since we're testing the model we don't need to perform backprop
        with torch.no_grad():
            for batch, labels in testloader:
                batch, labels = batch, labels
                output = model(batch)
                # this gives us the index with the highest value outputed from the last layer
                # which coressponds to the most probable label/classification for an image
                predicted = torch.max(output.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    model = Net()
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    device = torch.device('cpu')

    # net, loss = train_model(model,
    #                         epochs=1,
    #                         criterion=nn.CrossEntropyLoss(),
    #                         optimizer=optim.SGD(net.parameters(), lr=0.01),
    #                         trainloader=trainloader
    #                         )
    start_time = time.time()
    model, loss, accuracy = run_model(model, running_mode='train', train_set=trainset_updated, valid_set=None,
                                      test_set=testset,
                                      batch_size=10, learning_rate=learning_rate, n_epochs=100, stop_thr=1e-4,
                                      shuffle=True)
    end_time = time.time()
    running_time = end_time - start_time

    running_time_list.append(running_time)
    accuracy_list.append(np.mean(accuracy['train']))

    # model, train_loss, train_accuracy = _train(model, trainloader, optimizer, device=device)
    # test_model(model, testloader)

plt.figure()
plt.plot(training_dataset_size, running_time_list, label='Running Time')
plt.title('Running Time Vs. Number of Training Examples')
plt.xlabel('Training Examples')
plt.ylabel('Running Time')
plt.legend(loc="best")
plt.savefig("Q1a.png")
plt.show()

print("To train MNIST training set of 2000 samples will take:",
      running_time_list[-1])
print("To train the full MNIST training set will take:",
      running_time_list[-1] * 60000 / training_dataset_size[-1])

plt.figure()
plt.plot(training_dataset_size, accuracy_list, label='Accuracy')
plt.title('Accuracy Vs. Number of Training Examples')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.savefig("Q1c.png")
plt.show()
