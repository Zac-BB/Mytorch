import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 1

trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)



from NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuralNetwork.activation import relu, linear, softmax
from NeuralNetwork.optimizer import SGD
from NeuralNetwork.loss_function import CrossEntropyLoss

criterion = CrossEntropyLoss()
model = NeuralNetwork(2,[],3,[softmax])
optimizer = SGD(model,criterion,0.0001)
num_epochs = 50


for i in range(num_epochs):
    for i,data in enumerate(trainloader):

        inputs, labels = data
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.backward()
        optimizer.step()
