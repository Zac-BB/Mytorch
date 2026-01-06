
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 1


testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

with open('./TxtFiles/LabelsTest.txt', 'w') as f:
    for _, label in testset:
        f.write(str(label) + '\n')

classes = ('0', '1', '2', '3','4', '5', '6', '7', '8', '9')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # linear layer (784 -> 1 hidden node)
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # flatten image input
        # x = x.view(-1, 28 * 28)
        x = torch.flatten(x, 1)
        # add hidden layer, with relu activation function
        x = (F.relu(self.fc1(x)))
        x = (F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


fake_net = Net()
PATH = './cifar_net.pth'
fake_net.load_state_dict(torch.load(PATH, weights_only=True))



from NeuralNetwork import NeuralNetwork
from NeuralNetwork.activation import ReLU, Linear, Softmax



relu = ReLU()
linear = Linear()
softmax = Softmax()
net = NeuralNetwork.NeuralNetwork(28*28,[512,512],10,[relu,relu,linear])

W1 = fake_net.fc1.weight.detach().numpy()  # (out, in)
b1 = fake_net.fc1.bias.detach().numpy()

W2 = fake_net.fc2.weight.detach().numpy()
b2 = fake_net.fc2.bias.detach().numpy()

W3 = fake_net.fc3.weight.detach().numpy()
b3 = fake_net.fc3.bias.detach().numpy()
net.set_weights({"W":[W1,W2,W3],"b":[b1,b2,b3]})

def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """    
    I1 = Img
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    # I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1, axis=0)

    return I1Combined, I1

def Accuracy(Pred, GT):
    """
    Inputs: 
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return (np.sum(np.array(Pred)==np.array(GT))*100.0/len(Pred))


def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred
def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=('0', '1', '2', '3','4', '5', '6', '7', '8', '9'))
    disp.plot()
    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print('Accuracy: '+ str(Accuracy(LabelsPred, LabelsTrue)), '%')
    plt.show()




def TestOperation(model, testloader, LabelsPathPred):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.eval()

    preds = []
    from sklearn.utils import gen_batches
    # batchs = gen_batches(len())
    # with torch.no_grad():
    for image, _ in (testloader):
        image_np = image.numpy()
        image_flat = image_np.reshape([1,784])

        output = model.forward(image_flat)
        prediction = np.argmax(output)
        preds.append(prediction)

    with open(LabelsPathPred, 'w') as f:
        for p in preds:
            f.write(str(p) + '\n')



LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

import time


times = []
for i in tqdm(range(100)):
    start_time = time.perf_counter()
    TestOperation(net, testset, LabelsPathPred)
    end_time = time.perf_counter()
    times.append(end_time-start_time)

import matplotlib.pyplot as plt
import pickle
with open('no_batch.pkl', 'wb') as file:
    pickle.dump(times, file)

plt.hist(times)
plt.title("Times")
plt.xlabel("Func Times (Sec)")
plt.ylabel("Frequency")
plt.show()

LabelsPath = "./TxtFiles/LabelsTest.txt"

# Plot Confusion Matrix
LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
ConfusionMatrix(LabelsTrue, LabelsPred)