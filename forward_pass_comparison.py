"""
Forward-pass verification script

This script:
1. Defines a fully connected PyTorch MLP
2. Initializes deterministic weights
3. Extracts weights and biases
4. Runs a homemade forward pass
5. Compares outputs numerically

If outputs match (error ~1e-7), your forward pass is correct.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------
# 1. PyTorch reference model
# -----------------------------
class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5,3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -----------------------------
# 2. Deterministic initialization
# -----------------------------
torch.manual_seed(0)

model = TorchMLP()

for param in model.parameters():
    nn.init.uniform_(param, -0.5, 0.5)

model.eval()  # VERY important


# -----------------------------
# 3. Extract weights & biases
# -----------------------------
W1 = model.fc1.weight.detach().numpy()  # (out, in)
b1 = model.fc1.bias.detach().numpy()

W2 = model.fc2.weight.detach().numpy()
b2 = model.fc2.bias.detach().numpy()

W3 = model.fc3.weight.detach().numpy()
b3 = model.fc3.bias.detach().numpy()

# -----------------------------
# 4. Homemade NN forward pass
# -----------------------------
from NeuralNetwork import NeuralNetwork

def relu(x):
    return np.maximum(0, x)
linear = lambda x: x

diy_network = NeuralNetwork.NeuralNetwork(4,[5,5],3,[relu,relu,linear])
diy_network.set_weights([(W1,b1),(W2,b2),(W3,b3)])
print([(W1,b1),(W2,b2),(W3,b3)])
# -----------------------------
# 5. Run comparison test
# -----------------------------
import tqdm 

if __name__ == "__main__":
    num_tests = 10000
    for i in tqdm.tqdm(range(num_tests)):
        # Test input
        x = torch.tensor([np.random.randn(4)], dtype=torch.float32)

        # PyTorch output
        torch_out = model(x).detach().numpy()

        # Homemade output
        home_out = diy_network.forward(x.numpy())

        # Results
        # print("Input:")
        # print(x)
        # print()

        # print("PyTorch output:")
        # print(torch_out)
        # print()

        # print("Homemade output:")
        # print(home_out)
        # print()

        # print("Absolute difference:")
        # print(np.abs(torch_out - home_out))
        # print()

        # print("Max error:", np.max(np.abs(torch_out - home_out)))

        # Pass / fail check
        assert np.allclose(torch_out, home_out, atol=1e-6), "\n❌ Forward pass MISMATCH"

