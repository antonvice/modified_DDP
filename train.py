import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CelebA
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from model import DDP

# Set device to GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
batch_size = 64
num_epochs = 10
lr = 1e-4
num_diffusion_steps = 100

# Define transforms for data preprocessing
transform = transforms.Compose([
    transforms.CenterCrop(128),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the CelebA dataset
train_dataset = CelebA(root="./data", split="train", transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the DDP model
model = DDP(input_dim=64*64*3, output_dim=64*64*3, num_diffusion_steps=num_diffusion_steps, hidden_dim=256)
model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs = data[0].view(-1, 64*64*3).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Print statistics every 100 batches
        if i % 100 == 99:
            print(f"Epoch {epoch+1}, Batch {i+1}: Loss {running_loss/100}")
            running_loss = 0.0

print("Finished training")
