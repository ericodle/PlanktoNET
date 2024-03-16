import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, input_dim)
    
    def forward(self, x):
        print("Input shape (MlpBlock):", x.shape)
        x = F.gelu(self.fc1(x))
        print("After fc1 and gelu (MlpBlock):", x.shape)
        x = self.fc2(x)
        print("After fc2 (MlpBlock):", x.shape)
        return x

class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.token_mixing = MlpBlock(tokens_mlp_dim, tokens_mlp_dim)
        self.channel_mixing = MlpBlock(tokens_mlp_dim, channels_mlp_dim)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=hidden_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=tokens_mlp_dim)
    
    def forward(self, x):
        print("Input shape (MixerBlock):", x.shape)
        y = self.layer_norm1(x)
        print("After layer_norm1 (MixerBlock):", y.shape)
        y = y.permute(0, 1, 2)
        y = self.token_mixing(y)
        print("After token_mixing (MixerBlock):", y.shape)
        y = y.permute(0, 1, 2)
        x = x + y
        y = self.layer_norm2(x)
        print("After layer_norm2 (MixerBlock):", y.shape)
        y = self.channel_mixing(y)
        print("After channel_mixing (MixerBlock):", y.shape)
        return x + y

class MlpMixer(nn.Module):
    def __init__(self, patches, num_classes, num_blocks, hidden_dim, tokens_mlp_dim, channels_mlp_dim, model_name=None):
        super(MlpMixer, self).__init__()
        self.patches = patches
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        
        self.stem = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patches, stride=patches)
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.pre_head_layer_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        print("Input shape (MlpMixer):", x.shape)
        x = self.stem(x)
        print("After stem (MlpMixer):", x.shape)
        x = x.flatten(2).transpose(1, 2)
        for i, block in enumerate(self.blocks):
            print(f"Block {i + 1} input shape (MlpMixer):", x.shape)
            x = block(x)
            print(f"Block {i + 1} output shape (MlpMixer):", x.shape)
        x = self.pre_head_layer_norm(x)
        print("After pre_head_layer_norm (MlpMixer):", x.shape)
        x = x.mean(dim=1)
        print("After mean pooling (MlpMixer):", x.shape)
        if self.num_classes:
            x = self.head(x)
            print("After linear head (MlpMixer):", x.shape)
        return x

# Define parameters
patches = 16
num_classes = 200  # Example number of classes
num_blocks = 32
hidden_dim = 256
tokens_mlp_dim = 256
channels_mlp_dim = 2048

# Instantiate the model
model = MlpMixer(patches=patches,
                 num_classes=num_classes,
                 num_blocks=num_blocks,
                 hidden_dim=hidden_dim,
                 tokens_mlp_dim=tokens_mlp_dim,
                 channels_mlp_dim=channels_mlp_dim)

# Print the model architecture
print(model)

import torch
from torchvision import transforms
from PIL import Image

# Define the path to your image
image_path = "/home/eo/Desktop/PlanktoNET-main/mclanelabs/mclanelabs_set/amoeba/IFCB1_2006_270_170728_01626.png"

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),           # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])

# Load and preprocess the image
image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the input tensor to the device
input_batch = input_batch.to(device)

# Move the model to the same device
model = model.to(device)

# Pass the input through the model
with torch.no_grad():
    model.eval()
    output = model(input_batch)

# Get the predicted class
_, predicted_class = torch.max(output, 1)

# Print the predicted class
print("Predicted class:", predicted_class.item())

