import torch
from torchinfo import summary  # or torchsummary if you prefer
import configs.BEFUnet_Config as configs
from models.BEFUnet import BEFUnet3D

# -------------------------------
# 🔹 Device setup
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 🔹 Load Model Configuration
# -------------------------------
CONFIGS = {'BEFUnet3D': configs.get_BEFUnet_configs()}

# -------------------------------
# 🔹 Initialize the Model
# -------------------------------
model = BEFUnet3D(config=CONFIGS['BEFUnet3D'], n_classes=4).to(DEVICE)
print("✅ 3D BEFUnet model initialized successfully.\n")

# -------------------------------
# 🔹 Report Model Summary
# -------------------------------
print("🔍 Model Summary (using torchinfo):\n")
summary(model, input_size=(1, 4, 96, 96, 96))  # (batch, channels, depth, height, width)

# -------------------------------
# 🔹 Count Parameters
# -------------------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 Total Parameters: {total_params:,}")
print(f"🧠 Trainable Parameters: {trainable_params:,}")

if torch.cuda.is_available():
    print(f"💻 Device: CUDA ({torch.cuda.get_device_name(0)})")
else:
    print("💻 Device: CPU")
