"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import argparse
import os
import torch
import data_setup, engine, model_builder, utils
from utils import save_model

from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.003)
parser.add_argument("--hidden_units", type=int, default=8)
parser.add_argument("--train_dir", type=str, default="data/pizza_steak_sushi/train")
parser.add_argument("--test_dir", type=str, default="data/pizza_steak_sushi/test")

args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.lr

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f"Number of epochs : {NUM_EPOCHS}\nBatch size : {BATCH_SIZE}\nHidden Units : {HIDDEN_UNITS}\nLearning rate : {LEARNING_RATE}\nTraining directory : {train_dir}\nTesting directory : {test_dir}")

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
