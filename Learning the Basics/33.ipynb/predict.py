print(__name__)

import torch
import torchvision
import matplotlib.pyplot as plt
from typing import List
import argparse
import model_builder
from torchvision import transforms

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, default=None)
parser.add_argument("--file_path", type=str, default=None)

args = parser.parse_args()

model = model_builder.TinyVGG(input_shape=3, hidden_units=10, output_shape=3)
model.load_state_dict(torch.load(f=args.model_path))
file_path = args.file_path
print(f"Model : {model}\nFile path : {file_path}")

transform = transforms.Compose([
    transforms.Resize(size=(64,64))
])

class_names = ['pizza', 'steak', 'sushi']

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names : List[str] = class_names,
                        transform=transform,
                        device="cpu"):
    """Makes a prediction on a target image and plots the image with its prediction."""
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32) / 255

    if transform is not None:
        target_image = transform(target_image)
    
    model.to(device)

    model.eval()
    with torch.inference_mode():
        target_image = target_image.unsqueeze(0)
        target_image_pred = model(target_image.to(device))
    
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_labels = torch.argmax(target_image_pred_probs, dim=1)

    # plt.imshow(target_image.squeeze().permute(1,2,0))
    # if class_names:
    #     title = f"Prediction : {class_names[target_image_pred_labels.cpu()]} | Probability : {target_image_pred_probs.max().cpu():.3f}"

    # else:
    #     title = f"Prediction: {target_image_pred_labels} | Probability: {target_image_pred_probs.max().cpu():.3f}"
    
    # plt.title(title)
    # plt.axis("off")

    print(f"Prediction : {class_names[target_image_pred_labels]}\nProbability : {target_image_pred_probs.max()}")
    return
