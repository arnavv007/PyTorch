import torch
import os
import gradio as gr

from typing import Tuple, Dict
from model import Effnetb2_creator
from timeit import default_timer as timer

class_names = ["pizza", "steak", "sushi"]

effnetb2, effnetb2_transforms = Effnetb2_creator(num_classes=3)

effnetb2.load_state_dict(
    torch.load(
        f="../models/9_effnetb2_feature_extractor.pth",
        map_location=torch.device("cpu")
    )
)

def predict(img) -> Tuple[Dict, float]:
    """
    Transforms an image and makes a prediction. It returns the predicted probability and time taken to make the prediction.
    """

    start_timer = timer()
    img = effnetb2_transforms(img).unsqueeze(0)

    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)

    pred_labels_and_probs = {class_names[i] : pred_probs[0][i] for i in range(len(class_names))}
    pred_time = round(timer() - start_timer, 5)

    return pred_labels_and_probs, pred_time

title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor to classify an image into pizza, sushi or steak"
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="predictions"),
                             gr.Number(label="Prediction time (seconds)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article
)

demo.launch()
