import requests
from PIL import Image
import torch
import yaml
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from plots import plot_boxes
from object_detection import detect_objects

# Load the configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

processor_name = config.get("processor_name")
model_name = config.get("model_name")

# Load an image locally
filepath = "data/tower.jpg"
image = Image.open(filepath)
# image.show()
labels = [["antenna", "panel", "tower"]]

results = detect_objects(model_name=model_name,
                        processor_name=processor_name,
                        images=image,
                        labels=labels)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
# text = labels[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {label[i]} with confidence {round(score.item(), 3)} at location {box}")


# Display the image with bounding boxes
plot_boxes(image, boxes, [text[label] for label in labels])