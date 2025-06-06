import requests
from PIL import Image
import torch
import yaml
import pandas as pd
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from plots import plot_boxes
from object_detection import detect_objects
from analysis import count_objects

# Load the configuration file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

processor_name = config.get("processor_name")
model_name = config.get("model_name")

threshold = config.get("threshold", 0.5)

# Load an image locally
filepath = "data/tower2.jpg"
image = Image.open(filepath)

# image.show()
texts = [["antenna", "panel", "tower"]]

results = detect_objects(model_name=model_name,
                        processor_name=processor_name,
                        images=image,
                        labels=texts)

print(results)

# Process the results
i = 0  # Assuming we are processing the first image
text = texts[i]


boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Convert tensor to named labels
named_labels = [text[label] for label in labels]

# Print result
for box, score, label in zip(boxes, scores, named_labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")

# Save result in csv

df = pd.DataFrame({
    "label": named_labels,
    "score": [round(score.item(), 3) for score in scores],
    "box": [box.tolist() for box in boxes]
})
df.to_csv("results/results.csv", index=False)

# Display the image with bounding boxes
plot_boxes(image, df, threshold=threshold)

# Analyze the results
count_objects(df, threshold=threshold)