import requests
from PIL import Image
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection

processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-finetuned")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-finetuned")

# Load an image locally
filepath = "data/tower.jpg"
image = Image.open(filepath)
# image.show()
texts = [["antenna", "panel", "tower"]]
inputs = processor(text=texts, images=image, return_tensors="pt")

with torch.no_grad():
  outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")


# Display the image with bounding boxes
import matplotlib.pyplot as plt
def plot_boxes(image, boxes, labels):
    plt.imshow(image)
    ax = plt.gca()
    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = box
        width, height = x1 - x0, y1 - y0
        rect = plt.Rectangle((x0, y0), width, height, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, y0, label, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    plt.show()
plot_boxes(image, boxes, [text[label] for label in labels])