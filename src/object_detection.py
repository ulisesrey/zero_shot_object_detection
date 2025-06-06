import requests
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection


def detect_objects(model_name,
                   processor_name,
                   images,
                   texts):
    """
    Detect objects in an image using the OWL-ViT model.
    Args:
        model_name (str): The name of the pre-trained OWL-ViT model.
        processor_name (str): The name of the processor for the OWL-ViT model.
    Returns:
        results (list): A list of dictionaries containing bounding boxes, scores, and labels for detected objects.
    """
    # Load the pre-trained OWL-ViT model and processor
    processor = Owlv2Processor.from_pretrained(processor_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name)

 
    inputs = processor(text=texts, images=images, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([images.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to Pascal VOC Format (xmin, ymin, xmax, ymax)
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

    return results
