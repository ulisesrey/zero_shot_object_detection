import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

def plot_boxes(image, boxes, labels):
    # Assign unique colors to label types
    unique_labels = sorted(set(labels))
    colormap = plt.get_cmap("tab10_r")  # or "tab20", "Set3", etc.
    label_colors = {label: colormap(i % colormap.N) for i, label in enumerate(unique_labels)}

    plt.imshow(image)
    ax = plt.gca()

    for box, label in zip(boxes, labels):
        x0, y0, x1, y1 = box
        width, height = x1 - x0, y1 - y0
        color = label_colors[label]
        
        rect = plt.Rectangle((x0, y0), width, height, fill=False, color=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, y0, label, color='white', fontsize=12,
                bbox=dict(facecolor=color, alpha=0.5))

    plt.axis('off')
    plt.show()
