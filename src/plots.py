import matplotlib.pyplot as plt
import ast  # To safely parse string to list

def plot_boxes(image, df, threshold=0.5):
    """
    Plot boxes on an image using bounding box data from a DataFrame.

    Parameters:
    - image: PIL.Image object
    - df: pandas.DataFrame with columns ['label', 'score', 'box']
    - threshold: float, minimum score to plot a box
    """
    # Filter by score threshold
    filtered_df = df[df["score"] > threshold]

    if filtered_df.empty:
        print("No boxes above threshold.")
        return

    # Parse box strings into lists of floats # Not needed
    #filtered_df["box"] = filtered_df["box"].apply(lambda b: ast.literal_eval(b))

    # Get labels and boxes
    labels = filtered_df["label"].tolist()
    boxes = filtered_df["box"].tolist()

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


if __name__ == "__main__":
    import pandas as pd
    from PIL import Image

    # Example usage
    image_path = "data/tower2.jpg"
    df_path = "results/results.csv"

    image = Image.open(image_path)
    df = pd.read_csv(df_path)

    # Ensure 'box' column is parsed correctly
    df["box"] = df["box"].apply(ast.literal_eval)

    plot_boxes(image, df, threshold=0.15)