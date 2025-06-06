import pandas as pd

# Count unique objects
def count_objects(df, threshold=0.5):
    """
    Count unique objects above a certain score threshold.
    
    Parameters:
    - df: DataFrame containing 'label' and 'score' columns.
    - threshold: Minimum score to consider an object as detected.
    
    Returns:
    - DataFrame with counts of unique labels above the threshold.
    """
    filtered_df = df[df["score"] > threshold]
    value_counts = filtered_df["label"].value_counts()

    print(f"These are the number unique objects detected above the threshold of {threshold}.")
    print(value_counts)
    return 
