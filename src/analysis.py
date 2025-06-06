import pandas as pd

# Count unique objects
def count_objects(df):
    """
    Count unique objects above a certain score threshold.
    
    Parameters:
    - df: DataFrame containing 'label' and 'score' columns.
    - threshold: Minimum score to consider an object as detected.
    
    Returns:
    - DataFrame with counts of unique labels above the threshold.
    """
    value_counts = df["label"].value_counts()

    print(f"These are the number unique objects detected above the threshold of {threshold}.")
    print(value_counts)
    return 

if __name__ == "__main__":
    # Example usage
    df_path = "results/results.csv"
    df = pd.read_csv(df_path)
    count_objects(df)