import os
from download import download_dataset

try:
    import pandas as pd
except ImportError:
    print("You need to install pandas")
    exit()

try:
    import numpy as np
except ImportError:
    print("You need to install numpy")
    exit()

COLUMN_SIMILARITY_THRESHOLD = 95  # Drop columns with more than x% of same data - try 90 ?

def get_percentage_columns_similarity(df):
    return (1 - (df.nunique() / len(df))) * 100

def load_and_clean_data(data_path=download_dataset()):
    global_df = pd.DataFrame()
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, file))
                global_df = pd.concat([global_df, df], ignore_index=True)

    global_df.columns = global_df.columns.str.strip()

    # Drop columns according to similarity threshold
    columns_to_drop = get_percentage_columns_similarity(global_df)
    columns_to_drop = columns_to_drop[columns_to_drop > COLUMN_SIMILARITY_THRESHOLD].index
    # Don't drop the label column
    columns_to_drop = columns_to_drop.drop('Label')
    global_df = global_df.drop(columns=columns_to_drop)
    global_df = global_df.replace([float('-inf'), float('inf')], float('nan')).dropna()
    # drop duplicates
    global_df = global_df.drop_duplicates()

    return global_df



def save_features_and_labels(global_df, features_path='data/features.csv', labels_path='data/labels.csv'):
    # Separate label column from features
    global_df['Label'] = np.where(global_df['Label'].isin(['BENIGN']), 0, 1)
    print("Labels count : ", global_df['Label'].value_counts())
    labels = global_df['Label']
    features = global_df.drop(columns=['Label'])

    # Create features_path and labels_path's directory if it doesn't exist
    if not os.path.exists(os.path.dirname(features_path)):
        os.makedirs(os.path.dirname(features_path))
    if not os.path.exists(os.path.dirname(labels_path)):
        os.makedirs(os.path.dirname(labels_path))

    # Save to csv
    features.to_csv(features_path, index=False)
    labels.to_csv(labels_path, index=False)

    print(f"Features and labels saved to {features_path} and {labels_path}")


if __name__ == "__main__":
    global_df = load_and_clean_data()
    save_features_and_labels(global_df)
