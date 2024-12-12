import os

try:
    import pandas as pd
except ImportError:
    print("You need to install pandas")
    exit()

try:
    import seaborn as sns
except ImportError:
    print("You need to install seaborn")
    exit()

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("You need to install matplotlib")
    exit()

try:
    import numpy as np
except ImportError:
    print("You need to install numpy")
    exit()

try:
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    print("You need to install imbalanced-learn")
    exit()


def load_data(features_path, labels_path):
    features = pd.read_csv(os.path.abspath(features_path))
    labels = pd.read_csv(os.path.abspath(labels_path))
    return features, labels

def save_correlation_matrix(df, labels, filename):
    features_with_labels = df.copy()
    features_with_labels['Label'] = labels
    corr = features_with_labels.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.savefig(filename)
    print(f"Correlation matrix saved to {filename}")

def resample_data(features, labels):
    rus = RandomUnderSampler(random_state=17, sampling_strategy='majority')
    features_resampled, labels_resampled = rus.fit_resample(features, labels)
    return features_resampled, labels_resampled

def clean_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    # Create the upper triangle of the correlation matrix (no need to keep the lower triangle and diagonal)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop)

def save_data(features, labels, features_path, labels_path):
    features.to_csv(features_path, index=False)
    labels.to_csv(labels_path, index=False)


if __name__ == "__main__":
    features, labels = load_data('data/features.csv', 'data/labels.csv')
    save_correlation_matrix(features, labels, 'data/corr_matrix.png')
    
    features_resampled, labels_resampled = resample_data(features, labels)
    save_correlation_matrix(features_resampled, labels_resampled, 'data/corr_matrix_resampled.png')

    features = clean_correlated_features(features)
    save_correlation_matrix(features, labels, 'data/corr_matrix_cleaned.png')
    
    save_data(features_resampled, labels_resampled, 'data/features_cleaned.csv', 'data/labels_cleaned.csv')