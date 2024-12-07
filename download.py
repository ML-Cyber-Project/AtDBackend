import kagglehub

def download_dataset() -> str:
    # Download latest version and return path of downloaded files
    return kagglehub.dataset_download("chethuhn/network-intrusion-dataset")


if __name__ == "__main__":
    print(f"Path to dataset: {download_dataset()}")
