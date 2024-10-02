from datasets import load_dataset

def pull_dataset():
    # Load the convai2 dataset from Hugging Face
    dataset = load_dataset('convai2', split='train')
    dataset.to_csv('data/dataset.csv', index=False)
    print("Dataset pulled successfully.")

if __name__ == "__main__":
    pull_dataset()
