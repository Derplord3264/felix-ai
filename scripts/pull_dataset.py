import pandas as pd

def pull_dataset():
    # Example dataset URL (replace with a good conversational dataset URL)
    url = "https://example.com/dataset.csv"
    df = pd.read_csv(url)
    df.to_csv('data/dataset.csv', index=False)
    print("Dataset pulled successfully.")

if __name__ == "__main__":
    pull_dataset()
