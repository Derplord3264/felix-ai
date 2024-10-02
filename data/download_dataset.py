from datasets import load_dataset

def download_dataset():
    dataset = load_dataset('daily_dialog')
    dataset.save_to_disk('daily_dialog')

if __name__ == "__main__":
    download_dataset()
