from easy_whisper.data_loader import load_local_dataset
from easy_whisper.trainer import train_model
import multiprocessing

def main():
    # Your code here
    dataset = load_local_dataset(dataset_dir="test-data")
    model = train_model(
        dataset=dataset,
        model_name_or_path="openai/whisper-tiny",
        output_dir="./whisper-finetuned",
        safe_multiprocessing=True,
        num_workers=1
    )

if __name__ == "__main__":
    # Essential for Windows multiprocessing
    multiprocessing.freeze_support()
    main()