import os
import csv
import concurrent.futures
from functools import lru_cache
from datasets import Audio, Dataset, DatasetDict
from pydub import AudioSegment
from typing import List, Optional
from sklearn.model_selection import train_test_split
from easy_whisper import logger
from datasets import load_dataset

@lru_cache(maxsize=128)
def is_wav_file(audio_path: str) -> bool:
    """
    Check if the given audio file is in WAV format.

    This function is cached for performance to avoid repeatedly
    checking the same file extensions.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        bool: True if the file has a .wav extension, False otherwise.
    """
    return audio_path.lower().endswith(".wav")


def convert_audio_to_wav(audio_path: str) -> str:
    """
    Convert an audio file to WAV format if it is not already in WAV format.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        str: The path to the converted WAV file (or the original path if already WAV).

    Raises:
        Exception: If the conversion fails for any reason.
    """
    try:
        if not is_wav_file(audio_path):
            logger.info(f"Converting '{audio_path}' to WAV format.")
            audio = AudioSegment.from_file(audio_path)
            wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
            audio.export(wav_path, format="wav")
            return wav_path
        return audio_path
    except Exception as e:
        logger.error(f"Failed to convert '{audio_path}' to WAV: {e}")
        raise


def process_audio_batch(audio_paths: List[str]) -> List[str]:
    """
    Convert a batch of audio files to WAV format.

    This function processes each audio file in a list sequentially,
    converting them to WAV if necessary.

    Args:
        audio_paths (List[str]): A list of paths to audio files.

    Returns:
        List[str]: A list of paths to the resulting WAV files.
    """
    converted_paths = []

    for path in audio_paths:
        try:
            converted_path = convert_audio_to_wav(path)
            converted_paths.append(converted_path)
        except Exception as e:
            logger.warning(f"Skipping file '{path}' due to conversion error: {e}")

    return converted_paths


def process_audio_files_parallel(
    audio_paths: List[str], max_workers: Optional[int] = None
) -> List[str]:
    """
    Convert multiple audio files to WAV format in parallel using a thread pool.

    Args:
        audio_paths (List[str]): A list of paths to audio files.
        max_workers (Optional[int]): The maximum number of worker threads to use. If None,
            it is set to min(32, os.cpu_count() + 4).

    Returns:
        List[str]: A list of paths to the resulting WAV files.
    """
    if not audio_paths:
        return []

    # Determine optimal number of workers if not specified
    if max_workers is None:
        max_workers = min(32, os.cpu_count() + 4)

    # Split files into batches for better progress tracking
    batch_size = max(1, len(audio_paths) // max_workers)
    batches = [
        audio_paths[i : i + batch_size] for i in range(0, len(audio_paths), batch_size)
    ]

    converted_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(process_audio_batch, batch): batch for batch in batches
        }

        for future in concurrent.futures.as_completed(future_to_batch):
            batch_results = future.result()
            converted_paths.extend(batch_results)

    return converted_paths


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.2,
    random_state: int = 42
) -> DatasetDict:
    """
    Split a Hugging Face Dataset into training and test sets using train_test_split.

    Args:
        dataset (Dataset): The Hugging Face Dataset to split.
        test_size (float, optional): The fraction of data to include in the test split.
            Defaults to 0.2 (20%).
        random_state (int, optional): The random seed for reproducible splitting.
            Defaults to 42.

    Returns:
        DatasetDict: A dictionary-like object containing 'train' and 'test' splits of the dataset.
    """
    df = dataset.to_pandas()

    # Split the DataFrame
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=None,  # Add stratification if needed
    )

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    return DatasetDict({"train": train_dataset, "test": test_dataset})


def load_local_dataset(
    audio: Optional[List[str]] = None,
    transcription: Optional[List[str]] = None,
    dataset_dir: Optional[str] = None,
    max_workers: Optional[int] = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> DatasetDict:
    """
    Load or create a local Hugging Face DatasetDict with audio-transcription pairs.

    This function supports two approaches:
    1. Directly provide lists of `audio` file paths and corresponding `transcription`s.
    2. Provide a `dataset_dir` containing a `metadata.csv` file (and audio files).

    The function attempts to load the data using the `audiofolder` format when possible,
    and falls back to manual CSV creation and dataset creation if necessary.

    Args:
        audio (Optional[List[str]]): A list of paths to audio files.
        transcription (Optional[List[str]]): A list of transcriptions corresponding to the audio files.
            Must be the same length as `audio`.
        dataset_dir (Optional[str]): A directory containing a `metadata.csv` file and audio files.
        max_workers (Optional[int]): Maximum number of worker threads for parallel audio file processing.
            If None, it defaults to a value based on CPU count.
        test_size (float, optional): Fraction of data to be used as the test set. Defaults to 0.2.
        seed (int, optional): Random seed for reproducible dataset splitting. Defaults to 42.

    Returns:
        DatasetDict: A Hugging Face DatasetDict with 'train' and 'test' splits.

    Raises:
        ValueError: If both `audio/transcription` and `dataset_dir` are not provided,
            or if `audio` and `transcription` are of unequal length,
            or if no valid audio files are found.

    Example:
        >>> dataset = load_local_dataset(
        ...     audio=["/path/to/file1.mp3", "/path/to/file2.wav"],
        ...     transcription=["Hello world", "Foo bar"],
        ...     max_workers=4,
        ...     test_size=0.1
        ... )
        >>> print(dataset)
        DatasetDict({
            train: Dataset({
                features: ['audio', 'sentence'],
                num_rows: 1
            })
            test: Dataset({
                features: ['audio', 'sentence'],
                num_rows: 1
            })
        })
    """
    import tempfile
    import os

    if audio is not None and transcription is not None:
        if len(audio) != len(transcription):
            raise ValueError(
                "The 'audio' and 'transcription' lists must be of equal length."
            )

        # Convert audio files if needed
        converted_audio = process_audio_files_parallel(audio, max_workers)

        if not converted_audio:
            raise ValueError("No audio files could be processed from the provided list.")

        # Create a temporary CSV file with audio paths and transcriptions
        temp_csv = None
        try:
            # Create temporary metadata CSV
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".csv", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(["file_path", "transcription"])  # Header
                for audio_path, text in zip(converted_audio, transcription):
                    writer.writerow([audio_path, text])
                temp_csv = f.name

            logger.info(f"Created temporary metadata file at {temp_csv}")

            # Load using audiofolder format
            dataset = load_dataset(
                "audiofolder",
                data_files={"train": temp_csv},
                split="train",
                audio_column="audio",
                path_column="file_path",
                text_column="transcription",
            )

            # Create train/test split
            splits = dataset.train_test_split(test_size=test_size, seed=seed)
            dataset = DatasetDict({"train": splits["train"], "test": splits["test"]})

            # Rename columns if needed
            if (
                "transcription" in dataset["train"].column_names
                and "sentence" not in dataset["train"].column_names
            ):
                for split in dataset:
                    dataset[split] = dataset[split].rename_column(
                        "transcription", "sentence"
                    )

            # Clean up
            os.unlink(temp_csv)

        except Exception as e:
            # Clean up on error
            if temp_csv and os.path.exists(temp_csv):
                os.unlink(temp_csv)

            logger.error(f"Error with audiofolder loading: {e}")
            logger.info("Falling back to manual dataset creation")

            # Fallback approach
            dataset = Dataset.from_dict(
                {"audio": converted_audio, "sentence": transcription}
            )

            # Force Audio format
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

            # Create train/test split
            splits = dataset.train_test_split(test_size=test_size, seed=seed)
            dataset = DatasetDict({"train": splits["train"], "test": splits["test"]})

        return dataset

    elif dataset_dir is not None:
        if not os.path.isdir(dataset_dir):
            raise ValueError(
                f"Provided dataset_dir '{dataset_dir}' is not a valid directory."
            )

        # Check that metadata.csv exists
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise ValueError(f"'metadata.csv' not found in '{dataset_dir}'.")

        try:
            # First, try using the audiofolder format directly
            logger.info(f"Loading dataset from {dataset_dir} using audiofolder")

            # Load the dataset using audiofolder
            dataset = load_dataset(
                "audiofolder",
                data_dir=dataset_dir,
                split="train",
                audio_column="audio",
                path_column="file_name",
                text_column="transcription",
            )

            # Create train/test split
            splits = dataset.train_test_split(test_size=test_size, seed=seed)
            dataset = DatasetDict({"train": splits["train"], "test": splits["test"]})

            # Rename columns if needed
            if (
                "transcription" in dataset["train"].column_names
                and "sentence" not in dataset["train"].column_names
            ):
                for split in dataset:
                    dataset[split] = dataset[split].rename_column(
                        "transcription", "sentence"
                    )

            # Keep only the columns we need
            for split in dataset:
                if set(dataset[split].column_names) != {"audio", "sentence"}:
                    dataset[split] = dataset[split].select_columns(["audio", "sentence"])

            return dataset

        except Exception as e:
            logger.warning(f"Direct audiofolder loading failed: {e}")
            logger.info("Trying alternative approach...")

            # Try an alternative approach by creating a standardized CSV file
            temp_csv = None
            try:
                # Read the metadata file to identify column names
                with open(metadata_path, "r", newline="", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)

                    # Determine text column
                    text_col = (
                        "transcription"
                        if "transcription" in reader.fieldnames
                        else "sentence"
                    )

                    # Create a standardized temporary CSV
                    with tempfile.NamedTemporaryFile(
                        mode="w", delete=False, suffix=".csv", newline=""
                    ) as f:
                        writer = csv.writer(f)
                        writer.writerow(["file_path", "transcription"])  # Standard header

                        # Reset file pointer
                        csvfile.seek(0)
                        next(csv.reader(csvfile))  # Skip header

                        for row in reader:
                            file_name = row.get("file_name")
                            text = row.get(text_col)

                            if file_name and text:
                                file_path = os.path.join(dataset_dir, file_name)
                                if os.path.isfile(file_path):
                                    writer.writerow([file_path, text])

                        temp_csv = f.name

                # Try loading with audiofolder again using the temporary CSV
                dataset = load_dataset(
                    "audiofolder",
                    data_files={"train": temp_csv},
                    split="train",
                    audio_column="audio",
                    path_column="file_path",
                    text_column="transcription",
                )

                # Create train/test split
                splits = dataset.train_test_split(test_size=test_size, seed=seed)
                dataset = DatasetDict({"train": splits["train"], "test": splits["test"]})

                # Rename columns if needed
                if (
                    "transcription" in dataset["train"].column_names
                    and "sentence" not in dataset["train"].column_names
                ):
                    for split in dataset:
                        dataset[split] = dataset[split].rename_column(
                            "transcription", "sentence"
                        )

                # Clean up
                os.unlink(temp_csv)

                # Keep only the columns we need
                for split in dataset:
                    if set(dataset[split].column_names) != {"audio", "sentence"}:
                        dataset[split] = dataset[split].select_columns(["audio", "sentence"])

                return dataset

            except Exception as e:
                # Clean up on error
                if temp_csv and os.path.exists(temp_csv):
                    os.unlink(temp_csv)

                logger.error(f"Alternative audiofolder loading failed: {e}")
                logger.info("Falling back to traditional approach")

            # Fallback to original method if all else fails
            file_names = []
            sentences = []

            # Parse the metadata.csv file
            with open(metadata_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)

                # Determine text column
                text_col = (
                    "transcription"
                    if "transcription" in reader.fieldnames
                    else "sentence"
                )

                for row in reader:
                    file_name = row.get("file_name")
                    sentence = row.get(text_col)

                    if file_name and sentence:
                        file_path = os.path.join(dataset_dir, file_name)
                        if os.path.isfile(file_path):
                            file_names.append(file_name)
                            sentences.append(sentence)

            # Prepare full file paths
            audio_paths = [
                os.path.join(dataset_dir, file_name) for file_name in file_names
            ]

            if not audio_paths:
                raise ValueError(f"No valid audio files found in '{dataset_dir}'")

            # Convert audio files if needed
            converted_audio = process_audio_files_parallel(audio_paths, max_workers)

            # Create dataset
            dataset = Dataset.from_dict({"audio": converted_audio, "sentence": sentences})

            # Ensure audio is in the correct format
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

            # Create train/test split
            splits = dataset.train_test_split(test_size=test_size, seed=seed)
            dataset_dict = DatasetDict({"train": splits["train"], "test": splits["test"]})

            return dataset_dict

    else:
        raise ValueError(
            "Either 'audio' and 'transcription' lists or a 'dataset_dir' must be provided."
        )
