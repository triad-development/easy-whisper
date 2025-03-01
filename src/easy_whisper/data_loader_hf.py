from datasets import load_dataset, DatasetDict, Audio
from typing import Optional, List, Dict, Union
from easy_whisper import logger as main_logger

logger = main_logger.getChild("data_loader_hf")


def load_hf_dataset(
	dataset_name: str,
	audio_column: str = "audio",
	text_column: str = None,
	config_name: Optional[str] = None,
	split: Optional[Union[str, List[str]]] = None,
	token: Optional[str] = None,
	sampling_rate: int = 16000,
	test_size: float = 0.2,
	seed: int = 42,
	process_audio: bool = True,
	column_mapping: Optional[Dict[str, str]] = None,
	filter_fn: Optional[callable] = None,
	cache_dir: Optional[str] = None,
) -> DatasetDict:
	"""
	Load a dataset from Hugging Face's datasets hub and prepare it for speech recognition.

	Args:
	    dataset_name: Name of the dataset on Hugging Face (e.g., 'mozilla-foundation/common_voice_11_0')
	    audio_column: Name of the column containing audio data (default: 'audio')
	    text_column: Name of the column containing transcription text. If None, tries to find common
	                column names: ['text', 'sentence', 'transcription', 'transcript']
	    config_name: Optional dataset configuration name (e.g., 'en' for Common Voice)
	    split: Dataset split(s) to load. If None, loads all available splits
	    token: Hugging Face token for accessing private datasets or datasets requiring authentication
	    sampling_rate: Target sampling rate for audio (default: 16000)
	    test_size: Fraction of data to use for test set if dataset doesn't have splits (default: 0.2)
	    seed: Random seed for split generation (default: 42)
	    process_audio: Whether to process audio files right away (can be memory intensive for large datasets)
	    column_mapping: Dictionary mapping dataset columns to required columns ('audio' and 'sentence')
	    filter_fn: Optional function to filter the dataset (takes a row and returns True/False)
	    cache_dir: Optional directory to cache the downloaded dataset

	Returns:
	    DatasetDict with 'train' and 'test' splits, each containing 'audio' and 'sentence' columns
	"""
	logger.info(f"Loading dataset '{dataset_name}' from Hugging Face Hub")

	try:
		# Load the dataset
		dataset = load_dataset(
			dataset_name,
			name=config_name,
			split=split,
			token=token,
			cache_dir=cache_dir,
		)

		# Convert to DatasetDict if it's a single split
		if not isinstance(dataset, DatasetDict):
			if split and isinstance(split, str):
				# User specified a single split, wrap it in DatasetDict with that split name
				dataset = DatasetDict({split: dataset})
			else:
				# Create train/test split
				logger.info(f"Creating train/test split with test_size={test_size}")
				train_test = dataset.train_test_split(test_size=test_size, seed=seed)
				dataset = DatasetDict(
					{"train": train_test["train"], "test": train_test["test"]}
				)

		# Check and map columns to the expected format
		for split_name, split_dataset in dataset.items():
			# Identify the text column if not provided
			if text_column is None:
				common_text_columns = [
					"text",
					"sentence",
					"transcription",
					"transcript",
				]
				for col in common_text_columns:
					if col in split_dataset.column_names:
						text_column = col
						logger.info(f"Using '{text_column}' as text column")
						break
				if text_column is None:
					raise ValueError(
						f"Could not automatically determine text column. "
						f"Available columns: {split_dataset.column_names}. "
						f"Please specify 'text_column' parameter."
					)

			# Apply column mapping if provided
			if column_mapping:
				split_dataset = split_dataset.rename_columns(
					{
						old: new
						for old, new in column_mapping.items()
						if old in split_dataset.column_names
					}
				)

			# Standardize column names
			rename_dict = {}
			if audio_column != "audio" and audio_column in split_dataset.column_names:
				rename_dict[audio_column] = "audio"
			if text_column != "sentence" and text_column in split_dataset.column_names:
				rename_dict[text_column] = "sentence"

			if rename_dict:
				dataset[split_name] = split_dataset.rename_columns(rename_dict)

		# Filter dataset if filter function is provided
		if filter_fn:
			logger.info("Applying filter function to dataset")
			for split_name in dataset:
				dataset[split_name] = dataset[split_name].filter(filter_fn)

		# Process audio if needed
		if process_audio:
			for split_name in dataset:
				# Check if audio is already in the right format
				if "audio" in dataset[split_name].column_names:
					audio_feature = dataset[split_name].features["audio"]

					# If it's a path or something else, cast it to Audio
					if not isinstance(audio_feature, Audio):
						logger.info(
							f"Converting '{split_name}' audio column to Audio feature"
						)
						dataset[split_name] = dataset[split_name].cast_column(
							"audio", Audio(sampling_rate=sampling_rate)
						)
					# If it's already Audio but with different sampling rate
					elif audio_feature.sampling_rate != sampling_rate:
						logger.info(
							f"Resampling '{split_name}' audio from {audio_feature.sampling_rate}Hz to {sampling_rate}Hz"
						)
						dataset[split_name] = dataset[split_name].cast_column(
							"audio", Audio(sampling_rate=sampling_rate)
						)

		# Ensure required columns exist
		for split_name, split_dataset in dataset.items():
			if "audio" not in split_dataset.column_names:
				raise ValueError(
					f"Split '{split_name}' does not contain an 'audio' column after processing"
				)
			if "sentence" not in split_dataset.column_names:
				raise ValueError(
					f"Split '{split_name}' does not contain a 'sentence' column after processing"
				)

		# Keep only the necessary columns
		for split_name in dataset:
			if set(dataset[split_name].column_names) != {"audio", "sentence"}:
				dataset[split_name] = dataset[split_name].select_columns(
					["audio", "sentence"]
				)

		logger.info(f"Successfully loaded dataset with splits: {list(dataset.keys())}")
		for split_name, split_dataset in dataset.items():
			logger.info(f"  - {split_name}: {len(split_dataset)} examples")

		return dataset

	except Exception as e:
		logger.error(f"Error loading dataset '{dataset_name}': {e}")
		raise
