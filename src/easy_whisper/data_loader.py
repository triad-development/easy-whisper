import os
import csv
import logging
from typing import List, Optional
from datasets import Dataset, Audio
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_audio_to_wav(audio_path: str) -> str:
    try:
        if not audio_path.lower().endswith('.wav'):
            logger.info(f"Converting '{audio_path}' to WAV format.")
            audio = AudioSegment.from_file(audio_path)
            wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_path, format='wav')
            return wav_path
        return audio_path
    except Exception as e:
        logger.error(f"Failed to convert '{audio_path}' to WAV: {e}")
        raise

def load_local_dataset(
    audio: Optional[List[str]] = None, 
    transcription: Optional[List[str]] = None, 
    dataset_dir: Optional[str] = None
) -> Dataset:
    if audio is not None and transcription is not None:
        if len(audio) != len(transcription):
            raise ValueError("The 'audio' and 'transcription' lists must be of equal length.")
        
        # Process each provided audio file: convert if needed.
        converted_audio = []
        for a in audio:
            try:
                converted_path = convert_audio_to_wav(a)
                converted_audio.append(converted_path)
            except Exception as e:
                logger.warning(f"Skipping file '{a}' due to conversion error: {e}")
        
        if not converted_audio:
            raise ValueError("No audio files could be processed from the provided list.")
        
        dataset = Dataset.from_dict({"audio": converted_audio, "sentence": transcription})
        dataset = dataset.cast_column("audio", Audio())
        return dataset
    
    elif dataset_dir is not None:
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"Provided dataset_dir '{dataset_dir}' is not a valid directory.")
        
        # Check that metadata.csv exists in the dataset directory.
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise ValueError(f"'metadata.csv' not found in '{dataset_dir}'.")
        
        audio_files = []
        sentences = []
        
        # Parse the metadata.csv file.
        with open(metadata_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Validate that required columns are present.
            if 'file_name' not in reader.fieldnames:
                raise ValueError("metadata.csv must contain a 'file_name' column.")
            if not (('transcription' in reader.fieldnames) or ('sentence' in reader.fieldnames)):
                raise ValueError("metadata.csv must contain a 'transcription' or 'sentence' column.")
            
            for row in reader:
                file_name = row.get('file_name')
                sentence = row.get('transcription') or row.get('sentence')
                if not file_name or not sentence:
                    logger.warning("Skipping a row with missing 'file_name' or transcription.")
                    continue  # Skip rows missing essential information.
                file_path = os.path.join(dataset_dir, file_name)
                if os.path.isfile(file_path):
                    try:
                        converted_path = convert_audio_to_wav(file_path)
                        audio_files.append(converted_path)
                        sentences.append(sentence)
                    except Exception as e:
                        logger.warning(f"Skipping file '{file_path}' due to conversion error: {e}")
                else:
                    logger.warning(f"Audio file '{file_path}' does not exist. Skipping.")
        
        if not audio_files:
            raise ValueError(f"No valid audio files found in '{dataset_dir}'. At least one audio file is required.")
        
        dataset = Dataset.from_dict({"audio": audio_files, "sentence": sentences})
        dataset = dataset.cast_column("audio", Audio())
        if 'sentence' not in dataset or 'audio' not in dataset:
            raise ValueError("Either audio or sentence column found in dataset")
        dataset_columns = dataset.column_names
        dataset_columns.remove('sentence')
        dataset_columns.remove('audio')
        dataset.remove_columns(dataset_columns)
        return dataset
    else:
        raise ValueError("Either 'audio' and 'transcription' lists or a 'dataset_dir' must be provided.")
