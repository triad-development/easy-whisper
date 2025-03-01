
import os
import torch
import numpy as np
import gc
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
from datasets import DatasetDict, Audio
from tqdm import tqdm
from torch.utils.data import DataLoader
import evaluate
from easy_whisper import logger as main_logger
from easy_whisper.hf_login import ensure_hf_login
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftConfig
)

logger = main_logger.getChild("trainer")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for speech-to-text training that handles padding.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels-- they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # ff bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def prepare_dataset(batch, processor):
    """
    Prepare dataset by computing features and tokenizing sentences.
    
    Args:
        batch: A batch of data from the dataset
        processor: The Whisper processor
    
    Returns:
        Processed batch with input_features and labels
    """
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

def compute_metrics(pred, tokenizer):
    """
    Compute Word Error Rate (WER) metric.
    
    Args:
        pred: Model predictions
        tokenizer: The tokenizer to decode predictions
        
    Returns:
        Dictionary with WER score
    """
    metric = evaluate.load("wer")
    
    # get the predictions - handle different formats
    if isinstance(pred.predictions, tuple):
        # sometimes seq2seq models return a tuple
        pred_ids = pred.predictions[0] 
    else:
        pred_ids = pred.predictions
    
    # get label IDs
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Handle nested lists in predictions
    if len(pred_ids.shape) > 2:
        # take the first element if it's nested 
        pred_ids = pred_ids[:, 0, :]
    
    # decode 
    try:
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    except TypeError as e:
        # fallback to manual decoding if batch_decode fails
        logger.warning(f"Error in batch_decode: {e}. Falling back to individual decoding.")
        
        pred_str = []
        for ids in pred_ids:
            if isinstance(ids, list):
                pred_str.append(tokenizer.decode(ids, skip_special_tokens=True))
            else:
                pred_str.append(tokenizer.decode(ids.tolist(), skip_special_tokens=True))
                
        label_str = []
        for ids in label_ids:
            if isinstance(ids, list):
                label_str.append(tokenizer.decode(ids, skip_special_tokens=True))
            else:
                label_str.append(tokenizer.decode(ids.tolist(), skip_special_tokens=True))

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
def train_model(
    dataset: DatasetDict,
    model_name_or_path: str = "openai/whisper-tiny",
    output_dir: str = "./whisper-finetune",
    language: str = "english",
    task: str = "transcribe",
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-3,
    warmup_steps: int = 50,
    num_train_epochs: int = 3,
    fp16: bool = True,
    load_in_8bit: bool = True,
    use_auth_token: bool = False,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    hub_private_repo: bool = False,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    num_workers: int = 1,  # Default to 1 to avoid multiprocessing issues
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_total_limit: int = 3,
    seed: int = 42,
    force_hf_login: bool = False,
    safe_multiprocessing: bool = True  #  to control safe multiprocessing
) -> WhisperForConditionalGeneration:
    """
    Fine-tune a Whisper model with PEFT/LoRA.
    
    Args:
        dataset: DatasetDict with 'train' and 'test' splits
        model_name_or_path: Base model name or path (default: "openai/whisper-tiny")
        output_dir: Directory to save the model (default: "./whisper-finetune")
        language: Language for transcription (default: "english")
        task: Task type, e.g., "transcribe" (default: "transcribe")
        lora_r: LoRA attention dimension (default: 32)
        lora_alpha: LoRA alpha parameter (default: 64)
        lora_dropout: LoRA dropout rate (default: 0.05)
        batch_size: Training batch size (default: 8)
        gradient_accumulation_steps: Gradient accumulation steps (default: 1)
        learning_rate: Learning rate (default: 1e-3)
        warmup_steps: Learning rate warmup steps (default: 50)
        num_train_epochs: Number of training epochs (default: 3)
        fp16: Use mixed precision training (default: True)
        load_in_8bit: Load model in 8-bit precision (default: True)
        use_auth_token: Use auth token for accessing private models (default: False)
        push_to_hub: Push model to Hub (default: False)
        hub_model_id: Model ID on Hub (default: None)
        hub_private_repo: Make Hub repo private (default: False)
        max_train_samples: Maximum number of training samples (default: None)
        max_eval_samples: Maximum number of evaluation samples (default: None)
        num_workers: Number of workers for data loading (default: 1)
        evaluation_strategy: Evaluation strategy (default: "epoch")
        save_strategy: Save strategy (default: "epoch")
        save_total_limit: Maximum number of checkpoints to keep (default: 3)
        seed: Random seed (default: 42)
        force_hf_login: Force a new Hugging Face login even if token exists (default: False)
        safe_multiprocessing: Use safe approach for multiprocessing (default: True)
    
    Returns:
        Trained model
    """
    logger.info(f"Starting fine-tuning process for {model_name_or_path}")
    
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # set the random seed for reproducibility
    torch.manual_seed(seed)
    
    # handle Hugging Face Hub login if needed
    if push_to_hub:
        logger.info("Push to Hub is enabled, checking Hugging Face login...")
        if not ensure_hf_login(force_login=force_hf_login):
            logger.warning("Failed to log in to Hugging Face Hub. Disabling push_to_hub.")
            push_to_hub = False
    
    # 1. load processor, tokenizer, and feature extractor
    logger.info("Loading processor, tokenizer, and feature extractor...")
    try:
        processor = WhisperProcessor.from_pretrained(
            model_name_or_path, 
            language=language, 
            task=task,
            token=use_auth_token
        )
        tokenizer = processor.tokenizer
    except Exception as e:
        logger.error(f"Error loading processor: {e}")
        raise
    
    # 2. Prepare 
    logger.info("Preparing dataset...")
    
    # check if required columns exist in the dataset
    required_columns = ["audio", "sentence"]
    for split in dataset:
        for col in required_columns:
            if col not in dataset[split].column_names:
                raise ValueError(f"Column '{col}' not found in dataset['{split}']. Required columns: {required_columns}")
        
        # ensure audio column is in the Audio format
        if not isinstance(dataset[split].features["audio"], Audio):
            raise ValueError(f"Column 'audio' in dataset['{split}'] must be in Audio format")
    
    # limit dataset size if specified
    if max_train_samples is not None and "train" in dataset:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
    
    if max_eval_samples is not None and "test" in dataset:
        dataset["test"] = dataset["test"].select(range(min(max_eval_samples, len(dataset["test"]))))
    
    # process the dataset
    def process_func(batch):
        return prepare_dataset(batch, processor)
    
    column_names = {split: dataset[split].column_names for split in dataset}
    
    logger.info("Mapping dataset to compute input features and labels...")
    processed_dataset = DatasetDict()
    
    # determine best num_proc value to avoid multiprocessing issues
    import platform
    
    effective_num_workers = num_workers
    
    # automatically handle Windows multiprocessing issues
    if safe_multiprocessing and platform.system() == "Windows":
        # Handle Windows multiprocessing issues by using a single process
        effective_num_workers = 1
        logger.info("Running on Windows - Setting num_proc=1 to avoid multiprocessing issues")
    
    # limit num_proc based on dataset size and available CPU cores
    for split in dataset:
        logger.info(f"Processing {split} split...")
        
        # Limit num_proc based on dataset size
        split_size = len(dataset[split])
        max_effective_workers = min(effective_num_workers, split_size)
        
        if max_effective_workers < effective_num_workers:
            logger.info(f"Reducing num_proc to {max_effective_workers} for dataset of size {split_size}")
        
        try:
            # attempt  to process the dataset with multiple processes
            processed_dataset[split] = dataset[split].map(
                process_func,
                remove_columns=column_names[split],
                num_proc=max_effective_workers,
                desc=f"Processing {split} dataset"
            )
        except RuntimeError as e:
            if "An attempt has been made to start a new process" in str(e):
                logger.warning("Multiprocessing error detected. Falling back to single process.")
                # Fall back to single process if multiprocessing fails
                processed_dataset[split] = dataset[split].map(
                    process_func,
                    remove_columns=column_names[split],
                    num_proc=1,  # Use a single process
                    desc=f"Processing {split} dataset (single process)"
                )
            else:
                # reraise if it's not a multiprocessing error
                raise
    
    # 3. create data collator
    logger.info("Creating data collator...")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # 4. laod and prepare model
    logger.info(f"Loading model: {model_name_or_path}")
    try:
        device_map = "auto" if torch.cuda.is_available() else None
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name_or_path,
            load_in_8bit=load_in_8bit and torch.cuda.is_available(),
            device_map=device_map,
            token=use_auth_token
        )
        
        # set forced_decoder_ids to None to remove forced language tokens
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        
        # Prepare model for training with LoRA
        logger.info("Preparing model for PEFT training...")
        model = prepare_model_for_kbit_training(model)
        
        # set up LoRA configuration
        logger.info("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none"
        )
        
        # get PEFT model
        model = get_peft_model(model, lora_config)
        
        # print trainable parameters info
        model.print_trainable_parameters()
        
    except Exception as e:
        logger.error(f"Error loading or preparing model: {e}")
        raise
    
    # 5. Setup training arguments
    logger.info("Setting up training arguments...")
    
    # create a unique name for the model if pushing to hub
    if push_to_hub and hub_model_id is None:
        # generate a name based on the base model and configuration
        base_model_name = model_name_or_path.split("/")[-1]
        hub_model_id = f"{base_model_name}-finetuned-{language}"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        fp16=fp16 and torch.cuda.is_available(),
        per_device_eval_batch_size=batch_size,
        generation_max_length=128,
        logging_steps=25,
        remove_unused_columns=False,  
        label_names=["labels"],
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_private_repo=hub_private_repo,
        seed=seed
    )
    
    # 6. setup trainer
    logger.info("Setting up trainer...")
    
    # define compute_metrics function with tokenizer
    def compute_metrics_wrapper(pred):
        return compute_metrics(pred, tokenizer)
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset.get("test", None),
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper if "test" in processed_dataset else None,
        processing_class=processor.feature_extractor,
    )
    
    # 7. train model
    logger.info("Starting training...")
    model.config.use_cache = False  # Disable cache for training
    
    trainer.train()
    
    # 8. save model
    logger.info("Saving model...")
    trainer.save_model(output_dir)
    
    # 9. push to hub if requested
    if push_to_hub:
        logger.info(f"Pushing model to Hub: {hub_model_id}")
        trainer.push_to_hub()
    
    logger.info("Training completed successfully!")
    
    return model

def evaluate_model(
    model,
    dataset,
    processor,
    batch_size=8,
    language="english",
    task="transcribe",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Evaluate a fine-tuned Whisper model.
    
    Args:
        model: The Whisper model
        dataset: The evaluation dataset
        processor: The Whisper processor
        batch_size: Batch size for evaluation
        language: Language for transcription
        task: Task type, e.g., "transcribe"
        device: Device to use for evaluation
        
    Returns:
        Dictionary with WER score
    """
    logger.info(f"Evaluating model on {len(dataset)} samples")
    
    # craete data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # create dataloader
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    
    # load metric
    metric = evaluate.load("wer")
    
    # Get tokenizer
    tokenizer = processor.tokenizer
    
    # set decoder prompt ids based on language and task
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    
    # eval loop
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        with torch.cuda.amp.autocast("cuda"):
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to(device),
                        decoder_input_ids=None,  # Let the model generate from scratch
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        del generated_tokens, labels, batch
        gc.collect()
        
    # Compute WER
    wer = 100 * metric.compute()
    logger.info(f"WER: {wer:.2f}%")
    
    return {"wer": wer}

def load_finetuned_model(
    model_path_or_id,
    device="cuda" if torch.cuda.is_available() else "cpu",
    load_in_8bit=True,
    language="english",
    task="transcribe"
):
    """
    Load a fine-tuned Whisper model and processor.
    
    Args:
        model_path_or_id: Path to model directory or HuggingFace model ID
        device: Device to load the model on
        load_in_8bit: Whether to load in 8-bit precision
        language: Language for transcription
        task: Task type
        
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading fine-tuned model from: {model_path_or_id}")
    
    # Check if it's a PEFT 
    is_peft_model = os.path.exists(os.path.join(model_path_or_id, "adapter_config.json"))
    
    if is_peft_model:
        # Load PEFT 
        logger.info("Loading as PEFT model")
        peft_config = PeftConfig.from_pretrained(model_path_or_id)
        base_model_path = peft_config.base_model_name_or_path
        
        # Load base model
        device_map = "auto" if device == "cuda" else None
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_path,
            load_in_8bit=load_in_8bit and device == "cuda",
            device_map=device_map,
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, model_path_or_id)
        
        # load processor from base model
        processor = WhisperProcessor.from_pretrained(base_model_path, language=language, task=task)
    else:
        # load regular model and processor
        logger.info("Loading as regular model")
        model = WhisperForConditionalGeneration.from_pretrained(model_path_or_id)
        processor = WhisperProcessor.from_pretrained(model_path_or_id, language=language, task=task)
    
    return model, processor

def transcribe_audio(
    audio_path,
    model,
    processor,
    language="english",
    task="transcribe",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Transcribe a single audio file using a fine-tuned Whisper model.
    
    Args:
        audio_path: Path to the audio file
        model: The Whisper model
        processor: The Whisper processor
        language: Language for transcription
        task: Task type
        device: Device to use for inference
        
    Returns:
        Transcribed text
    """
    from transformers import AutomaticSpeechRecognitionPipeline
    
    # create pipeline
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    
    # get decoder prompt ids
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    
    # create pipeline
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device
    )
    
    # transcribe 
    with torch.cuda.amp.autocast():
        result = pipe(
            audio_path,
            generate_kwargs={"forced_decoder_ids": forced_decoder_ids},
            max_new_tokens=255
        )
    
    return result["text"]

def setup_gradio_demo(
    model,
    processor,
    language="english",
    task="transcribe",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Set up a Gradio demo for the fine-tuned Whisper model.
    
    Args:
        model: The Whisper model
        processor: The Whisper processor
        language: Language for transcription
        task: Task type
        device: Device to use for inference
        
    Returns:
        Gradio interface
    """
    import gradio as gr
    from transformers import AutomaticSpeechRecognitionPipeline
    
    # create pipeline
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    
    # get decoder prompt ids
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    
    # create pipeline
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device
    )
    
    #  transcription function
    def transcribe(audio):
        with torch.cuda.amp.autocast():
            text = pipe(
                audio,
                generate_kwargs={"forced_decoder_ids": forced_decoder_ids},
                max_new_tokens=255
            )["text"]
        return text
    
    # create Gradio interface
    model_name = model.config._name_or_path.split("/")[-1]
    title = f"PEFT LoRA + {'INT8' if getattr(model, 'is_loaded_in_8bit', False) else 'FP16'} {model_name} {language.capitalize()}"
    
    iface = gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(source="microphone", type="filepath"),
        outputs="text",
        title=title,
        description=f"Realtime demo for {language} speech recognition using `PEFT-LoRA` fine-tuned Whisper model.",
    )
    
    return iface