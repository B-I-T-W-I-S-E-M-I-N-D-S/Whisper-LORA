# # Import necessary modules and libraries
# import os
# import sys
# import argparse
# import logging
# import warnings
# from pathlib import Path

# import torch
# from torch.utils.data import DataLoader
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from peft import LoraConfig, get_peft_model
# import evaluate
# from tqdm.auto import tqdm

# from accelerate import Accelerator
# from accelerate.utils import set_seed

# from utils import count_parameters, compute_module_sizes
# from data_collate import DataCollatorSpeechSeq2SeqWithPadding
# from custom_dataset import CustomBanglaDataset, create_train_val_split, collate_fn_with_filter

# # Setting up logging to show information level messages
# logging.basicConfig(format="%(message)s", level=logging.INFO)

# def initialize_accelerator():
#     """
#     Initialize the Accelerator with the desired settings.

#     Returns:
#         Accelerator: Configured Accelerator instance.
#     """
#     accelerator = Accelerator(
#         mixed_precision='fp16',  # Enable mixed precision for faster training
#         device_placement=True,
#         log_with="all"  # Adjust based on your logging preference
#     )
#     return accelerator


# def set_environment(accelerator, seed):
#     """
#     Set random seed and adjust backend settings for optimal performance.

#     Args:
#         accelerator (Accelerator): Accelerator instance.
#         seed (int): Seed for reproducibility.
#     """
#     set_seed(seed)
#     if accelerator.device.type == "cuda":
#         torch.backends.cudnn.benchmark = True
#     logging.info(f"Training on: {accelerator.device}, using mixed precision: {accelerator.mixed_precision}")


# def load_and_prepare_datasets(args, processor):
#     """
#     Load and preprocess training and validation datasets from custom folder structure.

#     Args:
#         args (argparse.Namespace): Parsed arguments.
#         processor (WhisperProcessor): Whisper processor for handling text normalization.

#     Returns:
#         tuple: (train_dataset, val_dataset)
#     """
#     logging.info("Loading custom Bangla dataset...")

#     # Create the full dataset
#     full_dataset = CustomBanglaDataset(
#         audio_base_dir=args.audio_dir,
#         annotation_base_dir=args.annotation_dir,
#         processor=processor,
#         max_input_length=args.max_input_length
#     )

#     # Use a small subset if in debug mode
#     if args.debug:
#         logging.info(f"Debug mode: Using only {args.debug_subset_size} samples.")
#         subset_indices = list(range(min(args.debug_subset_size, len(full_dataset))))
#         full_dataset.data = [full_dataset.data[i] for i in subset_indices]
    
#     # Split into train and validation
#     train_dataset, val_dataset = create_train_val_split(
#         full_dataset, 
#         val_split=args.val_split,
#         seed=args.seed
#     )
    
#     logging.info(f"Training samples: {len(train_dataset)}")
#     logging.info(f"Validation samples: {len(val_dataset)}")

#     return train_dataset, val_dataset


# def setup_model(processor):
#     """
#     Load the Whisper model and apply LoRA adapters to fine-tune it.

#     Args:
#         processor (WhisperProcessor): The processor used for handling the model inputs.

#     Returns:
#         PeftModel: The Whisper model with LoRA adapters applied.
#     """
#     logging.info("Loading Whisper-large-v3 model and applying LoRA...")

#     whisper_model = WhisperForConditionalGeneration.from_pretrained(
#         "distil-whisper/distil-large-v3",
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
#     )

#     # Set up LoRA configuration
#     lora_config = LoraConfig(
#         inference_mode=False,
#         r=16,  # Dimensionality of LoRA
#         lora_alpha=32,  # Scaling factor for LoRA
#         lora_dropout=0.05,  # Dropout to prevent overfitting
#         target_modules=['q_proj', 'v_proj'],  # The parts of the model we're modifying
#         bias="none"  # No bias term for simplicity
#     )

#     # Add LoRA to the Whisper model
#     model = get_peft_model(whisper_model, lora_config)
    
#     # Check and log the model size and parameter counts
#     count_parameters(model)
#     module_sizes = compute_module_sizes(model)
#     logging.info(f"\nModel size: {module_sizes[''] * 1e-9:.2f} GB\n")

#     return model


# def custom_collate_fn(batch, data_collator):
#     """
#     Custom collate function that handles None values and uses the data collator.
    
#     Args:
#         batch: List of samples from the dataset
#         data_collator: DataCollatorSpeechSeq2SeqWithPadding instance
    
#     Returns:
#         Collated batch or None if all samples are invalid
#     """
#     # Filter out None values (failed/skipped samples)
#     batch = [item for item in batch if item is not None]
    
#     if len(batch) == 0:
#         return None
    
#     # Use the data collator to properly pad and format the batch
#     return data_collator(batch)


# def prepare_dataloaders(args, train_dataset, val_dataset, data_collator):
#     """
#     Create DataLoaders for the training and validation datasets.

#     Args:
#         args (argparse.Namespace): Parsed arguments containing data settings.
#         train_dataset: The training dataset.
#         val_dataset: The validation dataset.
#         data_collator (DataCollatorSpeechSeq2SeqWithPadding): Handles padding for batches.

#     Returns:
#         tuple: (train_dataloader, validation_dataloader)
#     """
#     logging.info("Creating DataLoaders...")

#     # Create a lambda function to wrap the collate_fn with the data_collator
#     collate_fn = lambda batch: custom_collate_fn(batch, data_collator)

#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=args.train_batch_size,
#         collate_fn=collate_fn,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         shuffle=True  # Randomize order of training samples
#     )

#     validation_dataloader = DataLoader(
#         val_dataset,
#         batch_size=args.train_batch_size,
#         collate_fn=collate_fn,
#         num_workers=args.num_workers,
#         pin_memory=True,
#         shuffle=False  # No need to shuffle validation data
#     )

#     return train_dataloader, validation_dataloader


# def setup_optimizer_scheduler(model, learning_rate):
#     """
#     Set up the optimizer and learning rate scheduler for training.

#     Args:
#         model (PeftModel): The model to be trained.
#         learning_rate (float): The learning rate for the optimizer.

#     Returns:
#         tuple: (optimizer, scheduler)
#     """
#     logging.info("Setting up optimizer and learning rate scheduler...")

#     # Train only the LoRA adapters, so we grab the parameters that require gradients
#     trainable_params = [p for p in model.parameters() if p.requires_grad]

#     # AdamW optimizer for weight decay and better generalization
#     optimizer = torch.optim.AdamW(params=trainable_params, lr=learning_rate)

#     # Set up a learning rate scheduler that reduces the LR when validation loss plateaus
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='min',  # Minimize the validation loss
#         factor=0.5,  # Halve the learning rate when triggered
#         patience=2,  # How many epochs to wait before reducing
#         min_lr=1e-6  # Set a floor for the learning rate
#     )

#     return optimizer, scheduler


# def train_epoch(model, dataloader, optimizer, accelerator):
#     """
#     Run one epoch of training.

#     Args:
#         model (PeftModel): The model to train.
#         dataloader (DataLoader): The training DataLoader.
#         optimizer (torch.optim.Optimizer): Optimizer.
#         accelerator (Accelerator): Accelerator to handle distributed training.

#     Returns:
#         float: The average loss for this training epoch.
#     """
#     model.train()
#     total_loss = 0.0
#     num_batches = 0

#     # Progress bar to track training
#     progress_bar = tqdm(dataloader, desc="Training", disable=not accelerator.is_local_main_process, leave=False)
    
#     for batch in progress_bar:
#         # Skip None batches (can happen if all samples in batch were filtered)
#         if batch is None:
#             continue
            
#         outputs = model(**batch, use_cache=False)
#         loss = outputs.loss
#         loss = loss / accelerator.num_processes  # Account for distributed training

#         # Backpropagation
#         accelerator.backward(loss)

#         # Clip gradients to avoid exploding gradients
#         accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

#         # Update model weights
#         optimizer.step()
#         optimizer.zero_grad()

#         total_loss += loss.item()
#         num_batches += 1
#         progress_bar.set_postfix({'loss': loss.item()})
    
#     avg_loss = total_loss / max(num_batches, 1)
#     return avg_loss


# def validate(model, dataloader, processor, wer_metric, accelerator):
#     """
#     Validate the model on the validation set and calculate WER.

#     Args:
#         model (PeftModel): The trained model.
#         dataloader (DataLoader): Validation DataLoader.
#         processor (WhisperProcessor): Processor to decode predictions.
#         wer_metric (evaluate.Metric): Metric to calculate Word Error Rate (WER).
#         accelerator (Accelerator): Accelerator for distributed validation.

#     Returns:
#         tuple: (average validation loss, average WER)
#     """
#     model.eval()
#     total_eval_loss = 0.0
#     total_wer = 0.0
#     num_batches = 0

#     progress_bar = tqdm(dataloader, desc="Validating", disable=not accelerator.is_local_main_process, leave=False)
    
#     for batch in progress_bar:
#         # Skip None batches
#         if batch is None:
#             continue
            
#         with torch.no_grad():
#             outputs = model(**batch, use_cache=False)
#             loss = outputs.loss
#             loss = loss / accelerator.num_processes  # Normalize loss for distributed training
#             total_eval_loss += loss.item()

#             # Decode predictions and calculate WER
#             logits = outputs.logits
#             predictions = torch.argmax(logits, dim=-1)
#             decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)

#             labels = batch["labels"]
#             labels = torch.where(labels != -100, labels, processor.tokenizer.pad_token_id)
#             decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

#             wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
#             total_wer += wer
#             num_batches += 1
    
#     avg_eval_loss = total_eval_loss / max(num_batches, 1)
#     avg_wer = total_wer / max(num_batches, 1)
#     return avg_eval_loss, avg_wer


# def save_model(model, processor, output_dir, accelerator):
#     """
#     Save the trained model and processor.

#     Args:
#         model (PeftModel): The fine-tuned model.
#         processor (WhisperProcessor): The processor used with the model.
#         output_dir (str): Directory where the model and processor should be saved.
#         accelerator (Accelerator): Accelerator instance to ensure saving happens only once.
#     """
#     if accelerator.is_main_process:
#         os.makedirs(output_dir, exist_ok=True)
#         model.save_pretrained(output_dir)  # Save only the LoRA adapters
#         processor.save_pretrained(output_dir)
#         logging.info(f"Model and processor saved to {output_dir}")

# def train(args):
#     """
#     Main function to handle the entire training process, including model setup, training, validation, 
#     and saving the model. 
#     """

#     # Initialize Accelerator for handling hardware optimizations
#     accelerator = initialize_accelerator()

#     # Set environment, including seed and backend settings
#     set_environment(accelerator, args.seed)

#     # Load the Whisper processor for handling data
#     logging.info("Loading Whisper processor...")
#     processor = WhisperProcessor.from_pretrained(
#         "distil-whisper/distil-large-v3", 
#         language="bengali",  # Changed to Bengali
#         task="transcribe"
#     )

#     # Load and preprocess datasets
#     train_dataset, val_dataset = load_and_prepare_datasets(args, processor)

#     # Set up the data collator for padding and batching
#     data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

#     # Load Whisper model with LoRA adapters applied
#     model = setup_model(processor)

#     # Prepare DataLoaders for training and validation
#     train_dataloader, validation_dataloader = prepare_dataloaders(
#         args,
#         train_dataset=train_dataset,
#         val_dataset=val_dataset,
#         data_collator=data_collator
#     )

#     # Set up optimizer and learning rate scheduler
#     optimizer, scheduler = setup_optimizer_scheduler(model, args.learning_rate)

#     # Prepare all components for training with Accelerator (handles parallelization and optimization)
#     model, optimizer, train_dataloader, validation_dataloader, scheduler = accelerator.prepare(
#         model, optimizer, train_dataloader, validation_dataloader, scheduler
#     )

#     # Make sure the output directory exists
#     if accelerator.is_main_process:
#         os.makedirs(args.output_dir, exist_ok=True)

#     # Set up the evaluation metric for Word Error Rate (WER)
#     wer_metric = evaluate.load("wer")

#     # Initialize variables for early stopping
#     best_wer = float('inf')  # Set the best WER to a large value initially
#     epochs_no_improve = 0  # Track how many epochs since the last improvement

#     logging.info("Starting the training process...")

#     # Loop through each epoch
#     for epoch in tqdm(range(args.num_train_epochs), desc="Epochs", disable=not accelerator.is_local_main_process, leave=False):
#         logging.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")

#         # Train for one epoch
#         avg_train_loss = train_epoch(model, train_dataloader, optimizer, accelerator)
#         logging.info(f"Average training loss: {avg_train_loss:.4f}")

#         # Validate the model to check performance
#         avg_eval_loss, avg_wer = validate(model, validation_dataloader, processor, wer_metric, accelerator)
#         logging.info(f"Validation loss: {avg_eval_loss:.4f}, WER: {avg_wer:.4f}")

#         # Adjust the learning rate based on validation loss
#         scheduler.step(avg_eval_loss)

#         # Check if this is the best model so far
#         if avg_wer < best_wer - args.early_stopping_min_delta:
#             best_wer = avg_wer
#             epochs_no_improve = 0
#             # Save the model if it improved
#             save_model(model, processor, args.output_dir, accelerator)
#             logging.info(f"New best WER: {best_wer:.4f}. Model saved.")
#         else:
#             epochs_no_improve += 1
#             logging.info(f"No improvement in WER for {epochs_no_improve} epoch(s).")
#             # If no improvement after a few epochs, stop training early
#             if epochs_no_improve >= args.early_stopping_patience:
#                 logging.info("Early stopping triggered. No significant improvement.")
#                 break

#     logging.info("Training process completed.")
#     if accelerator.is_main_process:
#         logging.info(f"Model and LoRA adapters have been saved to {args.output_dir}. You can now evaluate the performance.")










# Import necessary modules and libraries
import os
import sys
import argparse
import logging
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model
import evaluate
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from utils import count_parameters, compute_module_sizes
from data_collate import DataCollatorSpeechSeq2SeqWithPadding
from custom_dataset import CustomBanglaDataset, create_train_val_split, collate_fn_with_filter

# Setting up logging to show information level messages
logging.basicConfig(format="%(message)s", level=logging.INFO)

def initialize_accelerator():
    """
    Initialize the Accelerator with the desired settings.

    Returns:
        Accelerator: Configured Accelerator instance.
    """
    accelerator = Accelerator(
        mixed_precision='fp16',  # Enable mixed precision for faster training
        device_placement=True,
        log_with="all"  # Adjust based on your logging preference
    )
    return accelerator


def set_environment(accelerator, seed):
    """
    Set random seed and adjust backend settings for optimal performance.

    Args:
        accelerator (Accelerator): Accelerator instance.
        seed (int): Seed for reproducibility.
    """
    set_seed(seed)
    if accelerator.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logging.info(f"Training on: {accelerator.device}, using mixed precision: {accelerator.mixed_precision}")


def load_and_prepare_datasets(args, processor):
    """
    Load and preprocess training and validation datasets from custom folder structure.

    Args:
        args (argparse.Namespace): Parsed arguments.
        processor (WhisperProcessor): Whisper processor for handling text normalization.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    logging.info("Loading custom Bangla dataset...")

    # Create the full dataset
    full_dataset = CustomBanglaDataset(
        audio_base_dir=args.audio_dir,
        annotation_base_dir=args.annotation_dir,
        processor=processor,
        max_input_length=args.max_input_length
    )

    # Use a small subset if in debug mode
    if args.debug:
        logging.info(f"Debug mode: Using only {args.debug_subset_size} samples.")
        subset_indices = list(range(min(args.debug_subset_size, len(full_dataset))))
        full_dataset.data = [full_dataset.data[i] for i in subset_indices]
    
    # Split into train and validation
    train_dataset, val_dataset = create_train_val_split(
        full_dataset, 
        val_split=args.val_split,
        seed=args.seed
    )
    
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def setup_model(processor):
    """
    Load the Whisper model and apply LoRA adapters to fine-tune it.

    Args:
        processor (WhisperProcessor): The processor used for handling the model inputs.

    Returns:
        PeftModel: The Whisper model with LoRA adapters applied.
    """
    logging.info("Loading Whisper-large-v3 model and applying LoRA...")

    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Set up LoRA configuration
    lora_config = LoraConfig(
        inference_mode=False,
        r=16,  # Dimensionality of LoRA
        lora_alpha=32,  # Scaling factor for LoRA
        lora_dropout=0.05,  # Dropout to prevent overfitting
        target_modules=['q_proj', 'v_proj'],  # The parts of the model we're modifying
        bias="none"  # No bias term for simplicity
    )

    # Add LoRA to the Whisper model
    model = get_peft_model(whisper_model, lora_config)
    
    # Check and log the model size and parameter counts
    count_parameters(model)
    module_sizes = compute_module_sizes(model)
    logging.info(f"\nModel size: {module_sizes[''] * 1e-9:.2f} GB\n")

    return model


def custom_collate_fn(batch, data_collator):
    """
    Custom collate function that handles None values and uses the data collator.
    
    Args:
        batch: List of samples from the dataset
        data_collator: DataCollatorSpeechSeq2SeqWithPadding instance
    
    Returns:
        Collated batch or None if all samples are invalid
    """
    # Filter out None values (failed/skipped samples)
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Use the data collator to properly pad and format the batch
    return data_collator(batch)


def prepare_dataloaders(args, train_dataset, val_dataset, data_collator):
    """
    Create DataLoaders for the training and validation datasets.

    Args:
        args (argparse.Namespace): Parsed arguments containing data settings.
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        data_collator (DataCollatorSpeechSeq2SeqWithPadding): Handles padding for batches.

    Returns:
        tuple: (train_dataloader, validation_dataloader)
    """
    logging.info("Creating DataLoaders...")

    # Create a lambda function to wrap the collate_fn with the data_collator
    collate_fn = lambda batch: custom_collate_fn(batch, data_collator)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True  # Randomize order of training samples
    )

    validation_dataloader = DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False  # No need to shuffle validation data
    )

    return train_dataloader, validation_dataloader


def setup_optimizer_scheduler(model, learning_rate):
    """
    Set up the optimizer and learning rate scheduler for training.

    Args:
        model (PeftModel): The model to be trained.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        tuple: (optimizer, scheduler)
    """
    logging.info("Setting up optimizer and learning rate scheduler...")

    # Train only the LoRA adapters, so we grab the parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # AdamW optimizer for weight decay and better generalization
    optimizer = torch.optim.AdamW(params=trainable_params, lr=learning_rate)

    # Set up a learning rate scheduler that reduces the LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Minimize the validation loss
        factor=0.5,  # Halve the learning rate when triggered
        patience=2,  # How many epochs to wait before reducing
        min_lr=1e-6  # Set a floor for the learning rate
    )

    return optimizer, scheduler


def train_epoch(model, dataloader, optimizer, accelerator):
    """
    Run one epoch of training.

    Args:
        model (PeftModel): The model to train.
        dataloader (DataLoader): The training DataLoader.
        optimizer (torch.optim.Optimizer): Optimizer.
        accelerator (Accelerator): Accelerator to handle distributed training.

    Returns:
        float: The average loss for this training epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Progress bar to track training
    progress_bar = tqdm(dataloader, desc="Training", disable=not accelerator.is_local_main_process, leave=False)
    
    for batch in progress_bar:
        # Skip None batches (can happen if all samples in batch were filtered)
        if batch is None:
            continue
            
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        loss = loss / accelerator.num_processes  # Account for distributed training

        # Backpropagation
        accelerator.backward(loss)

        # Clip gradients to avoid exploding gradients
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model weights
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(model, dataloader, processor, wer_metric, accelerator):
    """
    Validate the model on the validation set and calculate WER.

    Args:
        model (PeftModel): The trained model.
        dataloader (DataLoader): Validation DataLoader.
        processor (WhisperProcessor): Processor to decode predictions.
        wer_metric (evaluate.Metric): Metric to calculate Word Error Rate (WER).
        accelerator (Accelerator): Accelerator for distributed validation.

    Returns:
        tuple: (average validation loss, average WER)
    """
    model.eval()
    total_eval_loss = 0.0
    total_wer = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Validating", disable=not accelerator.is_local_main_process, leave=False)
    
    for batch in progress_bar:
        # Skip None batches
        if batch is None:
            continue
            
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            loss = loss / accelerator.num_processes  # Normalize loss for distributed training
            total_eval_loss += loss.item()

            # Decode predictions and calculate WER
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)

            labels = batch["labels"]
            labels = torch.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

            wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
            total_wer += wer
            num_batches += 1
    
    avg_eval_loss = total_eval_loss / max(num_batches, 1)
    avg_wer = total_wer / max(num_batches, 1)
    return avg_eval_loss, avg_wer


def save_model(model, processor, output_dir, accelerator):
    """
    Save the trained model and processor.

    Args:
        model (PeftModel): The fine-tuned model.
        processor (WhisperProcessor): The processor used with the model.
        output_dir (str): Directory where the model and processor should be saved.
        accelerator (Accelerator): Accelerator instance to ensure saving happens only once.
    """
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)  # Save only the LoRA adapters
        processor.save_pretrained(output_dir)
        logging.info(f"Model and processor saved to {output_dir}")

def train(args):
    """
    Main function to handle the entire training process, including model setup, training, validation, 
    and saving the model. 
    """

    # Initialize Accelerator for handling hardware optimizations
    accelerator = initialize_accelerator()

    # Set environment, including seed and backend settings
    set_environment(accelerator, args.seed)

    # Load the Whisper processor for handling data
    logging.info("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v3", 
        language="bengali",  # Changed to Bengali
        task="transcribe"
    )

    # Load and preprocess datasets
    train_dataset, val_dataset = load_and_prepare_datasets(args, processor)

    # Set up the data collator for padding and batching
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Load Whisper model with LoRA adapters applied
    model = setup_model(processor)

    # Prepare DataLoaders for training and validation
    train_dataloader, validation_dataloader = prepare_dataloaders(
        args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        data_collator=data_collator
    )

    # Set up optimizer and learning rate scheduler
    optimizer, scheduler = setup_optimizer_scheduler(model, args.learning_rate)

    # Prepare all components for training with Accelerator (handles parallelization and optimization)
    model, optimizer, train_dataloader, validation_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, scheduler
    )

    # Make sure the output directory exists
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Set up the evaluation metric for Word Error Rate (WER)
    wer_metric = evaluate.load("wer")

    # Initialize variables for early stopping
    best_wer = float('inf')  # Set the best WER to a large value initially
    epochs_no_improve = 0  # Track how many epochs since the last improvement

    logging.info("Starting the training process...")

    # Loop through each epoch
    for epoch in tqdm(range(args.num_train_epochs), desc="Epochs", disable=not accelerator.is_local_main_process, leave=False):
        logging.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")

        # Train for one epoch
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, accelerator)
        logging.info(f"Average training loss: {avg_train_loss:.4f}")

        # Validate the model to check performance
        avg_eval_loss, avg_wer = validate(model, validation_dataloader, processor, wer_metric, accelerator)
        logging.info(f"Validation loss: {avg_eval_loss:.4f}, WER: {avg_wer:.4f}")

        # Adjust the learning rate based on validation loss
        scheduler.step(avg_eval_loss)

        # Check if this is the best model so far
        if avg_wer < best_wer - args.early_stopping_min_delta:
            best_wer = avg_wer
            epochs_no_improve = 0
            # Save the model if it improved
            save_model(model, processor, args.output_dir, accelerator)
            logging.info(f"New best WER: {best_wer:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in WER for {epochs_no_improve} epoch(s).")
            # If no improvement after a few epochs, stop training early
            if epochs_no_improve >= args.early_stopping_patience:
                logging.info("Early stopping triggered. No significant improvement.")
                break

    logging.info("Training process completed.")
    if accelerator.is_main_process:
        logging.info(f"Model and LoRA adapters have been saved to {args.output_dir}. You can now evaluate the performance.")