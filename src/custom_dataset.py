import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import logging

class CustomBanglaDataset(Dataset):
    """
    Custom dataset for loading Bangla audio files and their transcriptions.
    """
    def __init__(self, audio_base_dir, annotation_base_dir, processor, max_input_length=8.0, split='train'):
        """
        Args:
            audio_base_dir (str): Base directory containing region subfolders with audio files
            annotation_base_dir (str): Base directory containing CSV annotation files
            processor: WhisperProcessor for feature extraction
            max_input_length (float): Maximum audio length in seconds
            split (str): 'train' or 'validation' (for future train/val split)
        """
        self.audio_base_dir = audio_base_dir
        self.annotation_base_dir = annotation_base_dir
        self.processor = processor
        self.max_input_length = max_input_length
        self.sampling_rate = processor.feature_extractor.sampling_rate
        
        # List of regions
        self.regions = [
            'Barisal', 'Bhola', 'Bogura', 'Brahmanbaria', 'Chittagong', 
            'Comilla', 'Dhaka', 'Feni', 'Jessore', 'Jhenaidah', 
            'Khulna', 'Kushtia', 'Lakshmipur', 'Mymensingh', 'Natore', 
            'Noakhali', 'Pabna', 'Rajshahi', 'Rangpur', 'Sylhet'
        ]
        
        self.data = []
        self._load_dataset()
        
        logging.info(f"Loaded {len(self.data)} samples from custom dataset")
    
    def _load_dataset(self):
        """Load all audio files and their transcriptions from CSV files."""
        for region in self.regions:
            # Path to the CSV file for this region
            csv_path = os.path.join(self.annotation_base_dir, f"{region}.csv")
            
            # Path to the audio folder for this region
            audio_folder = os.path.join(self.audio_base_dir, region)
            
            # Check if both CSV and folder exist
            if not os.path.exists(csv_path):
                logging.warning(f"CSV file not found: {csv_path}")
                continue
            
            if not os.path.exists(audio_folder):
                logging.warning(f"Audio folder not found: {audio_folder}")
                continue
            
            # Read the CSV file
            try:
                df = pd.read_csv(csv_path)
                
                # Assume CSV has columns: filename (or similar) and transcription (or similar)
                # Adjust column names based on your actual CSV structure
                if df.shape[1] >= 2:
                    filename_col = df.columns[0]
                    transcription_col = df.columns[1]
                else:
                    logging.warning(f"CSV file {csv_path} doesn't have at least 2 columns")
                    continue
                
                # Iterate through each row in the CSV
                for _, row in df.iterrows():
                    filename = row[filename_col]
                    transcription = row[transcription_col]
                    
                    # Skip if transcription is empty or NaN
                    if pd.isna(transcription) or str(transcription).strip() == "":
                        continue
                    
                    # Construct full audio file path
                    audio_path = os.path.join(audio_folder, filename)
                    
                    # Check if audio file exists
                    if os.path.exists(audio_path):
                        self.data.append({
                            'audio_path': audio_path,
                            'transcription': str(transcription).strip(),
                            'region': region
                        })
                    else:
                        logging.warning(f"Audio file not found: {audio_path}")
                
                logging.info(f"Loaded {len([d for d in self.data if d['region'] == region])} samples from {region}")
                
            except Exception as e:
                logging.error(f"Error loading CSV {csv_path}: {e}")
                continue
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            dict: Dictionary containing 'input_features', 'labels', and 'input_length'
        """
        item = self.data[idx]
        audio_path = item['audio_path']
        transcription = item['transcription']
        
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
                waveform = resampler(waveform)
            
            # Convert to numpy array
            audio_array = waveform.squeeze().numpy()
            
            # Calculate audio length in seconds
            audio_length = len(audio_array) / self.sampling_rate
            
            # Skip if audio is too long
            if audio_length > self.max_input_length:
                # Return None to indicate this sample should be skipped
                return None
            
            # Process audio and text
            processed = self.processor(
                audio=audio_array,
                sampling_rate=self.sampling_rate,
                text=transcription,
            )
            
            # Add input length for potential filtering
            processed['input_length'] = audio_length
            
            return processed
            
        except Exception as e:
            logging.error(f"Error processing audio file {audio_path}: {e}")
            return None


def collate_fn_with_filter(batch):
    """
    Custom collate function that filters out None values (failed samples).
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    return batch


def create_train_val_split(dataset, val_split=0.1, seed=42):
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: The full dataset
        val_split (float): Fraction of data to use for validation
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    from torch.utils.data import random_split
    
    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Perform random split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    return train_dataset, val_dataset