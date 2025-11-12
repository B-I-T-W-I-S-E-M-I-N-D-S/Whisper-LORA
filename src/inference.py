"""
Enhanced batch inference script for fine-tuned Whisper model with Bangla support
Processes entire folder and exports results to CSV
"""
import argparse
import os
import sys
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from pathlib import Path
import pandas as pd
from datetime import datetime
import glob

# Set UTF-8 encoding for proper Bangla text display
if sys.platform == "win32":
    # For Windows console
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def load_model(model_dir):
    """Load the fine-tuned model and processor"""
    print(f"Loading model from {model_dir}...")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load processor
    processor = WhisperProcessor.from_pretrained(model_dir)
    
    # Load base model
    base_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_dir)
    
    # Merge LoRA weights for faster inference (optional)
    model = model.merge_and_unload()
    
    # Set to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"✓ Model loaded successfully on {device}")
    return model, processor, device

def preprocess_audio(audio_path, target_sr=16000):
    """Load and preprocess audio file"""
    # Check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to target sample rate if necessary
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Convert to numpy
    audio_array = waveform.squeeze().numpy()
    
    return audio_array, target_sr

def transcribe_audio(audio_path, model, processor, device, language="bn", task="transcribe"):
    """Transcribe a single audio file"""
    try:
        # Preprocess audio
        audio_array, sampling_rate = preprocess_audio(audio_path)
        
        # Get audio duration
        duration = len(audio_array) / sampling_rate
        
        # Process audio
        inputs = processor(
            audio=audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        # Move to device
        input_features = inputs.input_features.to(device)
        if device == "cuda":
            input_features = input_features.half()
        
        # Set language and task tokens for generation
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, 
            task=task
        )
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
                num_beams=5,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode
        transcription = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        # Clean up transcription
        transcription = transcription.strip()
        
        return transcription, duration
        
    except Exception as e:
        print(f"❌ Error transcribing {audio_path}: {e}")
        return None, 0

def get_audio_files(folder_path):
    """Get all audio files from folder"""
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg', '*.opus', '*.wma', '*.aac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
        audio_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    return sorted(audio_files)

def process_folder(folder_path, model, processor, device, language="bn", task="transcribe"):
    """Process all audio files in a folder"""
    # Get all audio files
    audio_files = get_audio_files(folder_path)
    
    if not audio_files:
        print(f"❌ No audio files found in {folder_path}")
        return None
    
    print(f"\n📁 Found {len(audio_files)} audio files")
    print("=" * 70)
    
    # Store results
    results = []
    
    # Process each file
    for idx, audio_path in enumerate(audio_files, 1):
        filename = os.path.basename(audio_path)
        print(f"\n[{idx}/{len(audio_files)}] 🎤 Processing: {filename}")
        
        transcription, duration = transcribe_audio(
            audio_path, 
            model, 
            processor, 
            device,
            language=language,
            task=task
        )
        
        if transcription:
            print(f"✓ Duration: {duration:.2f}s")
            print(f"✓ Transcription: {transcription[:100]}{'...' if len(transcription) > 100 else ''}")
            
            results.append({
                'audio_file': filename,
                'transcription': transcription
            })
        else:
            print(f"✗ Failed to transcribe")
            results.append({
                'audio_file': filename,
                'transcription': '[ERROR: Transcription failed]'
            })
    
    return results

def save_to_csv(results, output_path):
    """Save transcription results to CSV with proper UTF-8 encoding for Bangla"""
    try:
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV with UTF-8 encoding and BOM for Excel compatibility
        df.to_csv(
            output_path, 
            index=False, 
            encoding='utf-8-sig',  # UTF-8 with BOM for Excel
            quoting=1  # Quote all fields to preserve line breaks
        )
        
        print(f"\n💾 Transcriptions saved to: {output_path}")
        print(f"📊 Total files processed: {len(results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files using fine-tuned Whisper model for Bangla",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --folder_path ./audio_files
  python inference.py --folder_path ./audio_files --output transcriptions.csv
  python inference.py --folder_path ./audio_files --model_dir ./my-model --language bn
        """
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./whisper-bangla-LoRA",
        help="Directory containing the fine-tuned model (default: ./whisper-bangla-LoRA)"
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to folder containing audio files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: transcriptions_TIMESTAMP.csv)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="bn",
        help="Language code (default: bn for Bangla)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task: transcribe or translate (default: transcribe)"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("🎯 Bangla Whisper Batch Transcription System")
    print("=" * 70)
    
    # Check if folder exists
    if not os.path.exists(args.folder_path):
        print(f"❌ Folder not found: {args.folder_path}")
        sys.exit(1)
    
    try:
        # Load model
        model, processor, device = load_model(args.model_dir)
        
        # Process folder
        results = process_folder(
            args.folder_path,
            model,
            processor,
            device,
            language=args.language,
            task=args.task
        )
        
        if results:
            # Generate output filename if not specified
            if args.output:
                output_path = args.output
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"transcriptions_{timestamp}.csv"
            
            # Save to CSV
            if save_to_csv(results, output_path):
                print("\n" + "=" * 70)
                print("✅ Batch transcription completed successfully!")
                print("=" * 70)
            else:
                print("\n❌ Failed to save results to CSV")
                sys.exit(1)
        else:
            print("\n❌ No transcriptions generated")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
