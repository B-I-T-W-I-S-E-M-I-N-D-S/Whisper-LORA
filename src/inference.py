"""
Enhanced inference script for fine-tuned Whisper model with Bangla support
"""
import argparse
import os
import sys
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from pathlib import Path

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
    print(f"\n🎤 Transcribing: {audio_path}")
    
    try:
        # Preprocess audio
        audio_array, sampling_rate = preprocess_audio(audio_path)
        
        # Get audio duration
        duration = len(audio_array) / sampling_rate
        print(f"📊 Audio duration: {duration:.2f} seconds")
        
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
        print("⏳ Generating transcription...")
        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,  # Increased for longer transcriptions
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
        
        return transcription
        
    except Exception as e:
        print(f"❌ Error transcribing {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_transcription(transcription, output_file):
    """Save transcription to file with UTF-8 encoding"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(transcription)
        print(f"💾 Transcription saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving transcription: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using fine-tuned Whisper model for Bangla",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --audio_file audio.mp3
  python inference.py --audio_file audio.wav --output transcription.txt
  python inference.py --audio_file audio.mp3 --model_dir ./my-model --language bn
        """
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./whisper-bangla-LoRA",
        help="Directory containing the fine-tuned model (default: ./whisper-bangla-LoRA)"
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to audio file to transcribe (supports .wav, .mp3, .flac, etc.)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path to save transcription (optional)"
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
    print("🎯 Bangla Whisper Transcription System")
    print("=" * 70)
    
    try:
        # Load model
        model, processor, device = load_model(args.model_dir)
        
        # Transcribe audio
        transcription = transcribe_audio(
            args.audio_file, 
            model, 
            processor, 
            device,
            language=args.language,
            task=args.task
        )
        
        if transcription:
            # Display transcription
            print(f"\n{'=' * 70}")
            print(f"📝 Transcription:")
            print(f"{'=' * 70}")
            print(transcription)
            print(f"{'=' * 70}\n")
            
            # Save to file if output path is specified
            if args.output:
                save_transcription(transcription, args.output)
            else:
                # Auto-save with same name as audio file
                audio_path = Path(args.audio_file)
                output_path = audio_path.with_suffix('.txt')
                save_transcription(transcription, output_path)
        else:
            print("❌ Failed to transcribe audio")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("✅ Transcription completed successfully!")

if __name__ == "__main__":
    main()