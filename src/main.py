# main.py
import argparse
import sys
from train import train

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate Whisper-large-v3 model with LoRA on custom Bangla dataset"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # Sub-parser for training
    train_parser = subparsers.add_parser("train", help="Fine-tune Whisper model with LoRA on custom Bangla dataset")
    
    # Custom dataset paths
    train_parser.add_argument("--audio_dir", type=str, default='/content/Train/', 
                            help="Path to audio directory containing region subfolders")
    train_parser.add_argument("--annotation_dir", type=str, default='/content/Train_annotation/', 
                            help="Path to annotation directory containing CSV files")
    
    # Training parameters
    train_parser.add_argument("--num_train_epochs", type=int, default=10,
                            help="Number of training epochs")
    train_parser.add_argument("--train_batch_size", type=int, default=4,
                            help="Batch size for training")
    train_parser.add_argument("--learning_rate", type=float, default=5e-5,
                            help="Learning rate for optimizer")
    train_parser.add_argument("--output_dir", type=str, default="./whisper-bangla-LoRA", 
                            help="Where to save the LoRA weights and processor")
    
    # Data processing parameters
    train_parser.add_argument("--num_workers", type=int, default=2, 
                            help="Number of workers for data loading")
    train_parser.add_argument("--max_input_length", type=float, default=8.0, 
                            help="Max length of input audio (in seconds) - increase if you have enough memory")
    train_parser.add_argument("--val_split", type=float, default=0.1,
                            help="Fraction of data to use for validation (default: 0.1 = 10%)")
    
    # Early stopping parameters
    train_parser.add_argument("--early_stopping_patience", type=int, default=3, 
                            help="How many epochs to wait for improvement before stopping early")
    train_parser.add_argument("--early_stopping_min_delta", type=float, default=0.0, 
                            help="Minimum improvement needed to reset early stopping patience")
    
    # Debug and other parameters
    train_parser.add_argument("--debug", action='store_true', 
                            help="Run in debug mode (use a small subset of the data)")
    train_parser.add_argument("--debug_subset_size", type=int, default=24, 
                            help="Number of samples to use in debug mode")
    train_parser.add_argument("--seed", type=int, default=42, 
                            help="Random seed for reproducibility")

    # Sub-parser for evaluation
    eval_parser = subparsers.add_parser("eval", help="Evaluate LoRA fine-tuned Whisper model")
    
    # Custom dataset paths for evaluation
    eval_parser.add_argument("--audio_dir", type=str, default='/content/Train/', 
                           help="Path to audio directory containing region subfolders")
    eval_parser.add_argument("--annotation_dir", type=str, default='/content/Train_annotation/', 
                           help="Path to annotation directory containing CSV files")
    
    # Evaluation parameters
    eval_parser.add_argument("--batch_size", type=int, default=4,
                           help="Batch size for evaluation")
    eval_parser.add_argument("--model_dir", type=str, default="./whisper-bangla-LoRA", 
                           help="Directory containing the saved model")
    eval_parser.add_argument("--max_input_length", type=float, default=8.0, 
                           help="Max length of input audio (in seconds)")
    eval_parser.add_argument("--num_workers", type=int, default=4, 
                           help="Number of workers for data loading")

    args = parser.parse_args() 

    if args.command == "train":
        train(args)
    elif args.command == "eval":
        print("\n" + "="*70)
        print("EVALUATION NOT YET IMPLEMENTED FOR CUSTOM DATASET")
        print("="*70)
        print("\nThe evaluation script (eval.py) is still designed for Hugging Face datasets.")
        print("For now, you can test your trained model using:")
        print("\n  python inference.py \\")
        print("      --model_dir ./whisper-bangla-LoRA \\")
        print("      --audio_file /path/to/test_audio.wav")
        print("\n" + "="*70)
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()