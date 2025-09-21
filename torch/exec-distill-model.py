import argparse
import os
import sys
import torch

# Ensure local imports resolve when executed directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from train import (
    load_and_preprocess_data,
    prepare_optimized_data,
    distill_model_optimized,
    OptimizedTrainingConfig,
    check_gpu_memory,
    save_training_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run knowledge distillation for OptimizedModel (PyTorch)")
    parser.add_argument("--teacher_path", type=str, required=True, help="Path to teacher model checkpoint (.pth)")
    parser.add_argument("--student_hidden_size", type=int, default=128, help="Student hidden size")
    parser.add_argument("--student_num_layers", type=int, default=2, help="Student number of layers")
    parser.add_argument("--temperature", type=float, default=3.0, help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for distillation loss vs ground truth")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for student training")
    parser.add_argument("--model_name", type=str, default="distilled_student", help="Base name for saved artifacts")
    parser.add_argument("--data_path", type=str, default="training_data.h5", help="HDF5 file with preprocessed data")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Starting Optimized PyTorch Distillation")
    print("=" * 60)

    if not os.path.exists(args.teacher_path):
        raise FileNotFoundError(f"Teacher model not found: {args.teacher_path}")

    # Load data
    X, Y, T = load_and_preprocess_data(args.data_path)
    input_dim = X.shape[1]
    output_dim = Y.shape[1]

    # Training configuration
    config = OptimizedTrainingConfig()
    config.update(batch_size=args.batch_size)
    config.validate()

    # GPU memory check
    if not check_gpu_memory(required_gb=2.0):
        print("Warning: Insufficient GPU memory detected. Reducing batch size.")
        config.batch_size = min(config.batch_size, 32)

    # Prepare data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dict = prepare_optimized_data(
        X, Y, T,
        batch_size=config.batch_size,
        val_split=config.val_split,
        test_split=config.test_split,
        device=device,
    )
    data_dict['T'] = T

    # Run distillation
    student, train_losses, val_losses, best_val = distill_model_optimized(
        data_dict=data_dict,
        input_dim=input_dim,
        output_dim=output_dim,
        teacher_model_path=args.teacher_path,
        student_hidden_size=args.student_hidden_size,
        student_num_layers=args.student_num_layers,
        temperature=args.temperature,
        alpha=args.alpha,
        model_name=args.model_name,
        base_config=config,
    )

    # Save config and summary
    save_training_config(config, args.model_name)
    print("\nDistillation complete!")
    if best_val is not None:
        print(f"Best validation loss: {best_val:.6f}")
    print("Artifacts saved:")
    print(f"  - {args.model_name}_best_model.pth")
    print(f"  - {args.model_name}_swa_model.pth")
    print(f"  - {args.model_name}_ema_model.pth")
    print(f"  - {args.model_name}_config.json")


if __name__ == "__main__":
    main()


