"""
launch_sagemaker.py — Submit TinyStories training job to AWS SageMaker.

Prerequisites:
    pip install sagemaker boto3

    You need:
    1. An AWS account with SageMaker access
    2. An IAM role with SageMaker execution permissions
    3. Your tokenizer .pkl file uploaded to S3 (or included in source dir)

Usage:
    python launch_sagemaker.py

    # Or override defaults:
    python launch_sagemaker.py \
        --role arn:aws:iam::123456789:role/SageMakerRole \
        --instance_type ml.g5.12xlarge \
        --instance_count 1 \
        --tokenizer_s3 s3://my-bucket/tinystories/bpe_tokenizer_tinystories.pkl

Made with the help of Claude Opus
"""

import argparse
import sagemaker
from sagemaker.pytorch import PyTorch


def parse_args():
    parser = argparse.ArgumentParser(description="Launch TinyStories training on SageMaker")

    # AWS / SageMaker config
    parser.add_argument("--role", type=str, default=None,
                        help="SageMaker execution IAM role ARN. If None, uses sagemaker.get_execution_role()")
    parser.add_argument("--instance_type", type=str, default="ml.g5.12xlarge",
                        help="SageMaker instance type")
    parser.add_argument("--instance_count", type=int, default=1,
                        help="Number of instances (nodes)")
    parser.add_argument("--volume_size", type=int, default=100,
                        help="EBS volume size in GB")
    parser.add_argument("--max_runtime", type=int, default=48 * 3600,
                        help="Max training time in seconds (default: 48h)")

    # S3 paths
    parser.add_argument("--tokenizer_s3", type=str, default=None,
                        help="S3 URI to tokenizer .pkl file (e.g. s3://bucket/path/tokenizer.pkl). "
                             "If None, tokenizer must be in source/")
    parser.add_argument("--output_s3", type=str, default=None,
                        help="S3 URI for model output. If None, uses SageMaker default bucket.")

    # Training hyperparameters (passed to train.py)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)

    # Flags
    parser.add_argument("--pilot_run", action="store_true",
                        help="Quick test with small dataset subset")
    parser.add_argument("--spot", action="store_true",
                        help="Use spot instances (up to 90%% cheaper, but can be interrupted)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Get SageMaker session
    session = sagemaker.Session()

    # Get role
    if args.role:
        role = args.role
    else:
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            print("ERROR: Could not determine SageMaker execution role.")
            print("Either run this from a SageMaker notebook, or pass --role explicitly.")
            print("  python launch_sagemaker.py --role arn:aws:iam::ACCOUNT:role/ROLE_NAME")
            return

    # Output S3 path
    output_path = args.output_s3 or f"s3://{session.default_bucket()}/tinystories/output"

    # Hyperparameters passed to train.py as command-line args
    hyperparameters = {
        "dataset": "roneneldan/TinyStories",
        "tokenizer_path": "bpe_tokenizer_tinystories.pkl",
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "intermediate_size": args.intermediate_size,
        "max_seq_len": args.max_seq_len,
        "window_size": args.max_seq_len,  # match attention window to sequence length
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "amp": "",  # flag-style argument — SageMaker passes --amp
    }
    if args.pilot_run:
        hyperparameters["pilot_run"] = ""

    # Input channels
    inputs = {}
    if args.tokenizer_s3:
        inputs["tokenizer"] = args.tokenizer_s3

    print(f"{'='*60}")
    print(f"SageMaker Training Job Configuration")
    print(f"{'='*60}")
    print(f"Role:           {role[:60]}...")
    print(f"Instance:       {args.instance_count}x {args.instance_type}")
    print(f"Spot:           {args.spot}")
    print(f"Output:         {output_path}")
    print(f"Model:          {args.num_layers}L / {args.hidden_size}H / {args.num_heads}A")
    print(f"Batch/GPU:      {args.batch_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"{'='*60}")

    # Create PyTorch Estimator
    estimator = PyTorch(
        entry_point="train_tinystories_model_accelerate.py",
        source_dir="source/training",
        role=role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        volume_size=args.volume_size,
        max_run=args.max_runtime,
        output_path=output_path,
        framework_version="2.5.1",
        py_version="py311",
        hyperparameters=hyperparameters,

        # Distributed training — SageMaker sets up torchrun automatically
        distribution={
            "torch_distributed": {
                "enabled": True
            }
        },

        # Spot instance config (optional, saves $$)
        use_spot_instances=args.spot,
        max_wait=args.max_runtime * 2 if args.spot else None,

        # Tags for cost tracking
        tags=[
            {"Key": "Project", "Value": "tinystories"},
            {"Key": "Course", "Value": "stat359"},
        ],
    )

    # Submit the job
    print("\nSubmitting training job...")
    estimator.fit(inputs=inputs if inputs else None, wait=False)

    print(f"\nJob submitted: {estimator.latest_training_job.name}")
    print(f"\nMonitor at:")
    print(f"  https://console.aws.amazon.com/sagemaker/home#/training-jobs")
    print(f"\nOr in terminal:")
    print(f"  aws sagemaker describe-training-job --training-job-name {estimator.latest_training_job.name}")
    print(f"\nTo stream logs:")
    print(f"  estimator.logs()  # from Python")
    print(f"\nModel artifacts will be at:")
    print(f"  {output_path}/{estimator.latest_training_job.name}/output/model.tar.gz")


if __name__ == "__main__":
    main()