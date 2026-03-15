"""
download_model.py — Download trained model from SageMaker S3 output.

Usage:
    python download_model.py --job_name tinystories-2026-03-07-12-00-00-000
    python download_model.py --s3_uri s3://bucket/tinystories/output/job-name/output/model.tar.gz

Made with the help of Claude Opus
"""

import argparse
import os
import tarfile
import boto3
from urllib.parse import urlparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name", type=str, default=None,
                        help="SageMaker training job name (looks up S3 path automatically)")
    parser.add_argument("--s3_uri", type=str, default=None,
                        help="Direct S3 URI to model.tar.gz")
    parser.add_argument("--output_dir", type=str, default="tinystories_model_sagemaker",
                        help="Local directory to extract model into")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.job_name:
        # Look up S3 output path from the training job
        sm = boto3.client("sagemaker")
        job = sm.describe_training_job(TrainingJobName=args.job_name)
        status = job["TrainingJobStatus"]
        print(f"Job status: {status}")

        if status != "Completed":
            print(f"Job is not complete yet ({status}). Wait for it to finish.")
            if status == "InProgress":
                print(f"  Secondary status: {job.get('SecondaryStatus', 'Unknown')}")
            return

        s3_uri = job["ModelArtifacts"]["S3ModelArtifacts"]
        print(f"Model artifacts: {s3_uri}")

    elif args.s3_uri:
        s3_uri = args.s3_uri
    else:
        print("Provide either --job_name or --s3_uri")
        return

    # Download from S3
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    import tempfile
    local_tar = os.path.join(tempfile.gettempdir(), "model.tar.gz")
    print(f"Downloading {s3_uri}...")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_tar)

    # Extract
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Extracting to {args.output_dir}/...")
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=args.output_dir)

    os.remove(local_tar)

    # List what we got
    print(f"\nDownloaded files:")
    for f in os.listdir(args.output_dir):
        size = os.path.getsize(os.path.join(args.output_dir, f))
        print(f"  {f:40s}  {size / 1e6:.1f} MB")

    print(f"\nReady! Use with:")
    print(f"  python generate_tinystories_text.py --model_path {args.output_dir}/best_model.pth")
    print(f"  python chat_with_tinystories_model.py --model_path {args.output_dir}/final_model.pth")


if __name__ == "__main__":
    main()