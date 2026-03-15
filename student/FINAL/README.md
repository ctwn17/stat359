# TinyStories Training on AWS SageMaker

## Directory Structure

```
FINAL/
├── launch_sagemaker.py                             # This will run train_tinystories_model_accelerate.py on SageMaker
├── download_model.py                               # After running the training job, use this script to download the trained model from AWS S3
├── monitor_job.ps1                                 # This script will call AWS Sagemaker every few seconds and will beep when the training job is done or failed
├── README.md
└── source/                                         # Uploaded to SageMaker as training code
    ├── training/
        ├── train_tinystories_model_accelerate.py   # Entry point (Accelerate + SageMaker)
        ├── train_tinystories_model.py              # Original training script (local use)
        ├── train_tinystories_chat_model.py         # Modifies version of the chat training script (local use)
        ├── generate_tinystories_text.py            # Text generation (local use)
        ├── bpe_tokenizer.py                        # BPE tokenizer implementation
        ├── transformer_model.py                    # Model architecture
        └── requirements.txt                        # Python dependencies
    └── evaluation/                 
        ├── analyze_results.py                      # Generates graphs from the evaluation results
        ├── evaluate_stories.py                     # Evaluates a number of metrics on the generated stories
        └── generate_stories.py                     # Uses the trained models to generate stories for later comparison
```

## Prerequisites

```bash
pip install sagemaker boto3 awscli
aws configure   # set your AWS credentials + region
```

You need an IAM role with SageMaker permissions. If you don't have one:
1. Go to IAM console -> Roles -> Create role
2. Select "SageMaker" as the trusted service
3. Attach `AmazonSageMakerFullAccess` policy
4. Note the role ARN

## Training

### 1. Generate the BPE Tokenizer
In order to train the model we need to generate the tokenizer. This will be uploaded to Sagemaker for the online training
```bash
python -m training.train_bpe_tokenizer_hf
```

### 2. Train on Sagemaker
When trying to train the model locally, it was taking forever. Instead, I chose to offload this process to AWS Sagemaker.
```bash
cd FINAL
python launch_sagemaker.py \
    --role arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
    --instance_type ml.g5.xlarge \
    --spot #use this for cheaper instances at the potential cost of longer training time
    --pilot_run #use this to test the model on a small dataset to make sure everything works
```

### 2.5. Monitor training job
To monitor the job until it finishes, we need to get the job name from the output of the previous command.
We can then use the following command to monitor the job:
```ps
monitor_job.ps1 -JobName YOUR-JOB-NAME [-IntervalSeconds 30]
```

### 3. Download trained model
Once the training job is done, you can download the trained model from S3:
```bash
python download_model.py --job_name YOUR-JOB-NAME --output_dir tinystories_model_large
```

### 4a. (Optional) Generate dataset for chat training
We've found that the hugging face chat model is not very good at understanding the prompt-response relationship.
We can generate a dataset of stories that are more suitable for training the chat model.
```bash
python -m training.build_chat_dataset --count 10000 --output chat_dataset.json
```

### 4. Training the chat model
By default, this uses the tinystories-chat model on HuggingFace
```bash
python -m training.train_tinystories_chat_model --pretrained_model_path <path to model>
```
However, you can also load a local dataset using the `--dataset_path` argument.
```bash
python train_tinystories_chat_model.py --pretrained_model_path <path to model> --dataset_path chat_dataset.json
```

### 5. Persona Finetuning
After we have trained the chat model we need to finetune the personas. I had Claude generate the story datasets for the "shy" and "cowboy" personas.
```bash
python -m training.finetune_personas --pretrained_model_path <path to model> \
  --persona_data <path to persona data> \
  --persona_name <persona name>
```

## Evaluation

### 1. Generating Stories
To ensure that we are evaluating the models with as little bias as possible, we want to generate the same prompts for each model-persona combination.
To do this, we can use the following script to generate stories of the following:
- 3 models (baseline, shy, cowboy)
- 2 personas on baseline (shy, cowboy)
- 4 shot types
- 50 story prompts

This will give us 1000 stories
```bash
python -m evaulation.generate_stories --base_model_path <path to model> \
  --shy_model_path <path to shy model> \
  --cowboy_model_path <path to cowboy model> \
  --output_dir <output directory for generated prompts and responses>
```

### 2. Evaluating Stories
Next, we evaluate the generated stories on the following metrics:
- Distinct N (n=1 and n=2)
- Style Strength
- Persona Consistency
- Perplexity

```bash
python -m evaluation.evaluate_stories --input <the json file generated by evaluation.generate_stories> \
  --base_model_path <path to model for perplexity evaluation>
```

### 3. Analyzing Results
Finally, we generate a number of graphs for our data with the following script:
```bash
python -m evaluation.analyze_results --input <the json file generated by evaluation.evaluate_stories> \
  --output_dir <output directory for graphs>
```