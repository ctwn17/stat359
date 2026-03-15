import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from datasets import load_dataset
from training.bpe_tokenizer import BPETokenizer
from training.transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
from tqdm import tqdm
import time
import json

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed


def evaluate(model, val_dataloader, criterion, accelerator):
    """Evaluate model. Works on 1 GPU or many — Accelerate handles it."""
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            outputs = model(input_ids=inputs)
            logits = outputs["logits"]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            gathered_loss = accelerator.gather(loss.unsqueeze(0)).mean()
            total_loss += gathered_loss.item()
            num_batches += 1
    return total_loss / max(num_batches, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyStories on SageMaker with Accelerate")

    # SageMaker passes these via environment variables or hyperparameters
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer_tinystories.pkl")
    parser.add_argument("--max_seq_len", type=int, default=256)

    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--window_size", type=int, default=256)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--pilot_run", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--amp", action="store_true")

    # SageMaker-specific paths (set automatically by SageMaker)
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "tinystories_model"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))

    # SageMaker input channels — if you upload tokenizer to S3
    parser.add_argument("--tokenizer_channel", type=str, default=os.environ.get("SM_CHANNEL_TOKENIZER", None))

    return parser.parse_args()


def load_tokenizer(args):
    """Load tokenizer, checking SageMaker input channel first, then local path."""
    if args.tokenizer_channel and os.path.isdir(args.tokenizer_channel):
        # Tokenizer uploaded via SageMaker S3 channel
        tok_path = os.path.join(args.tokenizer_channel, args.tokenizer_path)
        if os.path.isfile(tok_path):
            return BPETokenizer.load(tok_path)
        # Try finding any .pkl file in the channel
        for f in os.listdir(args.tokenizer_channel):
            if f.endswith(".pkl"):
                return BPETokenizer.load(os.path.join(args.tokenizer_channel, f))

    # Fall back to local path (e.g., included in source_dir)
    if os.path.isfile(args.tokenizer_path):
        return BPETokenizer.load(args.tokenizer_path)

    raise FileNotFoundError(
        f"Tokenizer not found at channel={args.tokenizer_channel} or path={args.tokenizer_path}. "
        "Either upload it to S3 and pass as an input channel, or include it in your source_dir."
    )


class TinyStoriesDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512, split="train", max_samples=None):
        self.dataset = dataset[split]
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            pad_len = self.max_length - len(tokens)
            tokens += [self.tokenizer.token2id.get('<pad>', 0)] * pad_len
        return torch.tensor(tokens, dtype=torch.long)

class WarmupLinearScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr_scale = float(self.current_step) / float(max(1, self.warmup_steps))
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr_scale = max(0.0, 1.0 - progress)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale

def train_and_evaluate(args):
    # Initialize Accelerator
    mixed_precision = "fp16" if args.amp else "no"
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    accelerate_set_seed(args.seed)

    # Use SageMaker model_dir for all outputs
    output_dir = args.model_dir

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(args.output_data_dir, exist_ok=True)
        # TensorBoard logs go to output_data_dir (separate from model artifacts)
        writer = SummaryWriter(log_dir=args.output_data_dir)
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
    else:
        writer = None

    accelerator.print(f"Device: {accelerator.device}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")
    accelerator.print(f"Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Model output dir: {output_dir}")

    # Load tokenizer
    tokenizer = load_tokenizer(args)

    # Load dataset (downloaded from HuggingFace on each instance)
    accelerator.print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)

    train_dataset = TinyStoriesDataset(
        dataset, tokenizer, max_length=args.max_seq_len,
        split="train", max_samples=args.max_train_samples
    )
    val_dataset = TinyStoriesDataset(
        dataset, tokenizer, max_length=args.max_seq_len,
        split="validation", max_samples=args.max_eval_samples
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Create model
    config = TinyStoriesConfig(
        vocab_size=len(tokenizer.token2id),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_seq_len,
        window_size=args.window_size,
    )
    model = TinyStoriesForCausalLM(config)

    total_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Model: {args.num_layers}L, {args.hidden_size}H, {args.num_heads}A")
    accelerator.print(f"Total parameters: {total_params:,}")
    accelerator.print(f"Effective batch size: {args.batch_size} x {accelerator.num_processes} = {args.batch_size * accelerator.num_processes}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr

    # Accelerate prepares everything
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    scheduler = WarmupLinearScheduler(optimizer, args.warmup_steps, total_steps)

    pad_token_id = tokenizer.token2id.get('<pad>', 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    raw_model = accelerator.unwrap_model(model)
    raw_model.use_amp = (accelerator.mixed_precision != "no")

    # Resume
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    train_losses = []

    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        accelerator.print(f"Resuming from: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.current_step = checkpoint['scheduler_state_dict'].get('current_step', 0)
        start_epoch = checkpoint.get('epoch', -1) + 1
        global_step = checkpoint.get('global_step', 0)
        accelerator.print(f"Resumed at epoch {start_epoch}, step {global_step}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        accelerator.print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        epoch_loss = 0

        if accelerator.is_main_process:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        else:
            progress_bar = train_dataloader

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                outputs = model(input_ids=inputs)
                logits = outputs["logits"]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    train_losses.append(loss.item())
                    avg_loss = sum(train_losses[-100:]) / min(len(train_losses), 100)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})
                    writer.add_scalar('Loss/train', avg_loss, global_step)
                    writer.add_scalar('Perplexity/train', np.exp(avg_loss), global_step)

                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_dataloader, criterion, accelerator)
                    if accelerator.is_main_process:
                        val_ppl = np.exp(val_loss)
                        print(f"Step {global_step}: Val loss: {val_loss:.4f}, PPL: {val_ppl:.2f}")
                        writer.add_scalar('Loss/val', val_loss, global_step)
                        writer.add_scalar('Perplexity/val', val_ppl, global_step)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(raw_model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                            print(f"New best model saved (val_loss={val_loss:.4f})")
                    model.train()

                if global_step % args.save_steps == 0 and accelerator.is_main_process:
                    torch.save({
                        'epoch': epoch, 'global_step': global_step,
                        'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': {'current_step': scheduler.current_step},
                        'loss': loss.item(),
                    }, os.path.join(output_dir, f"checkpoint-{global_step}.pth"))

            epoch_loss += loss.item()

        # End of epoch
        val_loss = evaluate(model, val_dataloader, criterion, accelerator)
        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}: Train loss={avg_epoch_loss:.4f} | Val loss={val_loss:.4f} | Val PPL={np.exp(val_loss):.2f}")
            writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch+1)
            writer.add_scalar('Loss/val_epoch', val_loss, epoch+1)
            torch.save(raw_model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"))

    # Save final model + tokenizer + config to model_dir (SageMaker uploads to S3)
    if accelerator.is_main_process:
        torch.save(raw_model.state_dict(), os.path.join(output_dir, "final_model.pth"))
        # Save config so we can reconstruct the model later
        with open(os.path.join(output_dir, "model_config.json"), "w") as f:
            json.dump(vars(config), f, indent=4)
        # Copy tokenizer to model_dir for easy deployment
        import shutil
        tok_src = args.tokenizer_path
        if args.tokenizer_channel:
            for fn in os.listdir(args.tokenizer_channel):
                if fn.endswith(".pkl"):
                    tok_src = os.path.join(args.tokenizer_channel, fn)
                    break
        if os.path.isfile(tok_src):
            shutil.copy2(tok_src, os.path.join(output_dir, "bpe_tokenizer_tinystories.pkl"))
        print(f"All artifacts saved to {output_dir}")
        writer.close()

    return raw_model, accelerator.device


def generate_text(model, tokenizer, prompt, device, max_length=100, temperature=1.0, top_k=0, top_p=0.9):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)]).to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, max_length=max_length,
                                     temperature=temperature, top_k=top_k, top_p=top_p)
    return tokenizer.decode(output_ids[0].tolist())


if __name__ == '__main__':
    args = parse_args()
    if args.pilot_run:
        args.max_train_samples = 1000
        args.max_eval_samples = 1000

    start_time = time.time()
    model, device = train_and_evaluate(args)
    elapsed = (time.time() - start_time) / 60

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"\nTraining completed in {elapsed:.2f} minutes")
        tokenizer = load_tokenizer(args)
        text = generate_text(model, tokenizer, "Once upon a time, there was a", device)
        print(f"\nSample: {text}")