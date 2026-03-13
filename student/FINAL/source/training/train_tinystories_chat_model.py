import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import time
import json
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

from training.bpe_tokenizer import BPETokenizer
from training.transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
from training.train_tinystories_model_accelerate import WarmupLinearScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Instruction-tune TinyStories for chat")

    parser.add_argument("--pretrained_model_path", type=str, required=True,
                   help="Path to base model checkpoint (reads config from args.json)")

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, default=None,
                   help="Path to local chat JSON (prompt/response pairs)")
    parser.add_argument("--dataset", type=str, default="bochen0909/tinystories-conversations",
                        help="HuggingFace dataset name")
    parser.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer_tinystories.pkl",
                        help="Path to BPE tokenizer")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Misc arguments
    parser.add_argument("--output_dir", type=str, default="tinystories_chat_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pilot_run", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=3000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()

def evaluate(model, val_dataloader, criterion, accelerator):
    """Evaluate on chat dataset (dict batches with input_ids/labels)."""
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"][:, :-1]
            labels = batch["labels"][:, 1:]
            outputs = model(input_ids=input_ids)
            logits = outputs["logits"]
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            gathered_loss = accelerator.gather(loss.unsqueeze(0)).mean()
            total_loss += gathered_loss.item()
            num_batches += 1
    return total_loss / max(num_batches, 1)

class ChatDataset(Dataset):
    """Loads prompt/response pairs. Returns tokens + labels with prompt masked."""

    def __init__(self, data, tokenizer, max_length=256, max_samples=None):
        if max_samples is not None:
            data = data[:max_samples]
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.token2id.get("<pad>", 0)
        self.ignore_index = -100  # PyTorch CrossEntropyLoss ignores this

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt_text = item["prompt"] + " "
        response_text = item["response"]

        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)

        # Combine and truncate
        all_tokens = prompt_tokens + response_tokens
        if len(all_tokens) > self.max_length:
            all_tokens = all_tokens[:self.max_length]

        prompt_len = min(len(prompt_tokens), len(all_tokens))

        # Labels: ignore prompt tokens, only learn response
        labels = [self.ignore_index] * prompt_len + all_tokens[prompt_len:]

        # Pad
        pad_len = self.max_length - len(all_tokens)
        if pad_len > 0:
            all_tokens += [self.pad_id] * pad_len
            labels += [self.ignore_index] * pad_len

        return {
            "input_ids": torch.tensor(all_tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_pretrained_model(model_path, tokenizer):
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "args.json")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            a = json.load(f)
        config = TinyStoriesConfig(
            vocab_size=len(tokenizer.token2id),
            hidden_size=a.get("hidden_size", 256),
            num_hidden_layers=a.get("num_layers", 4),
            num_attention_heads=a.get("num_heads", 8),
            intermediate_size=a.get("intermediate_size", 1024),
            hidden_dropout_prob=a.get("dropout", 0.1),
            attention_probs_dropout_prob=a.get("dropout", 0.1),
            max_position_embeddings=a.get("max_seq_len", 256),
            window_size=a.get("window_size", 256),
        )
        print(f"Config from args.json: {config.num_hidden_layers}L / "
              f"{config.hidden_size}H / {config.num_attention_heads}A")
    else:
        print("WARNING: args.json not found, using default config")
        config = TinyStoriesConfig(vocab_size=len(tokenizer.token2id))

    model = TinyStoriesForCausalLM(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded pretrained model: {total_params:,} params")

    return model, config

def main():
    args = parse_args()

    if args.pilot_run:
        args.max_train_samples = 1000
        args.max_eval_samples = 100
    else:
        args.max_train_samples = None
        args.max_eval_samples = None

    accelerator = Accelerator(mixed_precision="fp16" if args.amp else "no")
    accelerate_set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    accelerator.print(f"Device: {accelerator.device} | "
                      f"Processes: {accelerator.num_processes}")

    tokenizer = BPETokenizer.load(args.tokenizer_path)
    model, config = load_pretrained_model(args.pretrained_model_path, tokenizer)

    if args.dataset_path:
        # Local JSON with {"prompt": ..., "response": ...} pairs
        accelerator.print(f"Loading local dataset: {args.dataset_path}")
        with open(args.dataset_path, "r") as f:
            all_data = json.load(f)

        if args.pilot_run:
            all_data = all_data[:500]

        val_size = max(1, int(len(all_data) * args.val_split))
        train_data = all_data[val_size:]
        val_data = all_data[:val_size]

        train_dataset = ChatDataset(train_data, tokenizer, max_length=args.max_seq_len)
        val_dataset = ChatDataset(val_data, tokenizer, max_length=args.max_seq_len)

    elif args.dataset:
        from datasets import load_dataset
        accelerator.print(f"Loading HuggingFace dataset: {args.dataset}")
        hf_ds = load_dataset(args.dataset)

        def convert_hf_conversations(split_data, max_samples=None):
            if max_samples:
                split_data = split_data.select(range(min(max_samples, len(split_data))))
            pairs = []
            for item in split_data:
                conversation = item["conversation"]
                for i in range(0, len(conversation) - 1, 2):
                    prompt = conversation[i]["text"]
                    response = conversation[i + 1]["text"] if i + 1 < len(conversation) else ""
                    if response:
                        pairs.append({"prompt": prompt, "response": response})
            return pairs

        max_train = 500 if args.pilot_run else None
        max_val = 100 if args.pilot_run else None
        train_data = convert_hf_conversations(hf_ds["train"], max_train)
        val_data = convert_hf_conversations(hf_ds["valid"], max_val)

        train_dataset = ChatDataset(train_data, tokenizer, max_length=args.max_seq_len)
        val_dataset = ChatDataset(val_data, tokenizer, max_length=args.max_seq_len)

    else:
        raise ValueError("Provide either --dataset_path (local JSON) or --dataset (HuggingFace)")

    accelerator.print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = args.lr

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = WarmupLinearScheduler(optimizer, args.warmup_steps, total_steps)

    pad_id = tokenizer.token2id.get("<pad>", 0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # ignores both prompt tokens and padding
    raw_model = accelerator.unwrap_model(model)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []

    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        accelerator.print(f"Resuming from: {args.resume_from_checkpoint}")
        ckpt = torch.load(args.resume_from_checkpoint, map_location="cpu")
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.current_step = ckpt["scheduler_state_dict"].get("current_step", 0)
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("global_step", 0)

    accelerator.print(f"\nInstruction tuning: {args.epochs} epochs, lr={args.lr}, "
                      f"batch={args.batch_size}x{accelerator.num_processes}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        accelerator.print(f"\nEpoch {epoch+1}/{args.epochs}")
        model.train()
        epoch_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}",
                        disable=not accelerator.is_main_process)

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"][:, :-1]
            labels = batch["labels"][:, 1:]

            outputs = model(input_ids=input_ids)
            loss = criterion(outputs["logits"].reshape(-1, config.vocab_size),
                             labels.reshape(-1))

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()

            # Log
            if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                train_losses.append(loss.item())
                avg = sum(train_losses[-100:]) / min(len(train_losses), 100)
                progress.set_postfix({"loss": f"{avg:.4f}", "step": global_step})

            # Eval
            if global_step % args.eval_steps == 0:
                val_loss = evaluate(model, val_loader, criterion, accelerator)
                if accelerator.is_main_process:
                    val_ppl = np.exp(val_loss)
                    print(f"\n  Step {global_step}: val={val_loss:.4f}, ppl={val_ppl:.2f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(raw_model.state_dict(),
                                   os.path.join(args.output_dir, "best_model.pth"))
                        print(f"  New best model saved")
                model.train()

            # Checkpoint
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                torch.save({
                    "epoch": epoch, "global_step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": {"current_step": scheduler.current_step},
                }, os.path.join(args.output_dir, f"checkpoint-{global_step}.pth"))

        # End of epoch
        val_loss = evaluate(model, val_loader, criterion, accelerator)
        if accelerator.is_main_process:
            avg_train = epoch_loss / len(train_loader)
            accelerator.print(f"Epoch {epoch+1}: train={avg_train:.4f}, "
                              f"val={val_loss:.4f}, ppl={np.exp(val_loss):.2f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(raw_model.state_dict(),
                           os.path.join(args.output_dir, "best_model.pth"))
            torch.save(raw_model.state_dict(),
                       os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth"))

    if accelerator.is_main_process:
        torch.save(raw_model.state_dict(),
                   os.path.join(args.output_dir, "final_model.pth"))

        # Copy args.json from base model so downstream scripts get the architecture
        base_args = os.path.join(os.path.dirname(args.pretrained_model_path), "args.json")
        out_args = os.path.join(args.output_dir, "args.json")
        if os.path.exists(base_args):
            # Merge: keep base architecture, add chat-specific args
            with open(base_args, "r") as f:
                base_cfg = json.load(f)
            base_cfg.update({
                "chat_tuned": True,
                "chat_dataset": args.dataset_path,
                "chat_epochs": args.epochs,
                "chat_lr": args.lr,
            })
            with open(out_args, "w") as f:
                json.dump(base_cfg, f, indent=2)

        elapsed = time.time() - start_time
        accelerator.print(f"\nDone in {elapsed/60:.1f} min. Best val loss: {best_val_loss:.4f}")
        accelerator.print(f"Output: {args.output_dir}")


def generate_chat_response(model, tokenizer, prompt, device,
                           max_length=100, temperature=0.7, top_p=0.9):
    model.eval()
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=True)], dtype=torch.long
    ).to(device)
    input_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, max_length=max_length,
            temperature=temperature, top_p=top_p,
        )

    return tokenizer.decode(output_ids[0][input_len:].tolist())


if __name__ == "__main__":
    main()