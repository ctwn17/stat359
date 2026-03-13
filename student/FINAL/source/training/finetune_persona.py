import os
import torch
import argparse
import shutil
import time
import json
import numpy as np

from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

from training.bpe_tokenizer import BPETokenizer
from training.transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM
from training.train_tinystories_model_accelerate import WarmupLinearScheduler
from training.train_tinystories_chat_model import evaluate, ChatDataset


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
    else:
        config = TinyStoriesConfig(vocab_size=len(tokenizer.token2id))

    model = TinyStoriesForCausalLM(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model, config


def freeze_layers(model, num_freeze):
    if num_freeze <= 0:
        return
    for param in model.transformer.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.transformer.encoder.layers):
        if i < num_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Froze embeddings + {num_freeze} layers ({100*trainable/total:.1f}% trainable)")


def main():
    p = argparse.ArgumentParser(description="Fine-tune TinyStories on persona data")
    p.add_argument("--pretrained_model_path", type=str, required=True)
    p.add_argument("--persona_data", type=str, required=True)
    p.add_argument("--persona_name", type=str, required=True, choices=["cowboy", "shy"])
    p.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer_tinystories.pkl")
    p.add_argument("--output_dir", type=str, default="persona_models/default")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--freeze_layers", type=int, default=0)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--pilot_run", action="store_true")
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=200)
    args = p.parse_args()

    accelerator = Accelerator(mixed_precision="fp16" if args.amp else "no")
    accelerate_set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "finetune_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    accelerator.print(f"Device: {accelerator.device} | Processes: {accelerator.num_processes}")

    tokenizer = BPETokenizer.load(args.tokenizer_path)
    model, config = load_pretrained_model(args.pretrained_model_path, tokenizer)
    accelerator.print(f"Model: {config.num_hidden_layers}L / {config.hidden_size}H / "
                      f"{sum(p.numel() for p in model.parameters()):,} params")

    if args.freeze_layers > 0:
        freeze_layers(model, args.freeze_layers)

    max_samples = 200 if args.pilot_run else None
    with open(args.persona_data, "r") as f:
        all_data = json.load(f)
    if max_samples:
        all_data = all_data[:max_samples]

    val_size = max(1, int(len(all_data) * args.val_split))
    train_data = all_data[val_size:]
    val_data = all_data[:val_size]

    train_dataset = ChatDataset(train_data, tokenizer, max_length=args.max_seq_len)
    val_dataset = ChatDataset(val_data, tokenizer, max_length=args.max_seq_len)
    accelerator.print(f"Dataset: {len(all_data)} ({len(train_data)} train / {len(val_data)} val)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = args.lr

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    scheduler = WarmupLinearScheduler(optimizer, args.warmup_steps,
                                       len(train_loader) * args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    raw_model = accelerator.unwrap_model(model)

    base_val_loss = evaluate(model, val_loader, criterion, accelerator)
    accelerator.print(f"Base val loss: {base_val_loss:.4f}, ppl: {np.exp(base_val_loss):.2f}")

    accelerator.print(f"\nFine-tuning: {args.epochs} epochs, lr={args.lr}, "
                      f"batch={args.batch_size}x{accelerator.num_processes}")

    best_val_loss = float("inf")
    global_step = 0
    train_losses = []
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                        disable=not accelerator.is_main_process)

        for batch in progress:
            input_ids = batch["input_ids"][:, :-1]
            labels = batch["labels"][:, 1:]
            outputs = model(input_ids=input_ids)
            loss = criterion(outputs["logits"].reshape(-1, config.vocab_size), labels.reshape(-1))

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item()

            if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                train_losses.append(loss.item())
                avg = sum(train_losses[-50:]) / min(len(train_losses), 50)
                progress.set_postfix({"loss": f"{avg:.4f}"})

            if global_step % args.eval_steps == 0:
                val_loss = evaluate(model, val_loader, criterion, accelerator)
                if accelerator.is_main_process and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(raw_model.state_dict(),
                               os.path.join(args.output_dir, "best_model.pth"))
                    print(f"\n  Step {global_step}: new best val_loss={val_loss:.4f}")
                model.train()

        # End of epoch
        val_loss = evaluate(model, val_loader, criterion, accelerator)
        if accelerator.is_main_process:
            accelerator.print(f"Epoch {epoch+1}: train={epoch_loss/len(train_loader):.4f}, "
                              f"val={val_loss:.4f}, ppl={np.exp(val_loss):.2f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(raw_model.state_dict(),
                           os.path.join(args.output_dir, "best_model.pth"))

    if accelerator.is_main_process:
        torch.save(raw_model.state_dict(), os.path.join(args.output_dir, "final_model.pth"))

        # Copy args.json from base model so generation scripts can load config
        base_args = os.path.join(os.path.dirname(args.pretrained_model_path), "args.json")
        if os.path.exists(base_args):
            shutil.copy(base_args, os.path.join(args.output_dir, "args.json"))

        elapsed = time.time() - start_time
        accelerator.print(f"\nDone in {elapsed/60:.1f} min. "
                          f"Best val loss: {best_val_loss:.4f} (base: {base_val_loss:.4f})")
        accelerator.print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()