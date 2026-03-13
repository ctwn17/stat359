import json
import argparse
import os
import torch
import numpy as np

from training.bpe_tokenizer import BPETokenizer
from training.transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM


"""
METRIC 1: Style Strength
Measures how strongly the persona vocabulary appears in the output.
"""

PERSONA_KEYWORDS = {
    "cowboy": [
        "howdy", "partner", "yee-haw", "yeehaw", "giddy", "ranch", "cowboy",
        "cowgirl", "horse", "rode", "saddle", "barn", "dusty", "trail",
        "campfire", "boots", "hat", "sunset", "lasso", "herd",
    ],
    "shy": [
        "quietly", "gently", "whispered", "softly", "tiptoed", "shy",
        "timid", "scared", "nervous", "careful", "gentle", "still",
        "calm", "peaceful", "silent", "slowly", "tiny", "little",
    ],
}


def compute_style_strength(story, persona):
    """
    Determines the style strength of the persona in the story by
    by measuring the ratio of persona keywords to total words.

    Args
        - story: The story text to analyze.
        - persona: The persona to evaluate.

    Returns
        - The style strength score, or None if persona is not recognized.
    """
    if persona not in PERSONA_KEYWORDS:
        return None

    words = story.lower().split()
    if len(words) == 0:
        return 0.0

    keywords = set(PERSONA_KEYWORDS[persona])
    hits = sum(1 for w in words if any(kw in w for kw in keywords))
    return hits / len(words)


"""
METRIC 2: Persona Consistency

Persona Consistency measures whether persona is maintained across the conversation.
We do this by comparing persona keyword density in the first half vs
second half of the story. Consistent persona = similar density in both.
To normalize the data we return the log2 of the ratio. This prevents
1/2 from misrepresenting persona consistency compared to 2/1 even though
they are semantically the same.
"""
def compute_persona_consistency(story, persona):
    """
        Determines the persona consistency of the persona in the story by
        by measuring the ratio of persona keywords in the first have
        to the ratio of persona keywords in the second half.

        1.0 = perfectly consistent. <1.0 = persona fades. >1.0 = persona grows.

        Args
            - story: The story text to analyze.
            - persona: The persona to evaluate.

        Returns
            - The persona consistency score, or None if persona is not recognized.
    """
    if persona not in PERSONA_KEYWORDS:
        return None

    words = story.lower().split()
    if len(words) < 10:
        return None

    mid = len(words) // 2
    first_half = words[:mid]
    second_half = words[mid:]
    keywords = set(PERSONA_KEYWORDS[persona])

    density_first = sum(1 for w in first_half if any(kw in w for kw in keywords)) / len(first_half)
    density_second = sum(1 for w in second_half if any(kw in w for kw in keywords)) / len(second_half)

    if density_first == 0:
        return 0.0 if density_second == 0 else 2.0

    return np.log2(density_second + 1e-4 / density_first + 1e-4)


"""
METRIC 3: Distinct-N

Measures n-gram diversity. Higher = more creative/varied vocabulary.
distinct-1 = unique unigrams / total unigrams
distinct-2 = unique bigrams / total bigrams
"""

def compute_distinct_n(story, n=2):
    """
        This function computes the ratio of unique n-grams to total n-grams.

        An n-gram is a sequence of consecutive words n long. Having too many repeated n-grams
        can show staleness or repetition.

        Args
            - story: The story text to analyze.
            - n: The number of consecutive words to consider.

        Returns
            - The ration of distinct n-grams to total n-grams.
    """
    words = story.lower().split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if len(ngrams) == 0:
        return 0.0

    return len(set(ngrams)) / len(ngrams)


"""
METRIC 4: Perplexity

Lower perplexity under the BASE model = more coherent/natural story.
This measures whether the model produces language that's consistent
with its training distribution.
"""

def compute_perplexity(story, model, tokenizer, device, max_length=256):
    """
        Perplexity of the story under the given model.
        Lower = more coherent / natural for that model.

        Args
            - story: The story text to analyze.
            - model: The model to use for perplexity calculation.
            - tokenizer: The tokenizer for the model.
            - device: The device to run the model on.
            - max_length: The maximum length of the story to consider.
        Returns
            - The perplexity score, or None if the story is too short or too long.
    """
    tokens = tokenizer.encode(story, add_special_tokens=True)
    if len(tokens) < 2:
        return None
    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    with torch.no_grad():
        outputs = model(input_ids=inputs)
        logits = outputs["logits"]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    return torch.exp(loss).item()

def load_model(model_path, tokenizer, device):
    config_path = os.path.join(os.path.dirname(model_path), "args.json")
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_story(story_data, base_model=None, tokenizer=None, device=None):
    """Compute all metrics for a single story."""
    story = story_data["generated_story"]
    persona = story_data.get("persona", "none")
    if persona == "none":
        persona = None

    metrics = {}


    metrics["style_strength"] = compute_style_strength(story, persona)
    metrics["persona_consistency"] = compute_persona_consistency(story, persona)
    metrics["distinct_1"] = compute_distinct_n(story, 1)
    metrics["distinct_2"] = compute_distinct_n(story, 2)

    # Perplexity (optional, needs model)
    if base_model is not None and tokenizer is not None:
        metrics["perplexity"] = compute_perplexity(
            story, base_model, tokenizer, device
        )
    else:
        metrics["perplexity"] = None

    return metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate generated stories")
    p.add_argument("--input", type=str, required=True,
                   help="Path to all_stories.json")
    p.add_argument("--base_model_path", type=str, default=None,
                   help="Base model for perplexity (optional)")
    p.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer_tinystories.pkl")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    # Load stories
    with open(args.input, "r") as f:
        stories = json.load(f)
    print(f"Loaded {len(stories)} stories")

    # Load model for perplexity
    base_model, tokenizer, device = None, None, None
    if args.base_model_path:
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                              else args.device if args.device != "auto" else "cpu")
        tokenizer = BPETokenizer.load(args.tokenizer_path)
        print(f"Loading base model for perplexity...")
        base_model = load_model(args.base_model_path, tokenizer, device)

    # Evaluate
    print("Evaluating...")
    results = []
    for i, story_data in enumerate(stories):
        metrics = evaluate_story(story_data, base_model, tokenizer, device)
        results.append({**story_data, "metrics": metrics})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(stories)}")

    # Save
    output_dir = os.path.dirname(args.input)
    output_file = os.path.join(output_dir, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

    # ── Summary ───────────────────────────────────────────────────────────
    from collections import defaultdict
    cond_metrics = defaultdict(lambda: defaultdict(list))

    for r in results:
        cond = f"{r.get('model')}_{r.get('persona', 'none')}_{r.get('shot_type', 'none')}"
        m = r.get("metrics")
        if not isinstance(m, dict):
            continue
        for key, val in m.items():
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                cond_metrics[cond][key].append(val)

    # Print table
    print(f"\n{'='*120}")
    print(f"{'Condition':<25s} {'Style':>7s} {'Consist':>8s} {'Dist-1':>7s} "
          f"{'Dist-2':>7s} {'Relev':>7s} {'PPL':>8s}")
    print("-" * 120)

    for cond in sorted(cond_metrics.keys()):
        cm = cond_metrics[cond]
        row = f"{cond:<25s}"

        for key in ["style_strength", "persona_consistency", "distinct_1",
                     "distinct_2", "prompt_relevance"]:
            if key in cm and cm[key]:
                row += f" {np.mean(cm[key]):>7.3f}"
            else:
                row += f" {'--':>7s}"

        if "perplexity" in cm and cm["perplexity"]:
            row += f" {np.mean(cm['perplexity']):>8.1f}"
        else:
            row += f" {'--':>8s}"


        print(row)

    print("=" * 120)
    print(f"\nResults: {output_file}")


if __name__ == "__main__":
    main()