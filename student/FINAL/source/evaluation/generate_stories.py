import torch
import argparse
import json
import os
import time
from datetime import datetime

from training.bpe_tokenizer import BPETokenizer
from training.transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM

# These are the prompts used for generating "Tell me a story that starts with..." prompts
STORY_PROMPTS = [
    "Once upon a time, there was a little dog named Max.",
    "There was a girl named Lily who loved to play in the garden.",
    "One day, a boy found a shiny red ball in the park.",
    "A little cat was sitting on the window and looking outside.",
    "There was a big tree in the middle of the forest.",
    "A small bird was flying over the blue lake.",
    "One morning, a bunny woke up and saw snow everywhere.",
    "There was a little fish that lived in a pond.",
    "A girl named Sara had a favorite toy bear.",
    "One day, the sun was very bright and warm.",
    "There was a boy who wanted to be very brave.",
    "A little mouse lived in a hole in the wall.",
    "One afternoon, two friends went to the beach.",
    "There was a baby duck that could not swim yet.",
    "A kind old woman lived in a small house on a hill.",
    "One night, a little boy heard a strange sound.",
    "There was a pretty butterfly in the garden.",
    "A girl and her mom went to the store together.",
    "One day, a puppy got lost in the big city.",
    "There was a little boat on the river.",
    "A boy named Tom had a new pair of shoes.",
    "One morning, a frog jumped out of the pond.",
    "There was a happy family that lived near the sea.",
    "A little girl found a key on the ground.",
    "One day, it started to rain very hard.",
    "There was a tall man who made yummy cookies.",
    "A boy wanted to learn how to ride a bike.",
    "One evening, the stars came out to play.",
    "There was a gentle horse in the green field.",
    "A girl named Emma had a secret garden.",
    "One day, a bear came out of the cave.",
    "There was a tiny ant carrying a big leaf.",
    "A boy and his dad went fishing at the lake.",
    "One morning, a rainbow appeared in the sky.",
    "There was a magic door at the end of the hall.",
    "A little lamb was looking for its mother.",
    "One day, the wind blew very strong.",
    "There was a wise owl sitting in the tree.",
    "A girl wanted to make a present for her friend.",
    "One afternoon, a squirrel found a big nut.",
    "There was a castle made of sand on the beach.",
    "A boy named Jack climbed a very tall hill.",
    "One day, a flower started to grow in the yard.",
    "There was a friendly dragon who loved to sing.",
    "A little turtle was walking very slowly.",
    "One morning, the rooster forgot to crow.",
    "There was a bright star that shone every night.",
    "A girl and her dog went on an adventure.",
    "One day, a snowman came to life.",
    "There was a little train that went choo choo.",
]


# These examples are used for both one shot and few shot prompts.
EXAMPLES = {
    "neutral": [
        "Lily found a flower in the garden. She showed it to her mom. They put it in water and it made the room nice.",
        "Sam had a red ball. One day it went over the fence. His neighbor threw it back and they became friends.",
        "A cat saw a bird outside. It jumped and jumped but could not reach. The cat got tired and fell asleep.",
    ],
    "shy": [
        "Rose was shy. She quietly left a flower for the new girl next door. The girl whispered thank you and they became gentle friends.",
        "A tiny mouse tiptoed out to find berries. He heard a sound and froze. It was just a leaf. He went home feeling a little brave.",
        "A shy bunny sat very still. A butterfly landed on his nose. He whispered hello. Sometimes friends come when you are quiet.",
    ],
    "cowboy": [
        "Howdy! Cowboy Jake rode his horse Star across the dusty trail. They found a lost sheep and brought it home. Yee-haw!",
        "Cowgirl Rosie put on her boots and hat. She rode to the creek. Howdy morning she said to the chickens on the ranch.",
        "A big storm came to the ranch. Dusty said giddy up partner! He got all the horses to the barn safe and sound.",
    ],
}

# These prompts are used for persona tuning via zero-shot prompt embedding
PERSONALITY_INSTRUCTION = {
    "shy": "Tell me a quiet, gentle story. Use words like quietly, whispered, gently, and tiptoed. ",
    "cowboy": "Tell me a cowboy story. Use words like howdy, partner, yee-haw, giddy up, and ranch. ",
}

# This prefix is added to Chain of Thought promps.
COT_PREFIX_CHAT = "Think step by step about what will happen, then tell the story. "


# We generate 50 prompts (1 for each story) for each of these cases
ALL_CONDITIONS = [
    {'model': 'baseline', 'persona': 'neutral', 'shot_type': 'zero'},
    {'model': 'baseline', 'persona': 'neutral', 'shot_type': 'one'},
    {'model': 'baseline', 'persona': 'neutral', 'shot_type': 'few'},
    {'model': 'baseline', 'persona': 'neutral', 'shot_type': 'cot'},

    {'model': 'baseline', 'persona': 'shy', 'shot_type': 'zero'},
    {'model': 'baseline', 'persona': 'shy', 'shot_type': 'one'},
    {'model': 'baseline', 'persona': 'shy', 'shot_type': 'few'},
    {'model': 'baseline', 'persona': 'shy', 'shot_type': 'cot'},

    {'model': 'baseline', 'persona': 'cowboy', 'shot_type': 'zero'},
    {'model': 'baseline', 'persona': 'cowboy', 'shot_type': 'one'},
    {'model': 'baseline', 'persona': 'cowboy', 'shot_type': 'few'},
    {'model': 'baseline', 'persona': 'cowboy', 'shot_type': 'cot'},

    {'model': 'shy', 'persona': 'shy', 'shot_type': 'zero'},
    {'model': 'shy', 'persona': 'shy', 'shot_type': 'one'},
    {'model': 'shy', 'persona': 'shy', 'shot_type': 'few'},
    {'model': 'shy', 'persona': 'shy', 'shot_type': 'cot'},

    {'model': 'cowboy', 'persona': 'cowboy', 'shot_type': 'zero'},
    {'model': 'cowboy', 'persona': 'cowboy', 'shot_type': 'one'},
    {'model': 'cowboy', 'persona': 'cowboy', 'shot_type': 'few'},
    {'model': 'cowboy', 'persona': 'cowboy', 'shot_type': 'cot'}
]

PROMPT_PREFIX = 'Tell me a story that starts with:'


def format_examples(examples, n):
    prefix = ""
    for i, ex in enumerate(examples[:n]):
        prefix += f"Example: {ex}\n\n"
    return prefix


def build_prompt(story_prompt, condition):
    model = condition['model']
    persona = condition['persona']
    shot_type = condition['shot_type']

    user_msg = ""

    # Personality instruction (only for prompt method, Paper 1)
    if model == 'baseline' and persona != 'neutral':
        user_msg += PERSONALITY_INSTRUCTION[persona]

    # Examples (Paper 1: few-shot style exemplars)
    example_key = persona if persona else "neutral"
    if example_key not in EXAMPLES:
        example_key = "neutral"

    if shot_type == "one":
        user_msg += format_examples(EXAMPLES[example_key], 1)
    elif shot_type == "few":
        user_msg += format_examples(EXAMPLES[example_key], 3)
    elif shot_type == "cot":
        user_msg += COT_PREFIX_CHAT

    # Story request + prompt always last
    user_msg += f"{PROMPT_PREFIX} {story_prompt}"

    return user_msg



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


def generate(model, tokenizer, prompt, device, max_length=200,
             temperature=0.8, top_k=50, top_p=0.9):
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=True)], dtype=torch.long
    ).to(device)
    input_len = input_ids.shape[1]
    eos_id = tokenizer.token2id.get("<eos>", None)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids, max_length=max_length,
            temperature=temperature, top_k=top_k, top_p=top_p,
            eos_token_id=eos_id,
        )

    # Only decode the NEW tokens (skip the input prompt)
    generated_tokens = output_ids[0].tolist()
    return tokenizer.decode(generated_tokens)



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--cowboy_model_path", type=str, default=None)
    p.add_argument("--shy_model_path", type=str, default=None)
    p.add_argument("--tokenizer_path", type=str, default="bpe_tokenizer_tinystories.pkl")
    p.add_argument("--output_dir", type=str, default="eval_output")
    p.add_argument("--conditions", nargs="+", default=None)
    p.add_argument("--max_length", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--pilot_run", action="store_true", help="Use 5 prompts only")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    print(f"Device: {device}")

    tokenizer = BPETokenizer.load(args.tokenizer_path)

    # Load models
    print("Loading base model...")
    models = {"baseline": load_model(args.base_model_path, tokenizer, device)}

    if args.shy_model_path:
        print("Loading shy model...")
        models["shy"] = load_model(args.shy_model_path, tokenizer, device)

    if args.cowboy_model_path:
        print("Loading cowboy model...")
        models["cowboy"] = load_model(args.cowboy_model_path, tokenizer, device)

    # Determine conditions
    conditions = args.conditions or ALL_CONDITIONS
    if "shy" not in models:
        conditions = [c for c in conditions if c['persona'] != 'shy']
    if "cowboy" not in models:
        conditions = [c for c in conditions if c['persona'] != 'cowboy']

    prompts = STORY_PROMPTS[:5] if args.pilot_run else STORY_PROMPTS
    total = len(prompts) * len(conditions)

    print(f"\n{len(conditions)} conditions × {len(prompts)} prompts = {total} stories")

    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Generate
    all_results = []
    generated = 0
    start_time = time.time()

    for condition in conditions:
        model_key = condition['model']
        persona_key = condition['persona']
        shot_type_key = condition['shot_type']
        output_filename = f"{model_key}_{persona_key}_{shot_type_key}.json"
        model = models[model_key]
        cond_results = []

        for i, story_prompt in enumerate(prompts):
            full_prompt = build_prompt(story_prompt, condition)
            story = generate(model, tokenizer, full_prompt, device,
                             args.max_length, args.temperature, args.top_k, args.top_p)

            result = {
                "prompt_id": i,
                "model": model_key,
                "persona": persona_key,
                "shot_type": shot_type_key,
                "original_prompt": story_prompt,
                "full_prompt": full_prompt,
                "generated_story": story[story.find(PROMPT_PREFIX) + len(PROMPT_PREFIX):],
            }
            cond_results.append(result)
            all_results.append(result)
            generated += 1

            if generated % 50 == 0:
                print(f"  {generated}/{total} ({condition})")

        with open(os.path.join(output_dir, output_filename), "w") as f:
            json.dump(cond_results, f, indent=2)

    with open(os.path.join(output_dir, "all_stories.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nDone: {len(all_results)} stories in {elapsed/60:.1f} min")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()