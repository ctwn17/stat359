import json
import argparse
import random
import re
from datasets import load_dataset


# Simple Requests (60%)
SIMPLE_PROMPTS = [
    "Tell me a story.",
    "Can you tell me a story?",
    "Tell me a fun story.",
    "I want to hear a story.",
    "Please tell me a story.",
    "Tell me a short story.",
    "Tell me a nice story.",
    "Can you make up a story?",
    "I would like a story please.",
    "Tell me a bedtime story.",
    "Tell me a happy story.",
    "Tell me a story for kids.",
]

# These get the story's first sentence appended as context
SIMPLE_PROMPTS_WITH_START = [
    "Tell me a story that starts with: {start}",
    "Can you tell a story that begins: {start}",
    "Write a story starting with: {start}",
    "Tell me a story. Start with: {start}",
    "Make up a story beginning with: {start}",
]

# Personality Requests (15%)
PERSONALITY_PROMPTS = {
    "shy": [
        "Tell me a quiet, gentle story.",
        "Tell me a shy, soft story. Use words like quietly and whispered.",
        "Can you tell a gentle story? Make it calm and peaceful.",
        "Tell me a story about someone shy.",
        "Tell me a soft, quiet story with gentle words.",
    ],
    "cowboy": [
        "Tell me a cowboy story.",
        "Tell me a story about a cowboy on a ranch.",
        "Can you tell a cowboy adventure? Use words like howdy and partner.",
        "Tell me a story with cowboys and horses.",
        "Tell me a fun ranch story. Say yee-haw!",
    ],
    "happy": [
        "Tell me a very happy story!",
        "Tell me an exciting, joyful story!",
        "Can you tell a story where everyone is happy?",
        "Tell me a story full of fun and laughter!",
        "Tell me a wonderful, amazing story!",
    ],
    "sad": [
        "Tell me a sad story.",
        "Tell me a story where something goes wrong but it gets better.",
        "Can you tell a story that is a little sad?",
        "Tell me a story about losing something.",
    ],
}

# CoT Requests (15%)
COT_PROMPTS = [
    "Think step by step about what will happen, then tell me a story.",
    "First plan what the story is about, then tell it to me.",
    "Think about the beginning, middle, and end, then tell the story.",
    "Plan the story step by step, then write it.",
    "First think about what happens, then tell the story.",
]

COT_TEMPLATES = [
    "Let me think. The story is about {topic}. First, {first}. Then, {then}. In the end, {end}. Here is the story:\n\n{story}",
    "Step by step: First, {first}. Next, {then}. Finally, {end}.\n\n{story}",
    "I will plan the story. It is about {topic}. {first}. Then {then}. At the end, {end}.\n\n{story}",
]

# One-Shot Example Requests (10%)
ONESHOT_PROMPTS = [
    "Here is an example story: {example}\n\nNow tell me a new story like that.",
    "Example: {example}\n\nTell me a similar story.",
    "Like this story: {example}\n\nCan you tell me another one?",
]

# Helpers
def get_first_sentence(text):
    """Extract first sentence from a story."""
    match = re.match(r'^(.+?[.!?])\s', text)
    if match:
        return match.group(1)
    words = text.split()
    return " ".join(words[:10]) + "."


def extract_topic(text):
    """Pull a rough topic from a story (first noun phrase)."""
    topics = [
        "a little girl", "a boy", "a dog", "a cat", "a bird", "a bunny",
        "a bear", "a fish", "friends", "a family", "a mouse", "a duck",
        "a flower", "a tree", "a ball", "a toy", "a horse", "a farm",
    ]
    text_lower = text.lower()
    for t in topics:
        if t in text_lower:
            return t
    return "a child"


def extract_events(text):
    """Pull rough first/then/end from a story."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) >= 3:
        first = sentences[0].lower()
        then = sentences[len(sentences)//2].lower()
        end = sentences[-1].lower()
    elif len(sentences) == 2:
        first = sentences[0].lower()
        then = "something happens"
        end = sentences[1].lower()
    else:
        first = "something begins"
        then = "something happens"
        end = "it ends well"
    return first, then, end


def truncate_story(text, max_words=80):
    """Truncate story to fit in context window."""
    words = text.split()
    if len(words) <= max_words:
        return text
    # Try to end at a sentence boundary
    truncated = " ".join(words[:max_words])
    last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_period > len(truncated) // 2:
        return truncated[:last_period + 1]
    return truncated + "."


def truncate_example(text, max_words=30):
    """Make a very short version for one-shot examples."""
    words = text.split()
    truncated = " ".join(words[:max_words])
    last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_period > 10:
        return truncated[:last_period + 1]
    return truncated + "."


# Builder
def build_simple(story):
    """60% — simple request → story."""
    story = truncate_story(story)

    if random.random() < 0.5:
        prompt = random.choice(SIMPLE_PROMPTS)
    else:
        start = get_first_sentence(story)
        prompt = random.choice(SIMPLE_PROMPTS_WITH_START).format(start=start)

    return {"prompt": prompt, "response": story}


def build_personality(story):
    """15% — personality-styled request → story."""
    story = truncate_story(story)
    persona = random.choice(list(PERSONALITY_PROMPTS.keys()))
    prompt = random.choice(PERSONALITY_PROMPTS[persona])
    return {"prompt": prompt, "response": story}


def build_cot(story):
    """15% — CoT request → reasoning + story."""
    story = truncate_story(story, max_words=60)

    topic = extract_topic(story)
    first, then, end = extract_events(story)
    prompt = random.choice(COT_PROMPTS)
    template = random.choice(COT_TEMPLATES)
    response = template.format(topic=topic, first=first, then=then, end=end, story=story)

    return {"prompt": prompt, "response": response}


def build_oneshot(story, example_story):
    """10% — one-shot example → new story."""
    story = truncate_story(story)
    example = truncate_example(example_story)
    prompt = random.choice(ONESHOT_PROMPTS).format(example=example)
    return {"prompt": prompt, "response": story}


# Main
def main():
    p = argparse.ArgumentParser(description="Build chat instruction-tuning dataset")
    p.add_argument("--count", type=int, default=10000,
                   help="Total examples to generate")
    p.add_argument("--output", type=str, default="chat_dataset.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_stories", type=int, default=50000,
                   help="Max stories to load from TinyStories (for speed)")
    args = p.parse_args()

    random.seed(args.seed)

    print(f"Loading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if args.max_stories and len(ds) > args.max_stories:
        ds = ds.select(range(args.max_stories))
    stories = [item["text"] for item in ds]
    print(f"Loaded {len(stories)} stories")

    # Shuffle and allocate
    random.shuffle(stories)
    n = args.count
    n_simple = int(n * 0.60)
    n_personality = int(n * 0.15)
    n_cot = int(n * 0.15)
    n_oneshot = n - n_simple - n_personality - n_cot

    print(f"Building {n} examples: {n_simple} simple, {n_personality} personality, "
          f"{n_cot} CoT, {n_oneshot} one-shot")

    dataset = []
    idx = 0

    # Simple
    for i in range(n_simple):
        dataset.append(build_simple(stories[idx % len(stories)]))
        idx += 1

    # Personality
    for i in range(n_personality):
        dataset.append(build_personality(stories[idx % len(stories)]))
        idx += 1

    # CoT
    for i in range(n_cot):
        dataset.append(build_cot(stories[idx % len(stories)]))
        idx += 1

    # One-shot (needs two stories: example + actual)
    for i in range(n_oneshot):
        example_idx = (idx + 1) % len(stories)
        dataset.append(build_oneshot(stories[idx % len(stories)],
                                               stories[example_idx]))
        idx += 2

    # Shuffle final dataset
    random.shuffle(dataset)

    # Save
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    # Stats
    prompt_lens = [len(item["prompt"].split()) for item in dataset]
    response_lens = [len(item["response"].split()) for item in dataset]
    print(f"\nSaved {len(dataset)} examples to {args.output}")
    print(f"Avg prompt words: {sum(prompt_lens)/len(prompt_lens):.0f}")
    print(f"Avg response words: {sum(response_lens)/len(response_lens):.0f}")

    # Samples
    print(f"\n{'='*60}")
    for label in ["Simple", "Personality", "CoT", "One-shot"]:
        for item in dataset:
            p = item["prompt"]
            if label == "Simple" and "Think" not in p and "Example" not in p and "quiet" not in p.lower() and "cowboy" not in p.lower():
                match = item
                break
            elif label == "Personality" and any(k in p.lower() for k in ["quiet", "cowboy", "happy", "sad"]):
                match = item
                break
            elif label == "CoT" and "step" in p.lower():
                match = item
                break
            elif label == "One-shot" and "Example:" in p:
                match = item
                break
        else:
            continue
        print(f"\n--- {label} ---")
        print(f"  Prompt:   {match['prompt'][:150]}")
        print(f"  Response: {match['response'][:150]}...")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()