import json
import argparse
import os
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("numpy required: pip install numpy")
    exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def _condition_key(item):
    """Get condition key from item"""
    return f"{item.get('model', '')}_{item.get('persona', '')}_{item.get('shot_type', '')}"


def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    by_cond = defaultdict(list)
    for item in data:
        by_cond[_condition_key(item)].append(item)
    return data, by_cond


def get_metric_values(items, metric):
    vals = []
    for item in items:
        v = item.get("metrics", {}).get(metric)
        if v is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(v)
    return vals


# Charts
def plot_grouped_bars(by_cond, output_dir):
    """
    This function generates the graphs for each of the defined "plot_metrics".
    One chart per metric: grouped by persona method, colored by shot type.
    """
    if not HAS_PLT:
        print("Install matplotlib for charts: pip install matplotlib")
        return

    # Shot Type ordered keys to ensure consistency across groups. Keys are model_tone_shottype
    groups = {
        "Baseline": ["baseline_neutral_zero", "baseline_neutral_one", "baseline_neutral_few", "baseline_neutral_cot"],
        "Prompt:Shy": ["baseline_shy_zero", "baseline_shy_one", "baseline_shy_few", "baseline_shy_cot"],
        "Prompt:Cowboy": ["baseline_cowboy_zero", "baseline_cowboy_one", "baseline_cowboy_few", "baseline_cowboy_cot"],
        "Embed:Shy": ["shy_shy_zero", "shy_shy_one", "shy_shy_few", "shy_shy_cot"],
        "Embed:Cowboy": ["cowboy_cowboy_zero", "cowboy_cowboy_one", "cowboy_cowboy_few", "cowboy_cowboy_cot"],
    }
    shots = ["zero", "one", "few", "cot"]
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]

    plot_metrics = [
        {
            'metric': 'distinct_1',
            'name': 'Unique Unigram (%)'
        },
        {
            'metric': 'distinct_2',
            'name': 'Unique Bigrams (%)'
        },
        {
            'metric': 'style_strength',
            'name': 'Style Strength',
        },
        {
            'metric': 'perplexity',
            'name': 'Perplexity',
        },
    ]

    for metric in plot_metrics:
        fig, ax = plt.subplots(figsize=(12, 4 if metric['metric'] == 'style_strength' else 5))
        x = np.arange(len(groups))
        w = 0.18
        group_keys = []

        has_data = False
        for j, (shot, color) in enumerate(zip(shots, colors)):
            means, stds = [], []
            for gname, conds in groups.items():
                if metric['metric'] == 'style_strength' and gname == "Baseline":
                    x = np.arange(len(groups)-1)
                    continue
                vals = get_metric_values(by_cond.get(conds[j], []), metric['metric'])
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    has_data = True
                else:
                    means.append(0)
                    stds.append(0)
                if gname not in group_keys:
                    group_keys.append(gname)
            x_positions = x + (j - 1.5) * w
            ax.errorbar(
                x_positions,
                means,
                yerr=stds,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=4,
                markersize=7,
                label=shot,
                alpha=0.9,
            )

        if not has_data:
            plt.close()
            continue

        # Distinct-N metrics are percentages, so we want to display the y-axis as 0-100 instead of 0-1
        if(metric['metric'].startswith('distinct')):
            ax.set_ylim(0, 1.0)
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}"))

        ax.set_ylabel(metric['name'])
        ax.set_title(f"{metric['name']} by Persona Method & Shot Type")
        ax.set_xticks(x)
        ax.set_xticklabels(group_keys, fontsize=9)
        ax.legend(title="Shot type")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        path = os.path.join(output_dir, f"chart_{metric['metric']}.png")
        print(path)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Chart: {path}")

def plot_persona_consistency(by_cond, output_dir):
    """
    Plot persona consistency by persona method and shot type.
    This is different to our grouped bars because we persona consistency
    is concerned with bias towards the first and second half.
    We want to change the y-axis to be the same on both sides to avoid
    visual skew and keep balanced consistency right in the middle.
    We omit the baseline-neutral group because there is no persona.
    """
    if not HAS_PLT:
        print("Install matplotlib for charts: pip install matplotlib")
        return

    groups = {
        "Prompt:Shy": ["baseline_shy_zero", "baseline_shy_one", "baseline_shy_few", "baseline_shy_cot"],
        "Prompt:Cowboy": ["baseline_cowboy_zero", "baseline_cowboy_one", "baseline_cowboy_few", "baseline_cowboy_cot"],
        "Embed:Shy": ["shy_shy_zero", "shy_shy_one", "shy_shy_few", "shy_shy_cot"],
        "Embed:Cowboy": ["cowboy_cowboy_zero", "cowboy_cowboy_one", "cowboy_cowboy_few", "cowboy_cowboy_cot"],
    }
    shots = ["zero", "one", "few", "cot"]
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(groups))
    w = 0.18
    group_labels = list(groups.keys())

    has_data = False

    plot_values = []
    for j, (shot, color) in enumerate(zip(shots, colors)):
        means, stds = [], []
        for gname, conds in groups.items():
            vals = get_metric_values(by_cond.get(conds[j], []), "persona_consistency")
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                means.append(mean_val)
                stds.append(std_val)
                plot_values.append(mean_val + std_val)
                plot_values.append(mean_val - std_val)
                has_data = True
            else:
                means.append(0)
                stds.append(0)

        x_positions = x + (j - 1.5) * w
        ax.errorbar(
            x_positions,
            means,
            yerr=stds,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=4,
            markersize=7,
            label=shot,
            alpha=0.9,
        )

    if not has_data:
        plt.close()
        return

    max_abs = max(abs(min(plot_values)), abs(max(plot_values))) if plot_values else 1
    if max_abs == 0:
        max_abs = 1
    max_abs = np.floor(max_abs * 1.2)
    ax.set_ylim(-max_abs, max_abs)

    ax.axhline(y=0, color="red", linestyle="-", linewidth=2)

    ax.set_title("Persona Consistency by Persona Method & Shot Type")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)

    ax.set_ylabel("")
    y_ticks = []
    y_ticklabels = []
    for i in [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]:
        val = i * max_abs
        y_ticks.append(val)
        y_ticklabels.append(f"{val:.1f}")

    y_ticklabels[0] = f"First Half Skewed ({y_ticklabels[0]})"
    y_ticklabels[int(len(y_ticklabels) / 2)] = f"Balanced (0.0)"
    y_ticklabels[len(y_ticklabels) - 1] = f"Second Half Skewed ({y_ticklabels[len(y_ticklabels) - 1]})"

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=9)

    ax.legend(title="Shot type", loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0.08, 0, 0.86, 1])

    path = os.path.join(output_dir, "chart_persona_consistency.png")
    print(path)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Chart: {path}")


def plot_persona_comparison(by_cond, output_dir):
    """
    Plot style strength by persona method and shot type.
    This is different to our grouped bars because we
    care about the comparison between prompting and embedding for the
    same persona. This means we omit the baseline-neutral group entirely
    and create a graph for each persona instead of a single for the metric.
    Additionally, for visual clarity of trends we add a horizontal line
    at the mean for the group.
    """
    if not HAS_PLT:
        return

    comparisons = {
        "Shy": [
            ("baseline_shy_zero", "shy_shy_zero", "Zero-shot"),
            ("baseline_shy_one", "shy_shy_one", "One-shot"),
            ("baseline_shy_few", "shy_shy_few", "Few-shot"),
            ("baseline_shy_cot", "shy_shy_cot", "COT")
        ],
        "Cowboy": [
            ("baseline_cowboy_zero", "cowboy_cowboy_zero", "Zero-shot"),
            ("baseline_cowboy_one", "cowboy_cowboy_one", "One-shot"),
            ("baseline_cowboy_few", "cowboy_cowboy_few", "Few-shot"),
            ("baseline_cowboy_cot", "cowboy_cowboy_cot", "COT")
        ]
    }
    for(name, comparisons) in comparisons.items():
        labels, prompt_vals, embed_vals = [], [], []
        for pc, ec, label in comparisons:
            pv = get_metric_values(by_cond.get(pc, []), "style_strength")
            ev = get_metric_values(by_cond.get(ec, []), "style_strength")
            if pv and ev:
                labels.append(label)
                prompt_vals.append(np.mean(pv))
                embed_vals.append(np.mean(ev))

        if not labels:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        w = 0.35

        prompt_color = "#fc8d62"
        embed_color = "#8da0cb"

        ax.bar(x - w / 2, prompt_vals, w, label="Prompt-based", color=prompt_color, alpha=0.85)
        ax.bar(x + w / 2, embed_vals, w, label="Embed-based", color=embed_color, alpha=0.85)

        _draw_persona_comparison_avg_line(ax, prompt_vals, prompt_color, "Prompt")
        _draw_persona_comparison_avg_line(ax, embed_vals, embed_color, "Embed")

        ax.set_ylabel("Style Strength (keyword density)")
        ax.set_title(f"Prompt vs Embedding: Style Strength (Persona: {name})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        path = os.path.join(output_dir, f"prompt_vs_embed_{name}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Prompt vs Embed chart: {path}")

def _draw_persona_comparison_avg_line(ax, values, color, name):
    """
    Helper function to draw horizontal line on persona comparison chart
    This line should be 20% darker than the color of the bars so that
    when the line is displayed over the bars, it is still visible.
    """
    mean = np.mean(values)
    dark_color = _decrease_brightness(color, factor=0.8)
    print(color)
    ax.axhline(
        y=mean,
        color=dark_color,
        linestyle=":",
        linewidth=2,
        zorder=10
    )
    ax.text(
        0.01,
        mean,
        f"{name} Mean: ({mean:.2f})",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        color=dark_color,
        fontsize=9,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor=dark_color, boxstyle="round,pad=0.2", alpha=0.6),
        zorder=11
    )

def _decrease_brightness(hex_color, factor=0.5):
    """Converts hex string to color, lowers brightness and converts back to hex string"""
    hex_color = hex_color.lstrip("#")

    if len(hex_color) != 6:
        raise ValueError("hex_color must be a 6-digit hex string")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))

    return f"#{r:02X}{g:02X}{b:02X}"



# Main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Path to evaluation results JSON file")
    p.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    args = p.parse_args()

    data, by_cond = load_results(args.input)
    output_dir = args.output_dir or os.path.dirname(args.input)

    print(f"Loaded {len(data)} stories across {len(by_cond)} conditions")

    # Charts
    plot_grouped_bars(by_cond, output_dir)
    plot_persona_consistency(by_cond, output_dir)
    plot_persona_comparison(by_cond, output_dir)

    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()