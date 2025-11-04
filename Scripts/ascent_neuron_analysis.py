"""ASCENT Neuron Analysis Script
Generates salient, dormant, and random neuron sets from localized neuron activations.
"""

import os
import json
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def analyze_neuron_activations(file_path, use_ig_gold=True):
    """
    Identify neurons with highest and lowest activation scores per layer.
    Args:
        file_path: Path to neuron localization JSONL file.
        use_ig_gold: If True, use 'ig_gold'; otherwise, use 'ig_pred'.
    Returns:
        (layer_highest_neurons, layer_lowest_neurons): dicts of sets per layer.
    """
    layer_highest_neurons, layer_lowest_neurons = defaultdict(set), defaultdict(set)

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            for inner_list in data:
                source = inner_list[1].get("ig_gold", []) if use_ig_gold else inner_list[0].get("ig_pred", [])
                layer_max = defaultdict(lambda: (-1, float('-inf')))
                layer_min = defaultdict(lambda: (-1, float('inf')))

                for layer, neuron, score in source:
                    if score > layer_max[layer][1]:
                        layer_max[layer] = (neuron, score)
                    if score < layer_min[layer][1]:
                        layer_min[layer] = (neuron, score)

                for layer, (neuron, _) in layer_max.items():
                    layer_highest_neurons[layer].add(neuron)
                for layer, (neuron, _) in layer_min.items():
                    layer_lowest_neurons[layer].add(neuron)

    return layer_highest_neurons, layer_lowest_neurons


def calculate_random_neurons(layer_highest_neurons, layer_lowest_neurons, file_path, use_ig_gold=True):
    """
    Select random neurons per layer with count equal to average of salient and dormant counts.
    """
    layer_all_neurons = defaultdict(set)

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            for inner_list in data:
                source = inner_list[1].get("ig_gold", []) if use_ig_gold else inner_list[0].get("ig_pred", [])
                for layer, neuron, _ in source:
                    layer_all_neurons[layer].add(neuron)

    random_neurons = {}
    for layer in sorted(set(layer_highest_neurons.keys()).union(layer_lowest_neurons.keys())):
        high_count = len(layer_highest_neurons.get(layer, []))
        low_count = len(layer_lowest_neurons.get(layer, []))
        avg_count = max(1, (high_count + low_count) // 2)

        all_neurons = list(layer_all_neurons[layer])
        random_selection = random.sample(all_neurons, min(avg_count, len(all_neurons)))
        random_neurons[layer] = random_selection

    return random_neurons, layer_all_neurons


def save_neurons_to_json(neurons_dict, output_path):
    """Save neuron indices per layer as JSON."""
    with open(output_path, 'w') as f:
        json.dump({int(k): list(v) for k, v in neurons_dict.items()}, f, indent=4)


def save_statistics(layer_highest, layer_lowest, random_neurons, layer_all, output_path):
    """Save neuron statistics per layer to Excel."""
    rows = []
    layers = sorted(set(layer_all.keys()).union(layer_highest.keys(), layer_lowest.keys()))
    for layer in layers:
        rows.append([
            layer,
            len(layer_all.get(layer, [])),
            len(layer_highest.get(layer, [])),
            len(layer_lowest.get(layer, [])),
            len(random_neurons.get(layer, [])),
        ])

    df = pd.DataFrame(rows, columns=["Layer", "Activated", "Salient", "Dormant", "Random"])
    df.to_excel(output_path, index=False)


def plot_distribution(layer_highest, layer_lowest, dataset_name, output_path):
    """Plot neuron distributions for salient and dormant neurons."""
    layers = sorted(set(layer_highest.keys()).union(layer_lowest.keys()))
    highest_counts = [len(layer_highest[layer]) for layer in layers]
    lowest_counts = [len(layer_lowest[layer]) for layer in layers]

    bar_width = 0.4
    x = range(len(layers))

    plt.figure(figsize=(8, 4))
    plt.bar([p - bar_width / 2 for p in x], highest_counts, width=bar_width, label='Salient')
    plt.bar([p + bar_width / 2 for p in x], lowest_counts, width=bar_width, label='Dormant')
    plt.xlabel('Layer')
    plt.ylabel('Neuron Count')
    plt.title(f'Salient vs Dormant Neurons – {dataset_name.upper()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ASCENT Neuron Analysis")
    parser.add_argument("--input_file", type=str, required=True, help="Path to .jsonl neuron localization file")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset identifier (lap14, res14, mams)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store results")
    parser.add_argument("--use_ig_gold", action="store_true", help="Use IG-Gold activations (default: IG-Pred)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = args.dataset_name.lower()

    salient_path = os.path.join(args.output_dir, f"neurons_{base_name}_salient.json")
    dormant_path = os.path.join(args.output_dir, f"neurons_{base_name}_dormant.json")
    random_path = os.path.join(args.output_dir, f"neurons_{base_name}_random.json")
    stats_path = os.path.join(args.output_dir, f"neurons_{base_name}_stats.xlsx")
    plot_path = os.path.join(args.output_dir, f"neurons_{base_name}_distribution.png")

    print(f"Analyzing neuron activations for {args.dataset_name.upper()} ...")

    layer_highest, layer_lowest = analyze_neuron_activations(args.input_file, args.use_ig_gold)
    random_neurons, layer_all = calculate_random_neurons(layer_highest, layer_lowest, args.input_file, args.use_ig_gold)

    save_neurons_to_json(layer_highest, salient_path)
    save_neurons_to_json(layer_lowest, dormant_path)
    save_neurons_to_json(random_neurons, random_path)
    save_statistics(layer_highest, layer_lowest, random_neurons, layer_all, stats_path)
    plot_distribution(layer_highest, layer_lowest, args.dataset_name, plot_path)

    print(f"Neuron analysis completed for {args.dataset_name.upper()}. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
