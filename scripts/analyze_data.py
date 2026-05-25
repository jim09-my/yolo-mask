"""Analyze the long-tail distribution of the Face Mask Detection dataset."""

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path


def count_classes_in_split(label_dir):
    """Count class instances in a split's label files."""
    counter = Counter()
    for txt_file in Path(label_dir).glob("*.txt"):
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    counter[cls_id] += 1
    return counter


def main():
    project_root = Path(__file__).resolve().parent.parent
    labels_base = project_root / "data" / "processed" / "labels"
    
    splits = ["train", "val", "test"]
    all_counts = Counter()
    split_counts = {}
    
    for split in splits:
        label_dir = labels_base / split
        if label_dir.exists():
            c = count_classes_in_split(label_dir)
            split_counts[split] = c
            all_counts.update(c)
    
    # Class names
    class_names = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}
    
    # Prepare data for plotting
    categories = [class_names[i] for i in sorted(all_counts.keys())]
    counts = [all_counts[i] for i in sorted(all_counts.keys())]
    
    # Plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, counts, color=["#2ecc71", "#e74c3c", "#f39c12"])
    
    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f"{count}", ha="center", va="bottom", fontsize=11)
    
    plt.title("Long-Tail Distribution of Mask Wearing Classes", fontsize=14)
    plt.ylabel("Number of Instances", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    
    # Annotate imbalance ratio
    max_count = max(counts)
    min_count = min(counts)
    ratio = max_count / min_count
    plt.figtext(0.15, 0.85, f"Imbalance ratio: {ratio:.1f}:1", 
                bbox=dict(facecolor="white", alpha=0.8))
    
    # Save
    output_dir = project_root / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "longtail_distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "longtail_distribution.pdf", bbox_inches="tight")
    print(f"Figure saved to {output_dir / 'longtail_distribution.png'}")
    
    # Print statistics
    print("\nClass distribution across all splits:")
    for cls_id in sorted(all_counts.keys()):
        print(f"  {class_names[cls_id]:25s}: {all_counts[cls_id]:5d} instances")
    print(f"\nTotal instances: {sum(all_counts.values())}")
    print(f"Imbalance ratio (largest/smallest): {ratio:.2f}")


if __name__ == "__main__":
    main()