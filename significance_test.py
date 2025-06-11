import argparse
import json
import os
from datetime import datetime

from rouge_score import rouge_scorer
from scipy.stats import ttest_rel

def load_rouge_scores(file_path: str, metric: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["completed_samples"]
    scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
    scores = []
    for sample in samples:
        target = sample["target"]
        prediction = sample["prediction"]
        score = scorer.score(target, prediction)[metric].fmeasure
        scores.append(score)
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_a", help="First checkpoint file")
    parser.add_argument("file_b", help="Second checkpoint file")
    parser.add_argument("--metric", default="rougeL", choices=["rouge1", "rouge2", "rougeL"], help="ROUGE metric to use")
    parser.add_argument("--output", default=None, help="Path to save the t-test results text file")
    args = parser.parse_args()

    scores_a = load_rouge_scores(args.file_a, args.metric)
    scores_b = load_rouge_scores(args.file_b, args.metric)

    if len(scores_a) != len(scores_b):
        raise ValueError("The two checkpoint files contain different numbers of samples.")

    t_stat, p_val = ttest_rel(scores_a, scores_b)

    mean_a = sum(scores_a) / len(scores_a)
    mean_b = sum(scores_b) / len(scores_b)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or os.path.join("results", f"t_test_{timestamp}.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        f"Checkpoint A: {os.path.basename(args.file_a)}",
        f"Checkpoint B: {os.path.basename(args.file_b)}",
        f"Metric: {args.metric}",
        f"Samples: {len(scores_a)}",
        f"Mean {args.metric} A: {mean_a:.6f}",
        f"Mean {args.metric} B: {mean_b:.6f}",
        f"t-statistic: {t_stat:.6f}",
        f"p-value: {p_val:.6f}",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"t-test results saved to {output_path}")

if __name__ == "__main__":
    main()
