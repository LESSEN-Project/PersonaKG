#!/usr/bin/env python
import os
import pandas as pd
from cluster_analysis import PersonaClusterAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

def run_comparison(sample_size=500, split="train"):
    """
    Run clustering analysis with all three vectorization methods and compare results.
    
    Args:
        sample_size: Number of samples to use (default: 500)
        split: Dataset split to use (default: "train")
    """
    vectorization_methods = ["tfidf", "dense", "hybrid"]
    results = []
    
    print(f"Running clustering comparison with {sample_size} samples")
    print(f"Split: {split}")
    print("=" * 60)
    
    for vectorization in vectorization_methods:
        print(f"\n\nRUNNING CLUSTERING WITH {vectorization.upper()} VECTORIZATION")
        print("-" * 60)
        
        # Initialize analyzer with current vectorization method
        analyzer = PersonaClusterAnalysis(
            sample_size=sample_size,
            split=split,
            vectorization=vectorization
        )
        
        # Run clustering in separate mode (per dataset)
        analyzer.run(mode="separate")
        
        # Collect results by reading the summary files
        results_dir = os.path.join(analyzer.output_dir)
        dataset_results = []
        
        for dataset_name in analyzer.datasets:
            summary_file = os.path.join(
                results_dir, 
                f"dataset_{dataset_name}_{vectorization}", 
                "clustering_summary.txt"
            )
            
            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Extract key metrics
                    try:
                        total_statements = int(content.split("Total statements: ")[1].split("\n")[0])
                        num_clusters = int(content.split("Number of clusters: ")[1].split("\n")[0])
                        noise_line = content.split("Number of noise points: ")[1].split("\n")[0]
                        num_noise = int(noise_line.split(" ")[0])
                        noise_percentage = float(noise_line.split("(")[1].split("%")[0])
                        
                        dataset_results.append({
                            "Dataset": dataset_name,
                            "Vectorization": vectorization,
                            "Total Statements": total_statements,
                            "Number of Clusters": num_clusters,
                            "Noise Points": num_noise,
                            "Noise Percentage": noise_percentage
                        })
                    except:
                        print(f"Could not parse results for {dataset_name} with {vectorization}")
        
        results.extend(dataset_results)
    
    # Create a DataFrame with all results
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = f"clustering_comparison_{sample_size}_samples.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print(f"CLUSTERING COMPARISON SUMMARY ({sample_size} samples)")
    print("=" * 80)
    
    # Pivot table for better comparison
    pivot_clusters = results_df.pivot_table(
        index='Dataset', 
        columns='Vectorization', 
        values='Number of Clusters',
        aggfunc='first'
    )
    
    pivot_noise = results_df.pivot_table(
        index='Dataset', 
        columns='Vectorization', 
        values='Noise Percentage',
        aggfunc='first'
    )
    
    print("\nNumber of Clusters per Dataset and Vectorization Method:")
    print(pivot_clusters)
    
    print("\nNoise Percentage per Dataset and Vectorization Method:")
    print(pivot_noise)
    
    # Calculate averages
    avg_clusters = results_df.groupby('Vectorization')['Number of Clusters'].mean()
    avg_noise = results_df.groupby('Vectorization')['Noise Percentage'].mean()
    
    print("\nAverage Number of Clusters across Datasets:")
    print(avg_clusters)
    
    print("\nAverage Noise Percentage across Datasets:")
    print(avg_noise)
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare clustering with different vectorization methods")
    parser.add_argument("--sample_size", type=int, default=500, 
                        help="Sample size for each dataset (default: 500)")
    parser.add_argument("--split", type=str, default="train", 
                        choices=["train", "validation", "test"],
                        help="Dataset split to use (default: train)")
    
    args = parser.parse_args()
    
    run_comparison(sample_size=args.sample_size, split=args.split)
