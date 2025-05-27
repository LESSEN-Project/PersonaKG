#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import shutil
import pickle
import joblib
import traceback
from collections import defaultdict

from agglomerative_cluster_analysis import PersonaAgglomerativeClusterAnalysis

def run_hyperparameter_search(n_trials=100, output_dir="hyperparameter_results", clustering_mode="separate"):
    """
    Run random hyperparameter search for agglomerative clustering analysis.
    
    Args:
        n_trials: Number of random hyperparameter combinations to try
        output_dir: Directory to save results
        clustering_mode: Run clustering separately per dataset or combined
    """
    
    # Prepare output directory: create if it doesn't exist
    mode_dir = os.path.join(output_dir, clustering_mode)
    os.makedirs(mode_dir, exist_ok=True)

    # Define hyperparameter search space
    vectorization_methods = ["tfidf", "dense", "hybrid"]
    
    if clustering_mode == "separate":
        sample_sizes = [200, 300, 400, 500, 750]
    else:
        sample_sizes = [100, 200, 300, 400]
        
    # Agglomerative clustering specific parameters
    n_clusters_options = [10, 15, 20, 25, 30, 40, 50, 100]
    linkage_options = ['ward', 'complete', 'average', 'single']
    
    # For hybrid vectorization
    tfidf_weights = [0.3, 0.5, 0.7, 0.9] 
    
    # For dimensionality reduction
    pca_components_options = [16, 32, 48, 64] 
    
    results = []
    
    print(f"Running agglomerative clustering hyperparameter search with {n_trials} trials")
    print(f"Results will be saved to: {mode_dir}")
    print(f"Clustering mode: {clustering_mode}")
    print("=" * 80)
    
    for trial in range(n_trials):
        print(f"\n{'='*20} TRIAL {trial + 1}/{n_trials} {'='*20}")
        
        # Sample random hyperparameters
        vectorization = random.choice(vectorization_methods)
        sample_size = random.choice(sample_sizes)
        n_clusters = random.choice(n_clusters_options)
        
        # For ward linkage, only euclidean distance is valid
        linkage = random.choice(linkage_options)
        metric = 'euclidean'
            
        pca_components = random.choice(pca_components_options)

        if vectorization == "hybrid":
            # Sample hybrid-specific parameters
            tfidf_weight = random.choice(tfidf_weights)
        else:
            tfidf_weight = None
        
        trial_params = {
            'trial': trial + 1,
            'vectorization': vectorization,
            'sample_size': sample_size,
            'n_clusters': n_clusters,
            'linkage': linkage,
            'metric': metric,
            'tfidf_weight': tfidf_weight,
            'pca_components': pca_components,
            'clustering_mode': clustering_mode
        }
        
        print(f"Parameters: {trial_params}")
        
        # Create trial directory
        trial_prefix = f"trial_{trial + 1:02d}"
        trial_dir = os.path.join(mode_dir, trial_prefix)
        
        # Clear any existing trial directory
        if os.path.exists(trial_dir):
            for filename in os.listdir(trial_dir):
                file_path = os.path.join(trial_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        else:
            os.makedirs(trial_dir, exist_ok=True)

        # Save trial parameters immediately
        params_file = os.path.join(trial_dir, f"trial_parameters.json")
        with open(params_file, 'w') as f:
            json.dump(trial_params, f, indent=2)
        
        try:
            # Run clustering with current parameters
            trial_results = run_single_trial(
                vectorization=vectorization,
                sample_size=sample_size,
                n_clusters=n_clusters,
                linkage=linkage,
                metric=metric,
                tfidf_weight=tfidf_weight,
                pca_components=pca_components,
                trial_dir=trial_dir,
                clustering_mode=clustering_mode
            )
            
            # Add trial parameters to results
            for result in trial_results:
                result.update(trial_params)
            
            results.extend(trial_results)
            
            print(f"Trial {trial + 1} completed successfully")
            
        except Exception as e:
            print(f"Trial {trial + 1} failed with error: {str(e)}")
            print(traceback.format_exc())
            # Record failed trial
            failed_result = trial_params.copy()
            failed_result.update({
                'Dataset': 'FAILED',
                'Total_Statements': 0,
                'Number_of_Clusters': 0,
                'error': str(e)
            })
            results.append(failed_result)
    
    # Save all results to CSV
    try:
        results_df = pd.DataFrame(results)
        print(f"\nFinal results DataFrame shape: {results_df.shape}")
        
        # Make sure exp_dir exists
        os.makedirs(output_dir, exist_ok=True)
        
        final_results_file = os.path.join(output_dir, "agglomerative_hyperparameter_results.csv")
        results_df.to_csv(final_results_file, index=False)
        print(f"Saved final results to: {final_results_file}")
    except Exception as e:
        print(f"Error saving final results CSV: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"AGGLOMERATIVE CLUSTERING HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {final_results_file}")
    
    # Generate analysis and visualization
    try:
        if len(results_df) > 0:
            print("\nGenerating analysis and visualizations...")
            generate_analysis(results_df, output_dir)
        else:
            print("\nNo results to analyze, skipping analysis generation.")
    except Exception as e:
        print(f"Error generating analysis: {str(e)}")
    
    return results_df


def run_single_trial(vectorization, sample_size, n_clusters, linkage, metric, 
                tfidf_weight=0.5, pca_components=None, trial_dir=None, 
                clustering_mode="separate"):
    """
    Run a single clustering trial with given parameters.
    """
    # Initialize analyzer with current parameters
    analyzer = PersonaAgglomerativeClusterAnalysis(
        sample_size=sample_size,
        split="train",  # Use training split
        vectorization=vectorization,
        mode=clustering_mode,
        pca_components=pca_components,
        tfidf_weight=tfidf_weight,
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric
    )
    
    # Set clustering mode
    print(f"Running trial with clustering mode: {clustering_mode}")
    
    # Temporarily change output directory for this trial
    original_output_dir = analyzer.output_dir
    analyzer.output_dir = trial_dir
    
    # Load personas
    personas_dict = analyzer.load_all_personas()
    
    trial_results = []
    
    # Process each dataset or combined mode
    if clustering_mode == "combined":
        # Extract all statements from all datasets
        all_statements = analyzer.extract_persona_statements(personas_dict)
        print(f"Combined mode: extracted {len(all_statements)} statements from all datasets")
        
        if len(all_statements) > 0:
            # Vectorize statements based on chosen method
            if vectorization == "tfidf":
                vectors, model_info = analyzer.vectorize_statements_tfidf(all_statements, pca_components)
            elif vectorization == "dense":
                vectors, model_info = analyzer.vectorize_statements_dense(all_statements, pca_components)
            elif vectorization == "hybrid":
                vectors, model_info = analyzer.vectorize_statements_hybrid(all_statements, tfidf_weight, pca_components)
            
            # Run clustering
            cluster_labels, _ = analyzer.cluster_with_agglomerative(vectors, n_clusters, linkage, metric)
            
            # Save results
            results = analyzer.save_cluster_results(all_statements, cluster_labels, 'combined', dataset_name='Combined')
            print(f"Combined mode results: {results}")
            
            # Add to trial results, including diversity score
            combined_result = {
                'Dataset': 'Combined',
                'Total_Statements': len(all_statements),
                'Number_of_Clusters': results['n_clusters'],
                'Diversity_Score': getattr(analyzer, 'diversity_score', 0.0)
            }
            trial_results.append(combined_result)
            
    else:  # separate mode
        # Process each dataset separately
        for dataset_name in personas_dict.keys():
            print(f"\nProcessing {dataset_name}...")
            
            # Create dataset directory
            dataset_dir = os.path.join(trial_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Extract statements for this dataset
            statements = analyzer.extract_persona_statements(personas_dict, dataset_name)
            print(f"Extracted {len(statements)} statements from {dataset_name}")
            
            if len(statements) == 0:
                print(f"No statements found for {dataset_name}, skipping")
                continue
                
            # Vectorize statements based on chosen method
            if vectorization == "tfidf":
                vectors, model_info = analyzer.vectorize_statements_tfidf(statements, pca_components)
            elif vectorization == "dense":
                vectors, model_info = analyzer.vectorize_statements_dense(statements, pca_components)
            elif vectorization == "hybrid":
                vectors, model_info = analyzer.vectorize_statements_hybrid(statements, tfidf_weight, pca_components)
            
            # Run clustering
            cluster_labels, _ = analyzer.cluster_with_agglomerative(vectors, n_clusters, linkage, metric)
            
            # Save results
            results = analyzer.save_cluster_results(statements, cluster_labels, dataset_name, dataset_name=dataset_name)
            print(f"{dataset_name} results: {results}")
            
            # Add to trial results, including diversity score
            dataset_result = {
                'Dataset': dataset_name,
                'Total_Statements': len(statements),
                'Number_of_Clusters': results['n_clusters'],
                'Diversity_Score': getattr(analyzer, 'diversity_score', 0.0)
            }
            trial_results.append(dataset_result)
    
    # Save trial summary
    save_trial_summary(
        trial_dir, 
        trial_results,
        vectorization, 
        sample_size, 
        n_clusters,
        linkage,
        metric,
        tfidf_weight,
        pca_components,
        clustering_mode
    )
    
    # Restore original output directory
    analyzer.output_dir = original_output_dir
    
    return trial_results


def save_trial_summary(trial_dir, trial_results, vectorization, sample_size, n_clusters, linkage, metric,
                      tfidf_weight=None, pca_components=None, clustering_mode="separate"):
    """
    Save a summary file for the trial with parameters and results.
    """
    # Format the summary
    summary = []
    summary.append("="*50)
    summary.append(f"AGGLOMERATIVE CLUSTERING TRIAL SUMMARY")
    summary.append("="*50)
    summary.append(f"\nTrial Directory: {trial_dir}")
    summary.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("\nHYPERPARAMETERS:")
    summary.append(f"Clustering Mode: {clustering_mode}")
    summary.append(f"Vectorization Method: {vectorization}")
    summary.append(f"Sample Size: {sample_size}")
    summary.append(f"Number of Clusters: {n_clusters}")
    summary.append(f"Linkage: {linkage}")
    summary.append(f"Distance Metric: {metric}")
    
    if vectorization == "hybrid":
        summary.append(f"TF-IDF Weight: {tfidf_weight}")
    
    if pca_components is not None:
        summary.append(f"PCA Components: {pca_components}")
    
    summary.append("\nRESULTS:")
    for result in trial_results:
        dataset = result['Dataset']
        statements = result['Total_Statements']
        clusters = result['Number_of_Clusters']
        
        summary.append(f"\n{dataset}:")
        summary.append(f"  - Total Statements: {statements}")
        summary.append(f"  - Number of Clusters: {clusters}")
        if 'Diversity_Score' in result:
            summary.append(f"  - Diversity Score: {result['Diversity_Score']:.4f}")
    
    # Write summary to file
    summary_path = os.path.join(trial_dir, "trial_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary))
    
    # Also save as CSV for easier analysis
    csv_data = []
    for result in trial_results:
        record = {
            'Dataset': result['Dataset'],
            'Total_Statements': result['Total_Statements'],
            'Number_of_Clusters': result['Number_of_Clusters'],
            'Diversity_Score': result.get('Diversity_Score', 0.0)
        }
        csv_data.append(record)
    
    results_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(trial_dir, "trial_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    print(f"\nSaved trial summary to {summary_path}")
    print(f"Saved trial results to {csv_path}")


def generate_analysis(results_df, exp_dir):
    """
    Generate analysis and visualizations of hyperparameter search results.
    """
    # Filter out failed trials
    successful_trials = results_df[results_df['Dataset'] != 'FAILED'].copy()
    
    if len(successful_trials) == 0:
        print("No successful trials to analyze")
        return
    
    # Create analysis directory
    analysis_dir = os.path.join(exp_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # ----- Performance by Vectorization Method -----
    plt.figure(figsize=(10, 6))
    
    # Check if the column exists in the dataframe
    if 'Diversity_Score' in successful_trials.columns:
        primary_metric = 'Diversity_Score'
        primary_title = 'Diversity Score'
    else:
        print("Warning: Diversity_Score column not found in results")
        return
        
    # Secondary metric for additional analysis
    if 'Number_of_Clusters' in successful_trials.columns:
        secondary_metric = 'Number_of_Clusters'
        secondary_title = 'Number of Clusters'
    else:
        secondary_metric = None
    
    # ----- Diversity Score by Vectorization Method -----
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='vectorization', y=primary_metric, data=successful_trials)
    plt.title(f'Distribution of {primary_title} by Vectorization Method')
    plt.xlabel('Vectorization Method')
    plt.ylabel(primary_title)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'vectorization_diversity.png'))
    plt.close()
    
    # ----- Diversity Score by Linkage Type -----
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='linkage', y=primary_metric, data=successful_trials)
    plt.title(f'Distribution of {primary_title} by Linkage Type')
    plt.xlabel('Linkage')
    plt.ylabel(primary_title)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'linkage_diversity.png'))
    plt.close()
    
    # ----- Diversity Score by Distance Metric -----
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='metric', y=primary_metric, data=successful_trials)
    plt.title(f'Distribution of {primary_title} by Distance Metric')
    plt.xlabel('Distance Metric')
    plt.ylabel(primary_title)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'metric_diversity.png'))
    plt.close()
    
    # ----- Diversity Score by n_clusters -----
    plt.figure(figsize=(12, 6))
    n_clusters_groups = successful_trials.groupby('n_clusters')[primary_metric].mean().reset_index()
    sns.barplot(x='n_clusters', y=primary_metric, data=n_clusters_groups)
    plt.title(f'Average {primary_title} by Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel(f'Average {primary_title}')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'n_clusters_diversity.png'))
    plt.close()
    
    # ----- Also generate plots for Number of Clusters if available -----
    if secondary_metric:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='vectorization', y=secondary_metric, data=successful_trials)
        plt.title(f'Distribution of {secondary_title} by Vectorization Method')
        plt.xlabel('Vectorization Method')
        plt.ylabel(secondary_title)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'vectorization_clusters.png'))
        plt.close()
    
    # ----- Correlation Matrix -----
    # Select numeric columns for correlation
    numeric_cols = ['n_clusters', 'sample_size', 'pca_components']
    
    if 'tfidf_weight' in successful_trials.columns:
        # Fill NaN values for trials that didn't use hybrid vectorization
        successful_trials['tfidf_weight'] = successful_trials['tfidf_weight'].fillna(0)
        numeric_cols.append('tfidf_weight')
    
    # Add the primary and secondary metrics
    numeric_cols.append(primary_metric)
    if secondary_metric:
        numeric_cols.append(secondary_metric)
    
    # Compute correlation matrix
    corr_matrix = successful_trials[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Hyperparameters')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'correlation_matrix.png'))
    plt.close()
    
    # ----- Dataset-specific Analysis -----
    if 'Dataset' in successful_trials.columns:
        # Exclude 'Combined' dataset for this analysis if it exists
        dataset_df = successful_trials[successful_trials['Dataset'] != 'Combined']
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Dataset', y=primary_metric, data=dataset_df)
        plt.title(f'Distribution of {primary_title} by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel(primary_title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'dataset_diversity.png'))
        plt.close()
        
        if secondary_metric:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='Dataset', y=secondary_metric, data=dataset_df)
            plt.title(f'Distribution of {secondary_title} by Dataset')
            plt.xlabel('Dataset')
            plt.ylabel(secondary_title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'dataset_clusters.png'))
            plt.close()
    
    # ----- Generate Summary Stats -----
    # First for the primary metric (diversity score)
    summary_stats = successful_trials.groupby(['vectorization', 'linkage', 'metric'])[primary_metric].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    summary_stats = summary_stats.sort_values('mean', ascending=False)
    
    summary_file = os.path.join(analysis_dir, 'diversity_summary_stats.csv')
    summary_stats.to_csv(summary_file, index=False)
    
    # ----- Best Configurations -----
    best_configs = successful_trials.groupby(['vectorization', 'linkage', 'metric', 'n_clusters'])[primary_metric].mean().reset_index()
    best_configs = best_configs.sort_values(primary_metric, ascending=False).head(10)
    
    best_file = os.path.join(analysis_dir, 'best_configurations.csv')
    best_configs.to_csv(best_file, index=False)
    
    print("Analysis complete. Results saved to:", analysis_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyperparameter search for agglomerative clustering")
    parser.add_argument('--trials', type=int, default=100, help='Number of trials to run')
    parser.add_argument('--output', type=str, default='clustering/agglomerative_hyperparameter_results', help='Output directory for results')
    parser.add_argument('--mode', type=str, default='separate', choices=['separate', 'combined'], 
                        help='Run clustering separately per dataset or combined')
    
    args = parser.parse_args()
    
    # Run hyperparameter search
    results = run_hyperparameter_search(
        n_trials=args.trials,
        output_dir=args.output,
        clustering_mode=args.mode
    )
