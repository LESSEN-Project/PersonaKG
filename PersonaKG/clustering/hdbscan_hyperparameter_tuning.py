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
from collections import defaultdict

from hdbscan_cluster_analysis import PersonaClusterAnalysis

def run_hyperparameter_search(n_trials=100, output_dir="hyperparameter_search", clustering_mode="separate"):
    """
    Run random hyperparameter search for clustering analysis.
    
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
        sample_sizes = [100, 200, 300, 400, 500]
    # Define min_cluster_size as a ratio of sample_size instead of absolute values
    min_cluster_size_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    if clustering_mode == "combined":
        # For combined mode, we can use slightly larger ratios
        min_cluster_size_ratios = [0.02, 0.03, 0.04, 0.05, 0.06, 0.1]
    min_samples_ratios = [0.5, 0.7, 1.0]  
    tfidf_weights = [0.3, 0.5, 0.7, 0.9] 
    pca_components_options = [16, 32, 48] 
    
    results = []
    
    print(f"Running hyperparameter search with {n_trials} trials")
    print(f"Results will be saved to: {mode_dir}")
    print(f"Clustering mode: {clustering_mode}")
    print("=" * 80)
    
    for trial in range(n_trials):
        print(f"\n{'='*20} TRIAL {trial + 1}/{n_trials} {'='*20}")
        
        # Sample random hyperparameters
        vectorization = random.choice(vectorization_methods)
        sample_size = random.choice(sample_sizes)
        min_cluster_size_ratio = random.choice(min_cluster_size_ratios)
        
        # Calculate actual min_cluster_size based on sample_size and the chosen ratio
        min_cluster_size = max(5, int(sample_size * min_cluster_size_ratio))
        min_samples_ratio = random.choice(min_samples_ratios)
        min_samples = max(1, int(min_cluster_size * min_samples_ratio))
        
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
            'min_cluster_size_ratio': min_cluster_size_ratio,
            'min_cluster_size': min_cluster_size,  # Also store the actual computed value
            'min_samples': min_samples,
            'min_samples_ratio': min_samples_ratio,
            'tfidf_weight': tfidf_weight,
            'pca_components': pca_components,
            'clustering_mode': clustering_mode
        }
        
        print(f"Parameters: {trial_params}")
        
        # Save trial parameters as a separate JSON file next to the folders
        trial_prefix = f"trial_{trial + 1:02d}"
        trial_dir = os.path.join(mode_dir, trial_prefix)
        os.makedirs(trial_dir, exist_ok=True)
        if os.path.exists(trial_dir):
            for filename in os.listdir(trial_dir):
                file_path = os.path.join(trial_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        else:
            os.makedirs(trial_dir, exist_ok=True)

        params_file = os.path.join(trial_dir, f"{trial_prefix}_parameters.json")
        with open(params_file, 'w') as f:
            json.dump(trial_params, f, indent=2)
        
        try:
            # Run clustering with current parameters
            trial_results = run_single_trial(
                vectorization=vectorization,
                sample_size=sample_size,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
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
            # Record failed trial
            failed_result = trial_params.copy()
            failed_result.update({
                'Dataset': 'FAILED',
                'Total_Statements': 0,
                'Number_of_Clusters': 0,
                'Noise_Points': 0,
                'Noise_Percentage': 100.0,
                'error': str(e)
            })
            results.append(failed_result)
    
    # Save all results to CSV
    try:
        results_df = pd.DataFrame(results)
        print(f"\nFinal results DataFrame shape: {results_df.shape}")
        
        # Make sure exp_dir exists
        os.makedirs(mode_dir, exist_ok=True)
        
        final_results_file = os.path.join(mode_dir, "hyperparameter_search_results.csv")
        results_df.to_csv(final_results_file, index=False)
        print(f"Saved final results to: {final_results_file}")
    except Exception as e:
        print(f"Error saving final results CSV: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {final_results_file}")
    
    # Generate analysis and visualization
    try:
        if len(results_df) > 0:
            print("\nGenerating analysis and visualizations...")
            generate_analysis(results_df, mode_dir)
        else:
            print("\nNo results to analyze, skipping analysis generation.")
    except Exception as e:
        print(f"Error generating analysis: {str(e)}")
    
    return results_df

def run_single_trial(vectorization, sample_size, min_cluster_size, min_samples, 
                tfidf_weight=0.5, pca_components=None, trial_dir=None, 
                clustering_mode="separate"):
    """
    Run a single clustering trial with given parameters.
    """
    # Initialize analyzer with current parameters including the new hyperparameters
    analyzer = PersonaClusterAnalysis(
        sample_size=sample_size,
        split="train",  # Use training split as requested
        vectorization=vectorization,
        mode=clustering_mode,  # Pass the mode to ensure correct directory structure
        pca_components=pca_components,
        tfidf_weight=tfidf_weight,
        min_cluster_size=min_cluster_size,
        min_samples_ratio=min_samples/min_cluster_size if min_cluster_size > 0 else 0.5
    )
    
    # Set clustering mode
    print(f"Running trial with clustering mode: {clustering_mode}")
    
    # Temporarily change output directory for this trial
    original_output_dir = analyzer.output_dir
    analyzer.output_dir = trial_dir
    
    # Load personas
    personas_dict = analyzer.persona_dataset.load_all_personas()
    
    trial_results = []
    
    if clustering_mode == "combined":
        # Combined mode: cluster all datasets together
        print("\nRunning combined clustering across all datasets...")
        
        # Extract statements from all datasets
        all_statements = analyzer.extract_persona_statements(personas_dict)
        all_statements = analyzer.filter_similar_statements(all_statements)
        
        if len(all_statements) == 0:
            print("No statements found across all datasets")
            return trial_results
            
        print(f"Total statements across all datasets: {len(all_statements)}")
        
        try:
            # Create vectors with appropriate parameters
            if analyzer.vectorization == "tfidf":
                vectors, model_info = analyzer.vectorize_statements_tfidf(
                    all_statements,
                    pca_components=pca_components
                )
            elif analyzer.vectorization == "dense":
                vectors, model_info = analyzer.vectorize_statements_dense(
                    all_statements,
                    pca_components=pca_components
                )
            else:  # hybrid
                vectors, hybrid_info = analyzer.vectorize_statements_hybrid(
                    all_statements,
                    tfidf_weight=tfidf_weight,
                    pca_components=pca_components
                )
            
            # Perform clustering
            cluster_labels, clusterer, cluster_persistence = analyzer.cluster_with_hdbscan(
                vectors, 
                min_cluster_size=min_cluster_size, 
                min_samples=min_samples
            )
            
            analyzer.save_cluster_results(
                all_statements, 
                cluster_labels, 
                "clusters",
                cluster_persistence=cluster_persistence,
                dataset_name="combined"
            )
            
            # Calculate metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            noise_percentage = (n_noise / len(cluster_labels)) * 100
            
            # Calculate dataset distribution in clusters
            dataset_counts = defaultdict(int)
            for stmt in all_statements:
                dataset_counts[stmt['dataset']] += 1
            
            # Calculate average cluster size
            cluster_sizes = []
            for cluster_id in set(cluster_labels):
                if cluster_id != -1:
                    cluster_size = sum(1 for label in cluster_labels if label == cluster_id)
                    cluster_sizes.append(cluster_size)
            
            avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
            
            # Calculate silhouette score
            try:
                from sklearn.metrics import silhouette_score
                if n_clusters > 1 and len(set(cluster_labels)) > 1:
                    non_noise_mask = cluster_labels != -1
                    if np.sum(non_noise_mask) > 1:
                        silhouette = silhouette_score(vectors[non_noise_mask], cluster_labels[non_noise_mask])
                    else:
                        silhouette = -1
                else:
                    silhouette = -1
            except:
                silhouette = -1
            
            result = {
                'Dataset': 'Combined',
                'Total_Statements': len(all_statements),
                'Number_of_Clusters': n_clusters,
                'Noise_Points': n_noise,
                'Noise_Percentage': noise_percentage,
                'Avg_Cluster_Size': avg_cluster_size,
                'Silhouette_Score': silhouette,
                'Vector_Dimension': vectors.shape[1] if vectors is not None else 0,
                'Dataset_Distribution': str(dict(dataset_counts))
            }
            
            trial_results.append(result)
            
        except Exception as e:
            print(f"Error in combined clustering: {str(e)}")
            import traceback
            traceback.print_exc()
            result = {
                'Dataset': 'Combined',
                'Total_Statements': len(all_statements),
                'Number_of_Clusters': 0,
                'Noise_Points': len(all_statements),
                'Noise_Percentage': 100.0,
                'Avg_Cluster_Size': 0,
                'Silhouette_Score': -1,
                'Vector_Dimension': 0,
                'error': str(e)
            }
            trial_results.append(result)
            
    else:
        # Separate mode: cluster each dataset independently
        # Run clustering for each dataset separately
        for dataset_name in analyzer.datasets:
            if dataset_name not in personas_dict or not personas_dict[dataset_name]:
                print(f"Skipping {dataset_name} - no data loaded")
                continue
            
            print(f"\nProcessing {dataset_name}...")
            
            try:
                # Get statements for this dataset
                statements = analyzer.extract_persona_statements(personas_dict, dataset_filter=dataset_name)
                statements = analyzer.filter_similar_statements(statements)

                
                if len(statements) == 0:
                    print(f"No statements found for {dataset_name}")
                    continue
                    
                # Create vectors with appropriate parameters
                if analyzer.vectorization == "tfidf":
                    vectors, model_info = analyzer.vectorize_statements_tfidf(
                        statements,
                        pca_components=pca_components
                    )
                elif analyzer.vectorization == "dense":
                    vectors, model_info = analyzer.vectorize_statements_dense(
                        statements,
                        pca_components=pca_components
                    )
                else:  # hybrid
                    vectors, hybrid_info = analyzer.vectorize_statements_hybrid(
                        statements,
                        tfidf_weight=tfidf_weight,
                        pca_components=pca_components
                    )
                
                # Perform clustering with specified parameters
                cluster_labels, clusterer, cluster_persistence = analyzer.cluster_with_hdbscan(
                    vectors, 
                    min_cluster_size=min_cluster_size, 
                    min_samples=min_samples
                )
                
                # Use the mode directory as base and add trial prefix for saving results
                analyzer.output_dir = os.path.join(trial_dir, dataset_name)
                
                # Save clustering results to the dataset-specific directory
                analyzer.save_cluster_results(
                    statements, 
                    cluster_labels, 
                    "clusters",
                    cluster_persistence=cluster_persistence,
                    dataset_name=dataset_name
                )
                
                # Restore original output directory
                analyzer.output_dir = original_output_dir
                
                # Calculate metrics
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                noise_percentage = (n_noise / len(cluster_labels)) * 100
                
                # Calculate average cluster size (excluding noise)
                cluster_sizes = []
                for cluster_id in set(cluster_labels):
                    if cluster_id != -1:
                        cluster_size = sum(1 for label in cluster_labels if label == cluster_id)
                        cluster_sizes.append(cluster_size)
                
                avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
                
                # Calculate silhouette score if possible
                try:
                    from sklearn.metrics import silhouette_score
                    if n_clusters > 1 and len(set(cluster_labels)) > 1:
                        # Only calculate for non-noise points
                        non_noise_mask = cluster_labels != -1
                        if np.sum(non_noise_mask) > 1:
                            silhouette = silhouette_score(vectors[non_noise_mask], cluster_labels[non_noise_mask])
                        else:
                            silhouette = -1
                    else:
                        silhouette = -1
                except:
                    silhouette = -1
                
                result = {
                    'Dataset': dataset_name,
                    'Total_Statements': len(statements),
                    'Number_of_Clusters': n_clusters,
                    'Noise_Points': n_noise,
                    'Noise_Percentage': noise_percentage,
                    'Avg_Cluster_Size': avg_cluster_size,
                    'Silhouette_Score': silhouette,
                    'Vector_Dimension': vectors.shape[1] if vectors is not None else 0
                }
                
                trial_results.append(result)
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                result = {
                    'Dataset': dataset_name,
                    'Total_Statements': 0,
                    'Number_of_Clusters': 0,
                    'Noise_Points': 0,
                    'Noise_Percentage': 100.0,
                    'Avg_Cluster_Size': 0,
                    'Silhouette_Score': -1,
                    'Vector_Dimension': 0,
                    'error': str(e)
                }
                trial_results.append(result)
        
    # Save trial summary file in the trial directory
    try:
        print(f"\nSaving trial summary to {trial_dir}...")
        save_trial_summary(trial_dir, trial_results, vectorization, sample_size, min_cluster_size, min_samples,
                         tfidf_weight=tfidf_weight, pca_components=pca_components, clustering_mode=clustering_mode)
        print(f"Successfully saved trial summary.")
    except Exception as e:
        print(f"Error saving trial summary: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return trial_results

def save_trial_summary(trial_dir, trial_results, vectorization, sample_size, min_cluster_size, min_samples,
                   tfidf_weight=0.5, pca_components=None, clustering_mode="separate"):
    """
    Save a summary file for the trial with parameters and results.
    """
    # Make sure trial_dir exists
    if not os.path.exists(trial_dir):
        print(f"Creating directory: {trial_dir}")
        os.makedirs(trial_dir, exist_ok=True)
    
    print(f"Trial directory: {trial_dir}")
    print(f"Number of trial results: {len(trial_results)}")
    
    summary_file = os.path.join(trial_dir, "trial_summary.txt")
    print(f"Summary file path: {summary_file}")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"HYPERPARAMETER TUNING TRIAL SUMMARY\n")
        f.write(f"{'='*60}\n\n")
        
        # Write parameters
        f.write("PARAMETERS:\n")
        f.write(f"  Vectorization: {vectorization}\n")
        f.write(f"  Sample Size: {sample_size}\n")
        f.write(f"  Min Cluster Size: {min_cluster_size}\n")
        f.write(f"  Min Samples: {min_samples}\n")
        f.write(f"  Min Samples Ratio: {min_samples/min_cluster_size if min_cluster_size > 0 else 0.5:.2f}\n")
        
        # Write PCA parameters for all vectorization methods
        f.write(f"  PCA Components: {pca_components if pca_components is not None else 'None'}\n")
        f.write(f"  Clustering Mode: {clustering_mode}\n")
        
        # Write hybrid-specific parameters if applicable
        if vectorization == "hybrid":
            f.write(f"  TF-IDF Weight: {tfidf_weight}\n")
        f.write("\n")
        
        # Write results summary
        f.write("RESULTS SUMMARY:\n")
        
        if not trial_results:
            f.write("  No results available for this trial.\n")
        else:
            # Calculate overall metrics
            total_statements = sum(r['Total_Statements'] for r in trial_results)
            total_clusters = sum(r['Number_of_Clusters'] for r in trial_results)
            avg_clusters = sum(r['Number_of_Clusters'] for r in trial_results) / len(trial_results) if trial_results else 0
            avg_noise = sum(r['Noise_Percentage'] for r in trial_results) / len(trial_results) if trial_results else 0
            
            f.write(f"  Datasets analyzed: {len(trial_results)}\n")
            f.write(f"  Total statements: {total_statements}\n")
            f.write(f"  Total clusters: {total_clusters}\n")
            f.write(f"  Average clusters per dataset: {avg_clusters:.1f}\n")
            f.write(f"  Average noise percentage: {avg_noise:.1f}%\n\n")
            
            # Table of results by dataset
            f.write("DATASET RESULTS:\n")
            f.write(f"{'Dataset':<15} {'Statements':<10} {'Clusters':<10} {'Noise %':<10}\n")
            f.write("-" * 50 + "\n")
            
            for result in sorted(trial_results, key=lambda x: x['Dataset']):
                f.write(f"{result['Dataset']:<15} {result['Total_Statements']:<10} {result['Number_of_Clusters']:<10} {result['Noise_Percentage']:<10.1f}\n")
    
    # Also save as CSV for easier analysis
    csv_file = os.path.join(trial_dir, "trial_results.csv")
    
    try:
        df = pd.DataFrame(trial_results)
        print(f"DataFrame shape: {df.shape}")
        df.to_csv(csv_file, index=False)
        print(f"Trial CSV saved to {csv_file}")
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")
    
    print(f"Trial summary saved to {summary_file}")

def generate_analysis(results_df, exp_dir):
    """
    Generate analysis and visualizations of hyperparameter search results.
    """
    print(f"\nGenerating analysis for {len(results_df)} results, saving to {exp_dir}")
    """
    Generate analysis and visualizations of hyperparameter search results.
    """
    print("\nGenerating analysis and visualizations...")
    
    # Filter out failed trials for analysis
    valid_results = results_df[results_df['Dataset'] != 'FAILED'].copy()
    
    if len(valid_results) == 0:
        print("No valid results to analyze!")
        return
    
    # Calculate aggregate metrics per trial
    trial_metrics = []
    for trial in valid_results['trial'].unique():
        trial_data = valid_results[valid_results['trial'] == trial]
        
        if len(trial_data) == 0:
            continue
            
        # Get trial parameters (should be same for all rows in this trial)
        trial_params = trial_data.iloc[0][['trial', 'vectorization', 'sample_size', 
                                          'min_cluster_size', 'min_samples', 'min_samples_ratio']]
        
        # Calculate aggregate metrics
        metrics = {
            'avg_clusters': trial_data['Number_of_Clusters'].mean(),
            'avg_noise_percentage': trial_data['Noise_Percentage'].mean(),
            'avg_silhouette': trial_data['Silhouette_Score'].mean(),
            'avg_cluster_size': trial_data['Avg_Cluster_Size'].mean(),
            'total_datasets': len(trial_data),
            'successful_datasets': len(trial_data[trial_data['Number_of_Clusters'] > 0])
        }
        
        # Combine parameters and metrics
        trial_summary = {**trial_params.to_dict(), **metrics}
        trial_metrics.append(trial_summary)
    
    trial_metrics_df = pd.DataFrame(trial_metrics)
    
    # Save trial-level metrics
    try:
        os.makedirs(exp_dir, exist_ok=True)  # Ensure directory exists
        trial_summary_file = os.path.join(exp_dir, "trial_summary.csv")
        trial_metrics_df.to_csv(trial_summary_file, index=False)
        print(f"Trial metrics saved to: {trial_summary_file}")
    except Exception as e:
        print(f"Error saving trial metrics: {str(e)}")
    
    # Generate visualizations
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Clusters vs Sample Size
    if 'sample_size' in trial_metrics_df.columns:
        axes[0, 0].scatter(trial_metrics_df['sample_size'], trial_metrics_df['avg_clusters'], 
                          c=trial_metrics_df['vectorization'].astype('category').cat.codes, alpha=0.7)
        axes[0, 0].set_xlabel('Sample Size')
        axes[0, 0].set_ylabel('Average Number of Clusters')
        axes[0, 0].set_title('Clusters vs Sample Size')
    
    # 2. Noise vs Min Cluster Size
    if 'min_cluster_size' in trial_metrics_df.columns:
        axes[0, 1].scatter(trial_metrics_df['min_cluster_size'], trial_metrics_df['avg_noise_percentage'],
                          c=trial_metrics_df['vectorization'].astype('category').cat.codes, alpha=0.7)
        axes[0, 1].set_xlabel('Min Cluster Size')
        axes[0, 1].set_ylabel('Average Noise Percentage')
        axes[0, 1].set_title('Noise vs Min Cluster Size')
    
    # 3. Silhouette Score Distribution
    if 'avg_silhouette' in trial_metrics_df.columns:
        valid_silhouette = trial_metrics_df[trial_metrics_df['avg_silhouette'] > -1]['avg_silhouette']
        if len(valid_silhouette) > 0:
            axes[0, 2].hist(valid_silhouette, bins=20, alpha=0.7)
            axes[0, 2].set_xlabel('Average Silhouette Score')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Silhouette Score Distribution')
    
    # 4. Vectorization Method Comparison
    if 'vectorization' in trial_metrics_df.columns:
        vectorization_metrics = trial_metrics_df.groupby('vectorization')['avg_clusters'].mean()
        axes[1, 0].bar(vectorization_metrics.index, vectorization_metrics.values)
        axes[1, 0].set_xlabel('Vectorization Method')
        axes[1, 0].set_ylabel('Average Number of Clusters')
        axes[1, 0].set_title('Clusters by Vectorization Method')
    
    # 5. Parameter Correlation Heatmap
    numeric_cols = ['sample_size', 'min_cluster_size', 'min_samples', 'avg_clusters', 'avg_noise_percentage']
    available_cols = [col for col in numeric_cols if col in trial_metrics_df.columns]
    if len(available_cols) > 1:
        corr_matrix = trial_metrics_df[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Parameter Correlation Matrix')
    
    # 6. Best Trial Summary
    axes[1, 2].axis('off')
    if len(trial_metrics_df) > 0:
        # Find best trial based on combination of metrics
        # Normalize metrics and create composite score
        trial_copy = trial_metrics_df.copy()
        
        # Add min_cluster_size_ratio if it doesn't exist (for backward compatibility)
        if 'min_cluster_size_ratio' not in trial_copy.columns and 'min_cluster_size' in trial_copy.columns and 'sample_size' in trial_copy.columns:
            trial_copy['min_cluster_size_ratio'] = trial_copy['min_cluster_size'] / trial_copy['sample_size']
        
        # Higher is better for these metrics
        if 'avg_silhouette' in trial_copy.columns:
            trial_copy['silhouette_norm'] = (trial_copy['avg_silhouette'] - trial_copy['avg_silhouette'].min()) / (trial_copy['avg_silhouette'].max() - trial_copy['avg_silhouette'].min() + 1e-8)
        else:
            trial_copy['silhouette_norm'] = 0.5
            
        # Lower is better for noise
        if 'avg_noise_percentage' in trial_copy.columns:
            trial_copy['noise_norm'] = 1 - (trial_copy['avg_noise_percentage'] / 100)
        else:
            trial_copy['noise_norm'] = 0.5
            
        # Moderate number of clusters is often better
        if 'avg_clusters' in trial_copy.columns:
            trial_copy['clusters_norm'] = 1 - abs(trial_copy['avg_clusters'] - trial_copy['avg_clusters'].median()) / (trial_copy['avg_clusters'].max() + 1e-8)
        else:
            trial_copy['clusters_norm'] = 0.5
        
        # Composite score
        trial_copy['composite_score'] = (trial_copy['silhouette_norm'] + trial_copy['noise_norm'] + trial_copy['clusters_norm']) / 3
        
        best_trial = trial_copy.loc[trial_copy['composite_score'].idxmax()]
        
        best_trial_text = f"Best Trial (#{int(best_trial['trial'])}):\n"
        best_trial_text += f"Vectorization: {best_trial['vectorization']}\n"
        best_trial_text += f"Sample Size: {int(best_trial['sample_size'])}\n"
        best_trial_text += f"Min Cluster Size Ratio: {best_trial.get('min_cluster_size_ratio', best_trial['min_cluster_size']/best_trial['sample_size']):.3f}\n"
        best_trial_text += f"Min Cluster Size: {int(best_trial['min_cluster_size'])}\n"
        best_trial_text += f"Min Samples: {int(best_trial['min_samples'])}\n"
        best_trial_text += f"Avg Clusters: {best_trial['avg_clusters']:.1f}\n"
        best_trial_text += f"Avg Noise %: {best_trial['avg_noise_percentage']:.1f}\n"
        if best_trial['avg_silhouette'] > -1:
            best_trial_text += f"Avg Silhouette: {best_trial['avg_silhouette']:.3f}\n"
        best_trial_text += f"Composite Score: {best_trial['composite_score']:.3f}"
        
        axes[1, 2].text(0.1, 0.5, best_trial_text, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 2].set_title('Best Configuration')
    
    plt.tight_layout()
    
    # Save plot
    try:
        os.makedirs(exp_dir, exist_ok=True)  # Ensure directory exists
        plot_file = os.path.join(exp_dir, "hyperparameter_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Analysis plot saved to: {plot_file}")
    except Exception as e:
        print(f"Error saving analysis plot: {str(e)}")
    
    print(f"Analysis plots saved to: {plot_file}")
    print(f"Trial summary saved to: {trial_summary_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("="*60)
    
    if len(trial_metrics_df) > 0:
        print(f"Total trials completed: {len(trial_metrics_df)}")
        print(f"Average clusters per trial: {trial_metrics_df['avg_clusters'].mean():.1f}")
        print(f"Average noise percentage: {trial_metrics_df['avg_noise_percentage'].mean():.1f}%")
        
        if 'avg_silhouette' in trial_metrics_df.columns:
            valid_silhouette = trial_metrics_df[trial_metrics_df['avg_silhouette'] > -1]
            if len(valid_silhouette) > 0:
                print(f"Average silhouette score: {valid_silhouette['avg_silhouette'].mean():.3f}")
        
        print(f"\nBest trial: #{int(best_trial['trial'])}")
        print(f"Best configuration: {best_trial['vectorization']}, "
              f"sample_size={int(best_trial['sample_size'])}, "
              f"min_cluster_size_ratio={best_trial['min_cluster_size_ratio']:.3f}, "
              f"min_cluster_size={int(best_trial['min_cluster_size'])}, "
              f"min_samples={int(best_trial['min_samples'])}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyperparameter search for clustering")
    parser.add_argument("-n", "--n_trials", type=int, default=100, 
                        help="Number of hyperparameter combinations to try (default: 20)")
    parser.add_argument("-o", "--output_dir", type=str, default="clustering/hdbscan_hyperparameter_results",
                        help="Base output directory for results (default: hdbscan_hyperparameter_results)")
    parser.add_argument("-cm", "--clustering_mode", type=str, default="separate", choices=["separate", "combined"],
                        help="Run clustering separately per dataset or combined (default: separate)")
    
    args = parser.parse_args()
    
    print(f"Starting hyperparameter search with mode: {args.clustering_mode}")
    print(f"Output directory: {args.output_dir}/{args.clustering_mode}")
    print(f"Number of trials: {args.n_trials}")
    
    try:
        results_df = run_hyperparameter_search(n_trials=args.n_trials, output_dir=args.output_dir, 
                                          clustering_mode=args.clustering_mode)
    except Exception as e:
        print(f"Error during hyperparameter search: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("Hyperparameter search completed successfully!")
