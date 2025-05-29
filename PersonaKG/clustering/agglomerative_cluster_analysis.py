import json
import os
import sys
import shutil
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import random
import argparse
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import silhouette_score

from PersonaKG.persona_dataset import PersonaDataset


class PersonaAgglomerativeClusterAnalysis:
    def __init__(self, sample_size=100000, split="train", vectorization="tfidf", mode="separate", pca_components=None, 
                 tfidf_weight=0.5, n_clusters=100, linkage='ward', metric='euclidean'):
        self.persona_dataset = PersonaDataset()
        self.sample_size = sample_size
        self.split = split
        self.vectorization = vectorization 
        self.mode = mode
        self.pca_components = pca_components
        self.tfidf_weight = tfidf_weight
        self.n_clusters = n_clusters  # AgglomerativeClustering parameter: number of clusters
        self.linkage = linkage  # AgglomerativeClustering parameter: linkage criterion
        self.metric = "euclidean" 

        self.output_dir = os.path.join("clustering/agglomerative_clusters", self.mode)
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        else:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Dataset names to analyze
        self.datasets = [
            "PersonaChat",
            "SyntheticPersonaChat", 
            "MSC",
            "PEC",
            "FoCus",
            "MPChat",
            "PER-CHAT"
        ]
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
        self.similarity_threshold = 0.6

    
    def load_all_personas(self):
        """Load personas from all datasets"""
        all_personas = {}
        
        for dataset_name in self.datasets:
            print(f"\nLoading personas from {dataset_name}...")
            try:
                personas = self.persona_dataset.get_personas_from_dataset(
                    dataset_name, 
                    split=self.split, 
                    sample_size=self.sample_size
                )
                
                all_personas[dataset_name] = personas
                print(f"Loaded {len(personas)} personas from {dataset_name}")
                
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                continue
                
        return all_personas
    
    def extract_persona_statements(self, personas_dict, dataset_filter=None):
        """Extract all persona statements with their metadata"""
        statements_data = []
        
        datasets_to_process = [dataset_filter] if dataset_filter else personas_dict.keys()
        
        for dataset_name in datasets_to_process:
            if dataset_name not in personas_dict:
                continue
                
            personas = personas_dict[dataset_name]
            
            for persona in personas:
                persona_statements = persona.get('persona_statements', [])
                
                # Handle both string and list formats
                if isinstance(persona_statements, str):
                    persona_statements = [persona_statements]
                
                for statement in persona_statements:
                    if statement and isinstance(statement, str) and statement.strip():
                        statements_data.append({
                            'statement': statement.strip(),
                            'dataset': dataset_name,
                            'persona_id': persona['id'],
                            'dataset_id': persona['dataset_id']
                        })
        
        return statements_data
    
    def vectorize_statements_tfidf(self, statements, pca_components=None):
        """Convert persona statements to TF-IDF vectors with optional PCA"""
        texts = [s['statement'] for s in statements]
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better phrase capture
            min_df=2,
            max_df=0.95,
            sublinear_tf=True  # Use sublinear tf scaling
        )
        
        vectors = vectorizer.fit_transform(texts).toarray()
        
        model_info = {'vectorizer': vectorizer}
        
        # Apply PCA if requested and pca_components is not None
        if pca_components is not None and vectors.shape[1] > pca_components and vectors.shape[0] > 1:
            print(f"Applying PCA to TF-IDF vectors (from {vectors.shape[1]} to {pca_components} dimensions)")
            pca = PCA(n_components=min(pca_components, vectors.shape[0] - 1))
            vectors = pca.fit_transform(vectors)
            model_info['pca'] = pca
        
        return vectors, model_info
    
    def vectorize_statements_dense(self, statements, pca_components=None):
        """Convert persona statements to dense embeddings using sentence transformers with optional PCA"""
            
        texts = [s['statement'] for s in statements]
        
        # Generate embeddings (batch processing)
        embeddings = self.sentence_model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=64,
            convert_to_numpy=True
        )
        
        model_info = {}
        
        # Apply PCA if requested and pca_components is not None
        if pca_components is not None and embeddings.shape[1] > pca_components and embeddings.shape[0] > 1:
            print(f"Applying PCA to dense vectors (from {embeddings.shape[1]} to {pca_components} dimensions)")
            pca = PCA(n_components=min(pca_components, embeddings.shape[0] - 1))
            embeddings = pca.fit_transform(embeddings)
            model_info['pca'] = pca
        
        return embeddings, model_info
    
    def vectorize_statements_hybrid(self, statements, tfidf_weight=0.5, pca_components=None):
        """Combine TF-IDF and dense embeddings with optional PCA reduction"""
        # Get TF-IDF vectors
        tfidf_vectors, tfidf_info = self.vectorize_statements_tfidf(statements)
        
        # Get dense embeddings
        dense_vectors, dense_info = self.vectorize_statements_dense(statements)
        
        model_info = {}
        if 'vectorizer' in tfidf_info:
            model_info['vectorizer'] = tfidf_info['vectorizer']
        
        # Store original shapes for debugging
        tfidf_shape = tfidf_vectors.shape
        dense_shape = dense_vectors.shape
        
        # Get the minimum dimension to reduce both vector types to the same size
        # If one is already smaller than pca_components, use that as the target dimension
        common_dim = min(pca_components, dense_vectors.shape[1])
        
        # Apply PCA separately if requested
        if pca_components is not None:
            # For TF-IDF vectors
            if tfidf_vectors.shape[1] > common_dim and tfidf_vectors.shape[0] > 1:
                print(f"Applying PCA to TF-IDF vectors (from {tfidf_vectors.shape[1]} to {common_dim} dimensions)")
                tfidf_pca = PCA(n_components=min(common_dim, tfidf_vectors.shape[0] - 1))
                tfidf_vectors = tfidf_pca.fit_transform(tfidf_vectors)
                model_info['tfidf_pca'] = tfidf_pca
            
            # For dense vectors - only if they need reduction and are larger than common_dim
            if dense_vectors.shape[1] > common_dim and dense_vectors.shape[0] > 1:
                print(f"Applying PCA to dense vectors (from {dense_vectors.shape[1]} to {common_dim} dimensions)")
                dense_pca = PCA(n_components=min(common_dim, dense_vectors.shape[0] - 1))
                dense_vectors = dense_pca.fit_transform(dense_vectors)
                model_info['dense_pca'] = dense_pca
            
            # Print final dimensions to verify they match
            print(f"Final dimensions - TF-IDF: {tfidf_vectors.shape}, Dense: {dense_vectors.shape}")
        
        # Normalize vectors before combining
        scaler_tfidf = StandardScaler()
        scaler_dense = StandardScaler()
        
        tfidf_normalized = scaler_tfidf.fit_transform(tfidf_vectors)
        dense_normalized = scaler_dense.fit_transform(dense_vectors)
        
        # Ensure dimensions match before combining
        if tfidf_normalized.shape[1] != dense_normalized.shape[1]:
            raise ValueError(f"Dimension mismatch after normalization: TF-IDF shape {tfidf_normalized.shape}, Dense shape {dense_normalized.shape}. Original shapes: TF-IDF {tfidf_shape}, Dense {dense_shape}")
            
        # Combine with weights
        hybrid_vectors = (tfidf_weight * tfidf_normalized) + ((1-tfidf_weight) * dense_normalized)
        
        # Apply final PCA if requested
        if pca_components is not None and hybrid_vectors.shape[1] > pca_components:
            print(f"Applying final PCA to hybrid vectors (from {hybrid_vectors.shape[1]} to {pca_components} dimensions)")
            final_pca = PCA(n_components=min(pca_components, hybrid_vectors.shape[0] - 1))
            hybrid_vectors = final_pca.fit_transform(hybrid_vectors)
            model_info['pca'] = final_pca
        
        return hybrid_vectors, model_info
    
    def vectorize_statements(self, statements):
        """Vectorize statements using the specified method"""
        if self.vectorization == "tfidf":
            return self.vectorize_statements_tfidf(statements, pca_components=self.pca_components)
        elif self.vectorization == "dense":
            return self.vectorize_statements_dense(statements, pca_components=self.pca_components)
        elif self.vectorization == "hybrid":
            return self.vectorize_statements_hybrid(statements, tfidf_weight=self.tfidf_weight, pca_components=self.pca_components)
        else:
            raise ValueError(f"Unknown vectorization method: {self.vectorization}")
    
    def calculate_cluster_diversity(self, labels):
        """Calculate diversity metric for clustering results.
        
        A higher diversity score means more evenly distributed cluster sizes (better clustering).
        Score is normalized to be between 0 and 1, where:
        - 1 means perfectly balanced clusters (all clusters have the same number of samples)
        - 0 means highly imbalanced clusters (most samples in one cluster)
        """
        # Count samples in each cluster
        unique_labels = set(labels)
        cluster_sizes = [int(np.sum(labels == label)) for label in unique_labels]
        
        if len(cluster_sizes) <= 1:
            return 0.0  # No diversity with only one cluster
        
        # Calculate coefficient of variation (standard deviation / mean)
        mean_size = np.mean(cluster_sizes)
        std_dev = np.std(cluster_sizes)
        
        if mean_size == 0:
            return 0.0
            
        cv = std_dev / mean_size
        
        # Convert to diversity score (1 - normalized CV)
        # We use an exponential function to transform the CV into a 0-1 score
        # where 0 means high variation (poor diversity) and 1 means low variation (good diversity)
        diversity_score = np.exp(-cv)
        
        print(f"Cluster sizes: {cluster_sizes}")
        print(f"Diversity score: {diversity_score:.4f} (higher is better)")
        
        return diversity_score
    
    def cluster_with_agglomerative(self, vectors, n_clusters=10, linkage='ward', metric='euclidean'):
        """Perform Agglomerative clustering"""
        print(f"\nPerforming Agglomerative clustering...")
        print(f"Number of clusters: {n_clusters}, Linkage: {linkage}, metric: {metric}")
        print(f"Input vector dimensions: {vectors.shape[1]}")
                
        # Perform Agglomerative clustering
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric
        )
        
        cluster_labels = clusterer.fit_predict(vectors)
        
        # Get cluster statistics
        n_clusters_actual = len(set(cluster_labels))
        
        print(f"\nClustering results:")
        print(f"  Number of clusters: {n_clusters_actual}")
        
        # Calculate cluster diversity score
        diversity_score = self.calculate_cluster_diversity(cluster_labels)
        
        # Calculate silhouette score
        silhouette = -1  # Default value if calculation fails
        try:
            if n_clusters_actual > 1 and vectors.shape[0] > 1:
                silhouette = silhouette_score(vectors, cluster_labels)
                print(f"  Silhouette score: {silhouette:.4f}")
            else:
                print("  Silhouette score: Not applicable (requires at least 2 clusters)")
        except Exception as e:
            print(f"  Error calculating silhouette score: {e}")
        
        # Agglomerative clustering doesn't have a concept of noise points or probabilities
        
        # Store the scores and cluster model in the object for later use
        self.diversity_score = diversity_score
        self.silhouette_score = silhouette
        self.cluster_model = clusterer
        
        return cluster_labels, clusterer
    
    def save_cluster_results(self, statements, labels, output_prefix, 
                            n_samples=20, dataset_name=None):
        """Save clustering results to text files, overwriting any previous files"""

        # Determine the correct directory structure
        # For separate mode, each dataset gets its own folder under clusters/agglomerative_clusters/
        # For combined mode, results go in clusters/agglomerative_clusters/
        
        if self.mode == "separate" and dataset_name:
            # For separate mode with dataset name - save directly to dataset folder
            cluster_dir = os.path.join(self.output_dir, dataset_name)
        elif self.mode == "combined":
            # For combined mode, save to combined subfolder
            cluster_dir = os.path.join(self.output_dir)
        else:
            # Fallback (should not happen)
            cluster_dir = os.path.join(self.output_dir, output_prefix)
        
        # Create or clean directory
        if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)
        os.makedirs(cluster_dir, exist_ok=True)

        # Group statements by cluster
        clusters = defaultdict(list)
        for stmt, label in zip(statements, labels):
            clusters[label].append(stmt)

        # Save summary
        summary_path = os.path.join(cluster_dir, "clustering_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Agglomerative Clustering Analysis Summary\n")
            f.write(f"{('='*60)}\n")
            f.write(f"Total statements: {len(statements)}\n")
            f.write(f"Vectorization method: {self.vectorization}\n")
            f.write(f"Number of clusters: {len(clusters)}\n")
            f.write(f"Analysis type: {output_prefix}\n")
            f.write(f"Diversity score: {getattr(self, 'diversity_score', 0.0):.4f} (higher is better)\n")
            f.write(f"Silhouette score: {getattr(self, 'silhouette_score', -1):.4f} (higher is better)\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Dataset distribution
            dataset_counts = defaultdict(int)
            for stmt in statements:
                dataset_counts[stmt['dataset']] += 1

            f.write("Dataset Distribution:\n")
            for dataset, count in sorted(dataset_counts.items()):
                f.write(f"  {dataset}: {count} statements\n")
            f.write(f"\n{('='*60)}\n\n")

            f.write("Cluster Information:\n")
            cluster_info = []
            for cluster_id, cluster_statements in sorted(clusters.items()):
                cluster_info.append((cluster_id, len(cluster_statements)))

            cluster_info.sort(key=lambda x: x[1], reverse=True)

            for cluster_id, size in cluster_info:
                f.write(f"  Cluster {cluster_id}: {size} statements\n")

        # Save each cluster to a separate file
        for cluster_id, cluster_statements in sorted(clusters.items()):
            cluster_file = os.path.join(cluster_dir, f"cluster_{cluster_id:03d}.txt")
            with open(cluster_file, 'w', encoding='utf-8') as f:
                f.write(f"CLUSTER {cluster_id}\n")
                f.write(f"{('='*60)}\n")
                f.write(f"Total statements in cluster: {len(cluster_statements)}\n")
                f.write("\n")
                # Dataset distribution in this cluster
                cluster_dataset_counts = defaultdict(int)
                for stmt in cluster_statements:
                    cluster_dataset_counts[stmt['dataset']] += 1
                f.write("Dataset distribution in this cluster:\n")
                for dataset, count in sorted(cluster_dataset_counts.items()):
                    percentage = (count / len(cluster_statements)) * 100
                    f.write(f"  {dataset}: {count} ({percentage:.1f}%)\n")
                f.write(f"\n{('='*60)}\n\n")
                f.write("SAMPLE STATEMENTS FROM EACH DATASET:\n\n")
                # Group by dataset
                by_dataset = defaultdict(list)
                for stmt in cluster_statements:
                    by_dataset[stmt['dataset']].append(stmt)
                for dataset in sorted(by_dataset.keys()):
                    dataset_statements = by_dataset[dataset]
                    f.write(f"\n--- {dataset} ---\n")
                    f.write(f"Total from this dataset: {len(dataset_statements)}\n\n")
                    sample_size = min(n_samples, len(dataset_statements))
                    sampled = random.sample(dataset_statements, sample_size)
                    for i, stmt in enumerate(sampled, 1):
                        f.write(f"{i}. {stmt['statement']}\n")
                        f.write(f"   (ID: {stmt['dataset_id']})\n\n")
                f.write(f"\n{('='*60)}\n")

        print(f"Results saved to {cluster_dir}/")
        
        # Return results dictionary for hyperparameter tuning script
        return {
            'n_clusters': len(clusters),
            'total_statements': len(statements),
            'diversity_score': getattr(self, 'diversity_score', 0.0),
            'silhouette_score': getattr(self, 'silhouette_score', -1)
        }

    def filter_similar_statements(self, statements):
        """Filter out statements that are too similar to each other based on a threshold"""
        if not statements or len(statements) < 2:
            return statements
            
        print(f"\nFiltering similar statements with threshold {self.similarity_threshold}...")
        print(f"Initial number of statements: {len(statements)}")
        
        # Get text from statements
        texts = [s['statement'] for s in statements]
        
        # Generate embeddings
        embeddings = self.sentence_model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=64,
            convert_to_numpy=True
        )
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Set diagonal to 0 to avoid self-similarity
        np.fill_diagonal(similarity_matrix, 0)
        
        # Track which indices to keep and which to remove
        to_remove = set()
        similar_pairs = []  # Store examples of similar pairs
        
        for i in range(len(statements)):
            # Skip if this statement is already marked for removal
            if i in to_remove:
                continue
                
            # Find all statements that are too similar to this one
            similar_indices = np.where(similarity_matrix[i] > self.similarity_threshold)[0]
            
            # Mark similar statements for removal
            # But not the current statement (it's our reference point)
            for idx in similar_indices:
                if idx > i:  # Only consider statements we haven't processed yet
                    to_remove.add(idx)
                    # Store this pair as an example (statement i and statement idx)
                    similar_pairs.append((i, idx, similarity_matrix[i][idx]))
        
        # Keep only statements that weren't marked for removal
        indices_to_keep = [i for i in range(len(statements)) if i not in to_remove]
        filtered_statements = [statements[i] for i in indices_to_keep]
        
        num_removed = len(statements) - len(filtered_statements)
        removal_percentage = (num_removed / len(statements)) * 100 if statements else 0
        
        print(f"Removed {num_removed} similar statements ({removal_percentage:.2f}%)")
        print(f"Remaining statements: {len(filtered_statements)}")
        
        return filtered_statements

    def run_clustering_all_datasets(self, personas_dict):
        """Run clustering on all datasets combined"""
        print("\n" + "="*60)
        print("CLUSTERING ALL DATASETS TOGETHER")
        print("="*60)
        
        # Extract statements from all datasets
        statements = self.extract_persona_statements(personas_dict)
        print(f"Total persona statements extracted: {len(statements)}")
        
        # Filter similar statements to reduce redundancy
        statements = self.filter_similar_statements(statements)
        
        if len(statements) < 10:
            print("Not enough statements after filtering for clustering")
            return
        
        # Vectorize statements
        vectors, model_info = self.vectorize_statements(statements)
        
        # Dynamically adjust n_clusters based on data size if needed
        actual_n_clusters = min(self.n_clusters, len(statements) // 10)
        if actual_n_clusters != self.n_clusters:
            print(f"Adjusting number of clusters from {self.n_clusters} to {actual_n_clusters} based on data size")
        
        # Perform clustering
        labels, clusterer = self.cluster_with_agglomerative(
            vectors, 
            n_clusters=actual_n_clusters,
            linkage=self.linkage,
            metric=self.metric
        )
        
        # Save results
        self.save_cluster_results(
            statements, 
            labels, 
            f"all_datasets",
        )
        
        # Calculate statistics
        n_clusters = len(set(labels))
        
        # Collect results for summary
        results = {
            'total_statements': len(statements),
            'n_clusters': n_clusters,
            'n_noise': 0,  # Agglomerative clustering doesn't have noise points
            'noise_percentage': 0
        }
        
        # Save summary
        self.save_summary(results)
    
    def run_clustering_per_dataset(self, personas_dict):
        """Run clustering separately for each dataset"""
        print("\n" + "="*60)
        print("CLUSTERING EACH DATASET SEPARATELY")
        print("="*60)
        
        # Dictionary to collect results for summary
        all_results = {}
        
        for dataset_name in self.datasets:

            print(f"\n\nClustering {dataset_name}...")
            print("-" * 40)
            
            # Extract statements for this dataset only
            statements = self.extract_persona_statements(personas_dict, dataset_filter=dataset_name)
            print(f"Persona statements from {dataset_name}: {len(statements)}")
            
            # Filter similar statements
            statements = self.filter_similar_statements(statements)
            
            if len(statements) < 5:
                print(f"Not enough statements from {dataset_name} after filtering for clustering")
                continue
            
            # Vectorize
            vectors, model_info = self.vectorize_statements(statements)
            
            # Dynamically adjust n_clusters based on data size if needed
            actual_n_clusters = min(self.n_clusters, len(statements) // 10)
            if actual_n_clusters != self.n_clusters:
                print(f"Adjusting number of clusters from {self.n_clusters} to {actual_n_clusters} based on data size")
            
            # Perform clustering
            labels, clusterer = self.cluster_with_agglomerative(
                vectors,
                n_clusters=actual_n_clusters,
                linkage=self.linkage,
                metric=self.metric
            )
            
            # Save results
            self.save_cluster_results(
                statements, 
                labels, 
                f"{dataset_name}",
                dataset_name=dataset_name
            )
            
            # Calculate statistics
            n_clusters = len(set(labels))
            
            # Collect results for this dataset
            all_results[dataset_name] = {
                'total_statements': len(statements),
                'n_clusters': n_clusters,
                'n_noise': 0,  # Agglomerative clustering doesn't have noise points
                'noise_percentage': 0
            }
        
        # Save summary
        if all_results:
            self.save_summary(all_results)
    
    def save_hyperparameters(self, extra_params=None):
        """Save hyperparameters to a JSON file next to the dataset folders"""
        params = {
            "sample_size": self.sample_size,
            "split": self.split,
            "vectorization": self.vectorization,
            "mode": self.mode,
            "pca_components": self.pca_components,
            "tfidf_weight": self.tfidf_weight,
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
            "metric": self.metric,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Add any additional parameters
        if extra_params:
            params.update(extra_params)
            
        # Save the parameters to a JSON file
        params_file = os.path.join(self.output_dir, "hyperparameters.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
            
        print(f"Saved hyperparameters to {params_file}")
        
    def save_summary(self, results):
        """Save a summary of the clustering results"""
        if self.mode == "combined":
            # For combined mode: one summary file with overall stats
            summary_file = os.path.join(self.output_dir, "summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"CLUSTERING SUMMARY (Combined Mode)\n")
                f.write(f"{'-'*50}\n")
                f.write(f"Sample Size: {self.sample_size}\n")
                f.write(f"Vectorization: {self.vectorization}\n")
                f.write(f"Split: {self.split}\n")
                f.write(f"PCA Components: {self.pca_components}\n")
                f.write(f"TF-IDF Weight: {self.tfidf_weight}\n")
                f.write(f"Number of Clusters: {self.n_clusters}\n")
                f.write(f"Linkage: {self.linkage}\n")
                f.write(f"metric: {self.metric}\n\n")
                
                # Write the results
                f.write(f"Total Statements: {results['total_statements']}\n")
                f.write(f"Number of Clusters: {results['n_clusters']}\n")
            
            print(f"Saved summary to {summary_file}")
        else:
            # For separate mode: CSV-style table with each dataset's stats
            summary_file = os.path.join(self.output_dir, "summary.csv")
            with open(summary_file, 'w') as f:
                # Write header
                f.write("Dataset,Total Statements,Number of Clusters\n")
                
                # Write each dataset's results
                for dataset, stats in results.items():
                    f.write(f"{dataset},{stats['total_statements']},{stats['n_clusters']}\n")
            
            print(f"Saved summary to {summary_file}")
    
    def run(self, mode="both"):
        
        print(f"Starting Persona Agglomerative Clustering Analysis")
        print(f"Sample size: {self.sample_size}")
        print(f"Split: {self.split}")
        print(f"Vectorization: {self.vectorization}")
        print(f"Mode: {mode}")
        print(f"PCA Components: {self.pca_components}")
        print(f"TF-IDF Weight: {self.tfidf_weight}")
        print(f"Number of Clusters: {self.n_clusters}")
        print(f"Linkage: {self.linkage}")
        print(f"metric: {self.metric}")
        print(f"Results will be saved to: {self.output_dir}/")
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Load all personas
        personas_dict = self.load_all_personas()
        
        # Run clustering based on mode
        if mode == "combined":
            self.run_clustering_all_datasets(personas_dict)
        
        if mode == "separate":
            self.run_clustering_per_dataset(personas_dict)
        
        print(f"\nClustering analysis complete! Results saved to {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Persona Clustering Analysis using Agglomerative Clustering")
    add = parser.add_argument
    add("-s", "--sample_size", type=int, default=500, help="Sample size for each dataset (default: 500)")
    add("-t", "--split", type=str, default="train", choices=["train", "validation", "test"], help="Dataset split to use (default: train)")
    add("-m", "--mode", type=str, default="combined", choices=["combined", "separate"], help="Clustering mode: combined (all datasets together), or separate (per dataset)")
    add("-v", "--vectorization", type=str, default="dense", choices=["tfidf", "dense", "hybrid"], help="Vectorization method: tfidf, dense (sentence embeddings), or hybrid (both)")
    add("-p", "--pca_components", type=int, default=48, help="Number of PCA components to use (default: 48)")
    add("-w", "--tfidf_weight", type=float, default=0.5, help="Weight for TF-IDF vectors in hybrid vectorization (default: 0.5)")
    add("-n", "--n_clusters", type=int, default=35, help="Number of clusters for Agglomerative clustering (default: 50)")
    add("-l", "--linkage", type=str, default="complete", choices=["ward", "complete", "average", "single"], help="Linkage criterion for Agglomerative clustering (default: ward)")
    
    args = parser.parse_args()
    
    analyzer = PersonaAgglomerativeClusterAnalysis(
        sample_size=args.sample_size,
        split=args.split,
        vectorization=args.vectorization,
        mode=args.mode,
        pca_components=args.pca_components,
        tfidf_weight=args.tfidf_weight,
        n_clusters=args.n_clusters,
        linkage=args.linkage,
    )
    
    analyzer.run(mode=args.mode)


if __name__ == "__main__":
    main()
