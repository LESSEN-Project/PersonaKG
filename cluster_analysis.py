import json
import os
import shutil
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import HDBSCAN
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import random
from persona_dataset import PersonaDataset
import argparse
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class PersonaClusterAnalysis:
    def __init__(self, sample_size=100000, split="train", vectorization="tfidf"):
        self.persona_dataset = PersonaDataset()
        self.sample_size = sample_size
        self.split = split
        self.vectorization = vectorization  # 'tfidf', 'dense', or 'hybrid'
        self.output_dir = "clustering_results"
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
        
        # Initialize sentence transformer if using dense embeddings
        if self.vectorization in ['dense', 'hybrid']:
            print("Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.sentence_model = None
    
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
    
    def vectorize_statements_tfidf(self, statements, use_pca=True, pca_components=50):
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
        if use_pca and pca_components is not None and vectors.shape[1] > pca_components and vectors.shape[0] > 1:
            print(f"Applying PCA to TF-IDF vectors (from {vectors.shape[1]} to {pca_components} dimensions)")
            pca = PCA(n_components=min(pca_components, vectors.shape[0] - 1))
            vectors = pca.fit_transform(vectors)
            model_info['pca'] = pca
        
        return vectors, model_info
    
    def vectorize_statements_dense(self, statements, use_pca=True, pca_components=50):
        """Convert persona statements to dense embeddings using sentence transformers with optional PCA"""
        if self.sentence_model is None:
            print("Sentence transformer model not loaded! Initializing now...")
            self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
            
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
        if use_pca and pca_components is not None and embeddings.shape[1] > pca_components and embeddings.shape[0] > 1:
            print(f"Applying PCA to dense vectors (from {embeddings.shape[1]} to {pca_components} dimensions)")
            pca = PCA(n_components=min(pca_components, embeddings.shape[0] - 1))
            embeddings = pca.fit_transform(embeddings)
            model_info['pca'] = pca
        
        return embeddings, model_info
    
    def vectorize_statements_hybrid(self, statements, tfidf_weight=0.5, dense_weight=0.5, use_pca=True, pca_components=50):
        """Combine TF-IDF and dense embeddings with optional PCA reduction"""
        # Get TF-IDF vectors
        tfidf_vectors, tfidf_info = self.vectorize_statements_tfidf(statements, use_pca=False)
        
        # Get dense embeddings
        dense_vectors, dense_info = self.vectorize_statements_dense(statements, use_pca=False)
        
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
        if use_pca:
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
        hybrid_vectors = (tfidf_weight * tfidf_normalized) + (dense_weight * dense_normalized)
        
        # Apply final PCA if requested
        if use_pca and pca_components is not None and hybrid_vectors.shape[1] > pca_components:
            print(f"Applying final PCA to hybrid vectors (from {hybrid_vectors.shape[1]} to {pca_components} dimensions)")
            final_pca = PCA(n_components=min(pca_components, hybrid_vectors.shape[0] - 1))
            hybrid_vectors = final_pca.fit_transform(hybrid_vectors)
            model_info['pca'] = final_pca
        
        return hybrid_vectors, model_info
    
    def vectorize_statements(self, statements, tfidf_weight=0.5, dense_weight=0.5, use_pca=True, pca_components=50):
        """Vectorize statements based on the chosen method"""
        print(f"\nVectorizing {len(statements)} statements using {self.vectorization} method...")
        
        if self.vectorization == 'tfidf':
            return self.vectorize_statements_tfidf(statements, 
                                             use_pca=use_pca, 
                                             pca_components=pca_components)
        elif self.vectorization == 'dense':
            return self.vectorize_statements_dense(statements, 
                                              use_pca=use_pca, 
                                              pca_components=pca_components)
        elif self.vectorization == 'hybrid':
            return self.vectorize_statements_hybrid(statements, 
                                              tfidf_weight=tfidf_weight, 
                                              dense_weight=dense_weight, 
                                              use_pca=use_pca, 
                                              pca_components=pca_components)
        else:
            raise ValueError(f"Unknown vectorization method: {self.vectorization}")
    
    def cluster_with_hdbscan(self, vectors, min_cluster_size=5, min_samples=3):
        """Perform HDBSCAN clustering"""
        print(f"\nPerforming HDBSCAN clustering...")
        print(f"Min cluster size: {min_cluster_size}, Min samples: {min_samples}")
        print(f"Input vector dimensions: {vectors.shape[1]}")
        
        # No need for PCA here since we already apply it in the vectorization methods if requested
        
        # Perform HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(vectors)
        
        # Get cluster statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"\nClustering results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
        
        # Get cluster persistence (stability) if available
        if hasattr(clusterer, 'probabilities_'):
            # Use mean probabilities as a measure of cluster persistence
            cluster_persistence = {}
            for i in set(cluster_labels):
                if i != -1:  # Skip noise points
                    mask = cluster_labels == i
                    if hasattr(clusterer, 'probabilities_') and np.any(mask):
                        cluster_persistence[i] = np.mean(clusterer.probabilities_[mask])
                    else:
                        cluster_persistence[i] = 0.5  # Default value
        else:
            cluster_persistence = {i: 0.5 for i in set(cluster_labels) if i != -1}
        
        return cluster_labels, clusterer, cluster_persistence
    
    def save_cluster_results(self, statements, labels, output_prefix, 
                             cluster_persistence=None, n_samples=20):
        """Save clustering results to text files, overwriting any previous files"""

        # Prepare output directory: remove if exists, then (re)create
        cluster_dir = os.path.join(self.output_dir, output_prefix)
        if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)
        os.makedirs(cluster_dir, exist_ok=True)

        # Group statements by cluster
        clusters = defaultdict(list)
        for stmt, label in zip(statements, labels):
            clusters[label].append(stmt)

        # Separate noise points
        noise_points = clusters.get(-1, [])
        if -1 in clusters:
            del clusters[-1]

        # Save summary
        summary_path = os.path.join(cluster_dir, "clustering_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"HDBSCAN Clustering Analysis Summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total statements: {len(statements)}\n")
            f.write(f"Vectorization method: {self.vectorization}\n")
            f.write(f"Number of clusters: {len(clusters)}\n")
            f.write(f"Number of noise points: {len(noise_points)} ({len(noise_points)/len(statements)*100:.1f}%)\n")
            f.write(f"Analysis type: {output_prefix}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Dataset distribution
            dataset_counts = defaultdict(int)
            for stmt in statements:
                dataset_counts[stmt['dataset']] += 1

            f.write("Dataset Distribution:\n")
            for dataset, count in sorted(dataset_counts.items()):
                f.write(f"  {dataset}: {count} statements\n")
            f.write(f"\n{'='*60}\n\n")

            # Cluster sizes and persistence
            f.write("Cluster Information:\n")
            cluster_info = []
            for cluster_id, cluster_statements in sorted(clusters.items()):
                persistence = cluster_persistence[cluster_id] if cluster_persistence else None
                cluster_info.append((cluster_id, len(cluster_statements), persistence))

            cluster_info.sort(key=lambda x: x[1], reverse=True)

            for cluster_id, size, persistence in cluster_info:
                if persistence is not None:
                    f.write(f"  Cluster {cluster_id}: {size} statements (persistence: {persistence:.3f})\n")
                else:
                    f.write(f"  Cluster {cluster_id}: {size} statements\n")

        # Save noise points if any
        if noise_points:
            noise_file = os.path.join(cluster_dir, "noise_points.txt")
            with open(noise_file, 'w', encoding='utf-8') as f:
                f.write(f"NOISE POINTS (Not assigned to any cluster)\n")
                f.write(f"{'='*60}\n")
                f.write(f"Total noise points: {len(noise_points)}\n\n")

                # Group by dataset
                noise_by_dataset = defaultdict(list)
                for stmt in noise_points:
                    noise_by_dataset[stmt['dataset']].append(stmt)

                for dataset in sorted(noise_by_dataset.keys()):
                    dataset_noise = noise_by_dataset[dataset]
                    f.write(f"\n--- {dataset} ({len(dataset_noise)} points) ---\n\n")
                    sample_size = min(n_samples, len(dataset_noise))
                    sampled = random.sample(dataset_noise, sample_size)
                    for i, stmt in enumerate(sampled, 1):
                        f.write(f"{i}. {stmt['statement']}\n")
                        f.write(f"   (ID: {stmt['dataset_id']})\n\n")

        # Save each cluster to a separate file
        for cluster_id, cluster_statements in sorted(clusters.items()):
            cluster_file = os.path.join(cluster_dir, f"cluster_{cluster_id:03d}.txt")
            with open(cluster_file, 'w', encoding='utf-8') as f:
                f.write(f"CLUSTER {cluster_id}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Total statements in cluster: {len(cluster_statements)}\n")
                if cluster_persistence is not None:
                    f.write(f"Cluster persistence (stability): {cluster_persistence[cluster_id]:.3f}\n")
                f.write("\n")
                # Dataset distribution in this cluster
                cluster_dataset_counts = defaultdict(int)
                for stmt in cluster_statements:
                    cluster_dataset_counts[stmt['dataset']] += 1
                f.write("Dataset distribution in this cluster:\n")
                for dataset, count in sorted(cluster_dataset_counts.items()):
                    percentage = (count / len(cluster_statements)) * 100
                    f.write(f"  {dataset}: {count} ({percentage:.1f}%)\n")
                f.write(f"\n{'='*60}\n\n")
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
                f.write(f"\n{'='*60}\n")

        print(f"Results saved to {cluster_dir}/")

    def filter_similar_statements(self, statements, similarity_threshold=0.8):
        """Filter out statements that are too similar to each other based on a threshold"""
        if not statements or len(statements) < 2:
            return statements
            
        print(f"\nFiltering similar statements with threshold {similarity_threshold}...")
        print(f"Initial number of statements: {len(statements)}")
        
        # Use sentence transformer for computing embeddings
        if self.sentence_model is None:
            print("Sentence transformer model not loaded! Initializing now...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
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
            similar_indices = np.where(similarity_matrix[i] > similarity_threshold)[0]
            
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
        
        # Print sample similar pairs that were filtered
        if similar_pairs:
            # Sort by similarity score (highest first)
            similar_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Show at most 5 examples
            print("\nExamples of similar statements that were removed:")
            
            # Pick either top 5 or a random sample of 5 if there are many
            if len(similar_pairs) > 20:
                # Mix of highest similarity and random samples
                sample_pairs = similar_pairs[:3]  # 3 highest similarity pairs
                sample_pairs.extend(random.sample(similar_pairs[3:], min(2, len(similar_pairs)-3)))  # 2 random pairs
            else:
                sample_pairs = similar_pairs[:min(5, len(similar_pairs))]
                
            for i, idx, sim_score in sample_pairs:
                print(f"\nSimilarity: {sim_score:.3f}")
                print(f"Original: \"{statements[i]['statement']}\"")
                print(f"Removed:  \"{statements[idx]['statement']}\"")
                print(f"Datasets: {statements[i]['dataset']} / {statements[idx]['dataset']}")
        
        return filtered_statements

    def run_clustering_all_datasets(self, personas_dict):
        """Run clustering on all datasets combined"""
        print("\n" + "="*60)
        print("CLUSTERING ALL DATASETS TOGETHER")
        print("="*60)
        
        # Extract all statements
        statements = self.extract_persona_statements(personas_dict)
        print(f"\nTotal persona statements extracted: {len(statements)}")
        
        if len(statements) < 10:
            print("Not enough statements for clustering")
            return
        
        # Filter similar statements
        statements = self.filter_similar_statements(statements, similarity_threshold=0.8)
        
        if len(statements) < 10:
            print("Not enough statements after filtering for clustering")
            return
        
        # Vectorize
        vectors, model_info = self.vectorize_statements(statements)
        
        # Determine min_cluster_size based on data size
        min_cluster_size = max(5, int(len(statements) * 0.05))  # 0.5% of data
        min_samples = max(3, int(min_cluster_size * 0.5))
        
        # Perform clustering
        labels, clusterer, persistence = self.cluster_with_hdbscan(
            vectors, 
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        
        # Save results
        self.save_cluster_results(
            statements, 
            labels, 
            f"all_datasets_{self.vectorization}",
            cluster_persistence=persistence
        )
    
    def run_clustering_per_dataset(self, personas_dict):
        """Run clustering separately for each dataset"""
        print("\n" + "="*60)
        print("CLUSTERING EACH DATASET SEPARATELY")
        print("="*60)
        
        for dataset_name in self.datasets:

            print(f"\n\nClustering {dataset_name}...")
            print("-" * 40)
            
            # Extract statements for this dataset only
            statements = self.extract_persona_statements(personas_dict, dataset_filter=dataset_name)
            print(f"Persona statements from {dataset_name}: {len(statements)}")
            
            # Filter similar statements
            statements = self.filter_similar_statements(statements, similarity_threshold=0.8)
            
            if len(statements) < 5:
                print(f"Not enough statements from {dataset_name} after filtering for clustering")
                continue
            
            # Vectorize
            vectors, model_info = self.vectorize_statements(statements)
            
            # Adjust parameters for smaller datasets
            min_cluster_size = max(3, int(len(statements) * 0.02))
            # min_samples = None
            min_samples = max(2, int(min_cluster_size * 0.5))
            
            # Perform clustering
            labels, clusterer, persistence = self.cluster_with_hdbscan(
                vectors,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples
            )
            
            # Save results
            self.save_cluster_results(
                statements, 
                labels, 
                f"dataset_{dataset_name}_{self.vectorization}",
                cluster_persistence=persistence
            )
    
    def run(self, mode="both"):
        """Run the complete clustering analysis"""
        print(f"Starting Persona Clustering Analysis")
        print(f"Sample size: {self.sample_size}")
        print(f"Split: {self.split}")
        print(f"Vectorization: {self.vectorization}")
        print(f"Mode: {mode}")
        
        # Load all personas
        personas_dict = self.load_all_personas()
        
        # Run clustering based on mode
        if mode == "combined":
            self.run_clustering_all_datasets(personas_dict)
        
        if mode == "separate":
            self.run_clustering_per_dataset(personas_dict)
        
        print(f"\nClustering analysis complete! Results saved to {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Persona Clustering Analysis using HDBSCAN")
    parser.add_argument("--sample_size", type=int, default=100000, 
                       help="Sample size for each dataset (default: 10000)")
    parser.add_argument("--split", type=str, default="train", 
                       choices=["train", "validation", "test"],
                       help="Dataset split to use (default: train)")
    parser.add_argument("--mode", type=str, default="separate", 
                       choices=["combined", "separate"],
                       help="Clustering mode: combined (all datasets together), or separate (per dataset)")
    parser.add_argument("--vectorization", type=str, default="dense",
                       choices=["tfidf", "dense", "hybrid"],
                       help="Vectorization method: tfidf, dense (sentence embeddings), or hybrid (both)")
    
    args = parser.parse_args()
    
    analyzer = PersonaClusterAnalysis(
        sample_size=args.sample_size,
        split=args.split,
        vectorization=args.vectorization
    )
    
    analyzer.run(mode=args.mode)


if __name__ == "__main__":
    main()
