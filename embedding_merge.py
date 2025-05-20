import json
import numpy as np
import time
import os
from tqdm import tqdm
import gc
import torch

from sentence_transformers import SentenceTransformer
from persona_dataset import PersonaDataset


class EmbeddingPersonaMerger:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize with a small but effective sentence transformer model"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.persona_dataset = PersonaDataset()
        
    def compute_embeddings(self, personas, batch_size=32):
        """Compute embeddings for personas in batches"""
        if not personas:
            return np.array([])
            
        # Process in batches to avoid memory issues
        embeddings = []
        for i in tqdm(range(0, len(personas), batch_size), desc="Computing embeddings"):
            batch = personas[i:i+batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.append(batch_embeddings)
            
            # Explicitly call garbage collection after each batch
            if (i+1) % 10 == 0:
                gc.collect()
            
        # Combine all batches
        embeddings = np.vstack(embeddings)
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norm
        
        # Force garbage collection
        gc.collect()
        
        return normalized_embeddings
        
    def find_similar_personas(self, personas, embeddings, threshold=0.85):
        """Find clusters of similar personas within a dataset using embedding similarity"""
        if len(personas) <= 1 or len(embeddings) <= 1:
            return [[p] for p in personas]  # Each persona in its own cluster if nothing to compare
            
        # Calculate similarity matrix (this is efficient with numpy)
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Find clusters
        clusters = []
        used_indices = set()
        
        for i in tqdm(range(len(personas)), desc="Finding similar personas"):
            if i in used_indices:
                continue
                
            # Find all personas similar to this one
            cluster = [i]
            for j in range(i+1, len(personas)):
                if j not in used_indices and similarity_matrix[i, j] > threshold:
                    cluster.append(j)
                    
            if len(cluster) > 1:  # We found similar personas
                clusters.append([personas[idx] for idx in cluster])
                used_indices.update(cluster)
            else:
                # No similar personas found, keep as is
                if i not in used_indices:
                    clusters.append([personas[i]])
                    used_indices.add(i)
        
        return clusters
        
    def process_dataset(self, dataset_name, similarity_threshold=0.85):
        """Process a single dataset, removing redundancy within it"""
        try:
            print(f"\n===== Processing dataset: {dataset_name} =====")
            start_time = time.time()
            
            # Get personas for this dataset
            personas = self.persona_dataset.get_personas_from_dataset(dataset_name)
            if not personas:
                print(f"No personas found for {dataset_name}")
                return []
                
            print(f"{dataset_name}: {len(personas)} personas loaded")
            
            # Compute embeddings for this dataset only
            embeddings = self.compute_embeddings(personas)
            if len(embeddings) == 0:
                return personas  # No embeddings could be computed, just return as is
                
            # Find clusters within this dataset
            clusters = self.find_similar_personas(personas, embeddings, threshold=similarity_threshold)
            original_count = len(personas)
            
            # Select a representative from each cluster
            unique_personas = []
            for cluster in clusters:
                unique_personas.append(cluster[0])
                
            print(f"{dataset_name} results:")
            print(f"  Original: {original_count} personas")
            print(f"  After removing redundancy: {len(unique_personas)} personas")
            print(f"  Reduced by: {original_count - len(unique_personas)} personas ({((original_count - len(unique_personas)) / original_count * 100):.2f}%)")
            print(f"  Processing time: {time.time() - start_time:.2f} seconds")
            
            return unique_personas
                
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            return []
    
    def save_intermediate_results(self, personas, filename="unique_personas_partial.json"):
        """Save intermediate results to a JSON file"""
        print(f"Saving {len(personas)} unique personas to {filename}")
        with open(filename, "w") as f:
            json.dump(personas, f, indent=2)
    
    def load_intermediate_results(self, filename="unique_personas_partial.json"):
        """Load intermediate results from a JSON file"""
        if os.path.exists(filename):
            print(f"Loading intermediate results from {filename}")
            with open(filename, "r") as f:
                return json.load(f)
        return []
    
    def get_processed_datasets(self, filename="processed_datasets.json"):
        """Get list of already processed datasets"""
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return json.load(f)
        return []
    
    def save_processed_datasets(self, datasets, filename="processed_datasets.json"):
        """Save list of processed datasets"""
        with open(filename, "w") as f:
            json.dump(datasets, f, indent=2)
    
    def run_per_dataset_deduplication(self, similarity_threshold=0.85):
        """Process each dataset independently, removing redundancy within each"""
        print("Starting per-dataset deduplication process")
        overall_start_time = time.time()
        
        # Get processed datasets if resuming
        processed_datasets = self.get_processed_datasets()
        
        # Load existing results if resuming
        final_personas = self.load_intermediate_results()
        
        # Get all dataset names
        dataset_names = list(self.persona_dataset.config.keys())
        print(f"Found {len(dataset_names)} datasets: {', '.join(dataset_names)}")
        
        # Show which datasets were already processed if resuming
        if processed_datasets:
            print(f"Already processed datasets: {', '.join(processed_datasets)}")
            print(f"Loaded {len(final_personas)} previously processed personas")
        
        # Process each dataset independently
        for dataset_name in dataset_names:
            if dataset_name in processed_datasets:
                print(f"Skipping already processed dataset: {dataset_name}")
                continue
                
            dataset_personas = self.process_dataset(dataset_name, similarity_threshold)
            final_personas.extend(dataset_personas)
            
            # Save progress after each dataset
            processed_datasets.append(dataset_name)
            self.save_processed_datasets(processed_datasets)
            self.save_intermediate_results(final_personas)
            
            # Force garbage collection
            gc.collect()
        
        print("\n===== Final Results =====")
        print(f"Total unique personas after per-dataset processing: {len(final_personas)}")
        print(f"Total processing time: {time.time() - overall_start_time:.2f} seconds")
        
        return final_personas

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process persona embeddings and perform deduplication')
    parser.add_argument('--threshold', type=float, default=0.99, help='Similarity threshold for deduplication')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2', help='Sentence transformer model to use')
    args = parser.parse_args()
    
    # Create merger
    merger = EmbeddingPersonaMerger(model_name=args.model)
    
    # Run per-dataset deduplication pipeline
    unique_personas = merger.run_per_dataset_deduplication(
        similarity_threshold=args.threshold, 
    )
    
    # Save results
    print(f"\nSaving {len(unique_personas)} unique personas to unique_personas.json")
    with open("unique_personas.json", "w") as f:
        json.dump(unique_personas, f, indent=2)
    
    # Clean up intermediate files if successful
    if os.path.exists("processed_datasets.json") and os.path.exists("unique_personas_partial.json"):
        print("Cleaning up intermediate files")
        try:
            os.remove("processed_datasets.json")
            os.remove("unique_personas_partial.json")
        except Exception as e:
            print(f"Error cleaning up: {e}")
    
    print("Done!")
