import os
from knowledge_graph import KnowledgeGraph
from collections import defaultdict
from difflib import SequenceMatcher
import re
from typing import Dict, List, Tuple, Set

def tokenize(text: str) -> Set[str]:
    """Convert text to lowercase and split into words, removing special characters."""
    return set(re.findall(r'\w+', text.lower()))

def word_overlap_ratio(text1: str, text2: str) -> float:
    """Calculate word overlap ratio between two strings."""
    words1 = tokenize(text1)
    words2 = tokenize(text2)
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def string_similarity(text1: str, text2: str) -> float:
    """Calculate string similarity using SequenceMatcher."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def find_similar_attributes(attributes: Dict[str, List[str]], 
                          word_overlap_threshold: float = 0.5,
                          string_sim_threshold: float = 0.7) -> Dict[str, List[Tuple[str, str, float, float]]]:
    """
    Find similar attributes within each category.
    Returns a dictionary of category -> list of (attr1, attr2, word_overlap, string_sim) tuples.
    """
    similar_pairs = defaultdict(list)
    
    for category, values in attributes.items():
        # Skip demographics as they are usually more structured
        if category == "demographics":
            continue
            
        for i, attr1 in enumerate(values):
            for attr2 in values[i+1:]:
                word_overlap = word_overlap_ratio(attr1, attr2)
                string_sim = string_similarity(attr1, attr2)
                
                # If either similarity measure is above threshold, consider them similar
                if word_overlap >= word_overlap_threshold or string_sim >= string_sim_threshold:
                    similar_pairs[category].append((attr1, attr2, word_overlap, string_sim))
    
    return similar_pairs

def print_similarity_statistics(attributes: Dict[str, List[str]], similar_pairs: Dict[str, List[Tuple[str, str, float, float]]]):
    """Print statistics about the proportion of similar attributes."""
    print("\nSimilarity Statistics")
    print("====================")
    
    total_attributes = 0
    total_similar = 0
    
    for category, values in attributes.items():
        if category == "demographics":
            continue
            
        num_attributes = len(values)
        total_attributes += num_attributes
        
        # Get unique attributes involved in similar pairs
        similar_attrs = set()
        if category in similar_pairs:
            for attr1, attr2, _, _ in similar_pairs[category]:
                similar_attrs.add(attr1)
                similar_attrs.add(attr2)
        
        num_similar = len(similar_attrs)
        total_similar += num_similar
        
        similarity_percentage = (num_similar / num_attributes * 100) if num_attributes > 0 else 0
        
        print(f"\n{category}:")
        print(f"  Total attributes: {num_attributes}")
        print(f"  Attributes with similarities: {num_similar}")
        print(f"  Percentage similar: {similarity_percentage:.1f}%")
        if category in similar_pairs:
            print(f"  Number of similar pairs: {len(similar_pairs[category])}")
    
    overall_percentage = (total_similar / total_attributes * 100) if total_attributes > 0 else 0
    print(f"\nOverall Statistics:")
    print(f"  Total attributes (excluding demographics): {total_attributes}")
    print(f"  Total attributes with similarities: {total_similar}")
    print(f"  Overall percentage similar: {overall_percentage:.1f}%")

def main():
    # Initialize KnowledgeGraph
    neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
    kg = KnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password=neo4j_password)
    
    # Get all attributes
    attributes = kg.get_existing_attributes()
    
    # Find similar attributes
    similar_pairs = find_similar_attributes(attributes)
    
    # Print similarity statistics
    print_similarity_statistics(attributes, similar_pairs)
    
    # Print detailed results
    print("\nDetailed Similar Pairs Analysis")
    print("==============================")
    
    for category, pairs in similar_pairs.items():
        if pairs:
            print(f"\n{category}:")
            print("-" * len(category))
            
            # Sort by combined similarity score (word overlap + string similarity)
            pairs.sort(key=lambda x: x[2] + x[3], reverse=True)
            
            for attr1, attr2, word_overlap, string_sim in pairs:
                print(f"\nPair:")
                print(f"  1: {attr1}")
                print(f"  2: {attr2}")
                print(f"  Word Overlap: {word_overlap:.2f}")
                print(f"  String Similarity: {string_sim:.2f}")

if __name__ == "__main__":
    main()
