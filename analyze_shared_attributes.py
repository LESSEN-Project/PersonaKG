import os
from knowledge_graph import KnowledgeGraph
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

class SharedAttributeAnalyzer:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        
    def get_attribute_persona_map(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        Get a mapping of categories to {attribute: set of persona_ids}.
        This shows which personas share each attribute.
        """
        with self.kg.driver.session() as session:
            # Get all personas first
            result = session.run("MATCH (p:Persona) RETURN p.id as persona_id")
            self.all_personas = {record["persona_id"] for record in result}
            
            # Get all attribute relationships
            result = session.run("""
                MATCH (p:Persona)-[r]->(a:Attribute)
                RETURN p.id as persona_id, a.category as category, 
                       a.value as value, a.key as key
            """)
            
            # Structure: category -> {attribute -> set(persona_ids)}
            attribute_map = defaultdict(lambda: defaultdict(set))
            
            for record in result:
                category = record["category"]
                value = record["value"]
                key = record["key"]
                persona_id = record["persona_id"]
                
                # For demographics, include the key in the attribute
                if category == "demographics" and key:
                    attribute = f"{key}: {value}"
                else:
                    attribute = value
                    
                attribute_map[category][attribute].add(persona_id)
            
            return attribute_map
    
    def analyze_sharing_patterns(self):
        """Analyze and print sharing patterns for attributes."""
        attribute_map = self.get_attribute_persona_map()
        total_personas = len(self.all_personas)
        
        print("\nAttribute Sharing Analysis")
        print("========================")
        print(f"\nTotal number of personas in database: {total_personas}")
        
        for category in sorted(attribute_map.keys()):
            print(f"\n{category}:")
            print("-" * len(category))
            
            # Get all attributes and their sharing counts
            sharing_counts = {
                attr: len(personas)
                for attr, personas in attribute_map[category].items()
            }
            
            if not sharing_counts:
                print("  No attributes found")
                continue
            
            # Calculate statistics
            total_attributes = len(sharing_counts)
            unique_attributes = sum(1 for count in sharing_counts.values() if count == 1)
            shared_attributes = sum(1 for count in sharing_counts.values() if count > 1)
            
            # Find personas with no attributes in this category
            personas_with_attributes = set().union(*attribute_map[category].values())
            personas_without_attributes = self.all_personas - personas_with_attributes
            
            # Calculate sharing statistics
            max_shared = max(sharing_counts.values()) if sharing_counts else 0
            avg_shared = sum(sharing_counts.values()) / len(sharing_counts) if sharing_counts else 0
            
            print(f"  Total attributes: {total_attributes}")
            print(f"  Unique attributes (used by only one persona): {unique_attributes} ({unique_attributes/total_attributes*100:.1f}%)")
            print(f"  Shared attributes (used by multiple personas): {shared_attributes} ({shared_attributes/total_attributes*100:.1f}%)")
            print(f"  Personas with no attributes in this category: {len(personas_without_attributes)} ({len(personas_without_attributes)/total_personas*100:.1f}%)")
            print(f"  Most shared attribute appears in: {max_shared} personas")
            print(f"  Average sharing per attribute: {avg_shared:.1f} personas")
            
            # Distribution of sharing
            print("\n  Sharing distribution:")
            sharing_dist = Counter(sharing_counts.values())
            for num_personas, num_attrs in sorted(sharing_dist.items()):
                print(f"    {num_attrs} attributes are shared by {num_personas} personas")

def main():
    # Initialize KnowledgeGraph
    neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
    kg = KnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password=neo4j_password)
    
    try:
        analyzer = SharedAttributeAnalyzer(kg)
        analyzer.analyze_sharing_patterns()
    finally:
        kg.close()

if __name__ == "__main__":
    main()
