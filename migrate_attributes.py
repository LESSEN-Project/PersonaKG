import os
from knowledge_graph import KnowledgeGraph
import re
from typing import Dict, List, Tuple

def find_location_attributes(attributes: List[str]) -> List[str]:
    """Find attributes that describe current location."""
    location_patterns = [
        r"lives? in .+",
        r"living in .+",
        r"currently (?:lives?|residing) in .+",
        r"based in .+",
        r"located in .+"
    ]
    return [attr for attr in attributes if any(re.search(pattern, attr.lower()) for pattern in location_patterns)]

def find_pet_attributes(attributes: List[str]) -> List[str]:
    """Find attributes that describe pet ownership."""
    pet_patterns = [
        r"has (?:a |one |two |three |four |five |\d+ )?(?:cat|dog|pet|bird|fish|hamster|rabbit)s?",
        r"owns? (?:a |one |two |three |four |five |\d+ )?(?:cat|dog|pet|bird|fish|hamster|rabbit)s?"
    ]
    return [attr for attr in attributes if any(re.search(pattern, attr.lower()) for pattern in pet_patterns)]

def find_allergy_attributes(attributes: List[str]) -> List[str]:
    """Find attributes that describe allergies."""
    allergy_patterns = [
        r"allergic to .+",
        r"has (?:a |an )?allerg(?:y|ies) to .+"
    ]
    return [attr for attr in attributes if any(re.search(pattern, attr.lower()) for pattern in allergy_patterns)]

def extract_location(attr: str) -> str:
    """Extract the location from a location attribute."""
    # Common patterns for location extraction
    patterns = [
        r"lives? in (.+)",
        r"living in (.+)",
        r"currently (?:lives?|residing) in (.+)",
        r"based in (.+)",
        r"located in (.+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, attr.lower())
        if match:
            return match.group(1).strip().title()
    return attr

def extract_pets(attr: str) -> str:
    """Extract pet information in a standardized format."""
    # Extract numbers and pet types
    numbers = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"}
    number_pattern = r"(?:a |one |two |three |four |five |\d+ )"
    pet_pattern = r"(?:cat|dog|pet|bird|fish|hamster|rabbit)s?"
    
    attr = attr.lower()
    match = re.search(f"(?:has|owns) ({number_pattern})?({pet_pattern}s?)", attr)
    if match:
        number = match.group(1).strip() if match.group(1) else "1"
        pet_type = match.group(2)
        
        # Standardize number words to digits
        number = numbers.get(number, number)
        
        # Make pet type singular if number is 1
        if number == "1":
            pet_type = pet_type.rstrip('s')
            
        return f"{number} {pet_type}"
    return attr

def extract_allergy(attr: str) -> str:
    """Extract allergy information in a standardized format."""
    match = re.search(r"allergic to (.+)", attr.lower())
    if match:
        return f"Allergic to {match.group(1).strip().title()}"
    return attr

def migrate_attributes(kg: KnowledgeGraph):
    """Migrate relevant attributes from additionalAttributes to demographics."""
    # Get all existing attributes
    attributes = kg.get_existing_attributes()
    additional_attrs = attributes.get("additionalAttributes", [])
    
    # Find attributes to migrate
    location_attrs = find_location_attributes(additional_attrs)
    pet_attrs = find_pet_attributes(additional_attrs)
    allergy_attrs = find_allergy_attributes(additional_attrs)
    
    print(f"Found attributes to migrate:")
    print(f"Locations: {len(location_attrs)}")
    print(f"Pets: {len(pet_attrs)}")
    print(f"Allergies: {len(allergy_attrs)}")
    
    # Execute migration using Neo4j transaction
    with kg.driver.session() as session:
        # For each persona with these attributes
        for attr_type, attrs, new_field, extract_func in [
            ("location", location_attrs, "currentLocation", extract_location),
            ("pet", pet_attrs, "pets", extract_pets),
            ("allergy", allergy_attrs, "allergies", extract_allergy)
        ]:
            if not attrs:
                continue
                
            print(f"\nMigrating {attr_type} attributes...")
            for old_attr in attrs:
                # Get all personas with this attribute
                result = session.run("""
                    MATCH (p:Persona)-[r:HAS_ADDITIONAL]->(a:Attribute {value: $value})
                    RETURN p.id as persona_id
                """, value=old_attr)
                
                personas = [record["persona_id"] for record in result]
                if not personas:
                    continue
                
                # Create new standardized value
                new_value = extract_func(old_attr)
                print(f"  Converting '{old_attr}' to '{new_value}'")
                
                # For each persona, create new demographic relationship and remove old one
                for persona_id in personas:
                    # Create new demographic attribute
                    session.run("""
                        MERGE (a:Attribute {category: 'demographics', key: $key, value: $value})
                        WITH a
                        MATCH (p:Persona {id: $persona_id})
                        MERGE (p)-[:HAS_DEMOGRAPHIC]->(a)
                        WITH p
                        MATCH (p)-[r:HAS_ADDITIONAL]->(old:Attribute {value: $old_value})
                        DELETE r
                        WITH old
                        MATCH (old)
                        WHERE NOT EXISTS (()-[]->(old))
                        DELETE old
                    """, key=new_field, value=new_value, persona_id=persona_id, old_value=old_attr)

def main():
    # Initialize KnowledgeGraph
    neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
    kg = KnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password=neo4j_password)
    
    try:
        print("Starting attribute migration...")
        migrate_attributes(kg)
        print("\nMigration completed successfully!")
    finally:
        kg.close()

if __name__ == "__main__":
    main()
