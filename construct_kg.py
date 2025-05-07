import os
import hashlib
import random
import json
import pathlib

from dataset import get_dataset, get_personas
from models import LLM
from knowledge_graph import KnowledgeGraph
from prompts import *


def generate_schema_hash(schema):
    """Generate a unique hash for the schema configuration"""
    schema_str = json.dumps(schema, sort_keys=True)
    return hashlib.md5(schema_str.encode()).hexdigest()

def check_db_schema_match(kg, schema):
    """Check if the database schema matches our schema configuration"""
    try:
        # Get current database schema info
        current_info = kg.get_current_schema_info()
        
        # If there's no data in the database, it's a fresh DB
        if not current_info["categories"]:
            return False
            
        # Extract schema categories and demographic fields
        new_categories = set(config[0] for config in schema)
        new_demographic_fields = set()
        for config in schema:
            if config[0] == "demographics" and len(config) > 1 and config[1]:
                new_demographic_fields = set(config[1])
        
        # Check for significant changes
        categories_match = set(current_info["categories"]) == new_categories
        demographic_fields_match = set(current_info["demographic_fields"]) == new_demographic_fields
        
        return categories_match and demographic_fields_match
    except Exception as e:
        print(f"Error checking schema match: {str(e)}")
        return False

def main():

    # -------------------------------------------------------------------------
    # CUSTOMIZE YOUR SCHEMA HERE
    # -------------------------------------------------------------------------
    # Format: [category_name, fields_list]
    # - category_name: String name for the category
    # - fields_list: For demographics/basic categories, provide a list of fields
    #               For other categories, use None
    # 
    schema = [
        ["demographics", ["age", "employmentStatus", "educationStatus", "location", "occupation"]],
        ["personality", None],
        ["hobbies", None],
        ["interests", None],
        ["projects", None]
    ]
    
    # Example: Customer profile schema
    # schema = [
    #     ["profile", ["age", "location", "income", "household"]],
    #     ["behaviors", None],
    #     ["preferences", None],
    #     ["purchaseHistory", None],
    #     ["customerService", None]
    # ]
    
    # Example: Character development schema
    # schema = [
    #     ["basics", ["age", "appearance", "background", "origin"]],
    #     ["traits", None],
    #     ["motivations", None],
    #     ["relationships", None],
    #     ["skills", None],
    #     ["story", None]
    # ]
    # -------------------------------------------------------------------------
    
    # Set to False to use a subset of personas, True to use the entire dataset
    use_whole_dataset = True
    
    # Number of personas to process if not using the whole dataset
    num_personas = 10
    
    # Generate a unique hash for the current schema
    schema_hash = generate_schema_hash(schema)
    
    # Directory for saving results
    results_dir = pathlib.Path("saved_results")
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir / f"canonized_results_{schema_hash}.json"
    print(f"Using results file: {results_file}")
    
    # Determine if we need to force rebuild - default to True, but may change based on saved results
    force_rebuild = True
    
    # Previously processed personas (persona_id → canonized_result)
    processed_personas = {}
    
    # Default to 0 for the last processed index
    last_processed_index = 0
    
    # Check if we have saved results for the current schema and using whole dataset
    if use_whole_dataset and results_file.exists():
        try:
            print(f"Found saved results for the current schema hash: {schema_hash}")
            with open(results_file, 'r') as f:
                saved_data = json.load(f)
                processed_personas = saved_data.get('processed_personas', {})
                last_processed_index = saved_data.get('last_processed_index', 0)
                
            print(f"Loaded {len(processed_personas)} previously processed personas")
            # Only set force_rebuild to False if we successfully loaded data
            force_rebuild = False
            
        except Exception as e:
            print(f"Error loading saved results: {str(e)}")
            print("Starting fresh with a forced rebuild")
            force_rebuild = True
            processed_personas = {}
            last_processed_index = 0
    else:
        # No saved results or not using whole dataset
        if use_whole_dataset:
            print(f"No saved results found for schema hash: {schema_hash}")
            print("Starting fresh with a forced rebuild")
        else:
            print("Using random sample of personas with forced rebuild")

    # Connect to Neo4j and initialize KG with custom schema
    neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
    
    # First initialize with just the connection, without providing schema yet
    kg = KnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password=neo4j_password)
    
    # Now manually handle database rebuild if needed instead of letting KG do it automatically
    if force_rebuild:
        print("Forcing database rebuild...")
        kg.drop_database()
    
    # After potential rebuild, update the schema but explicitly pass our force_rebuild flag
    # This will prevent the KnowledgeGraph from making its own decision to rebuild
    # Use the skip_schema_check=True when we've loaded valid saved results and don't want to force rebuild
    kg.update_schema(schema, force_rebuild=False, skip_schema_check=(not force_rebuild))
    
    # Initialize LLMs with prompts based on the custom schema
    persona_kg_extractor = LLM("GPT-4.1-mini", default_prompt=kg_prompt(schema=schema))
    persona_canonicalizer = LLM("GPT-4.1-mini", default_prompt=canonicalization_prompt())
    
    # Get personas from dataset
    dataset = get_dataset()
    personas = get_personas(dataset, "train")
    
    # Choose either the whole dataset or a random subset
    if use_whole_dataset:
        selected_personas = personas
        print(f"Using the entire dataset: {len(selected_personas)} personas")
    else:
        selected_personas = random.sample(personas, min(num_personas, len(personas)))
        print(f"Using a random sample of {len(selected_personas)} personas")
    
    # Display current schema information
    print("Using schema with these categories:")
    for category in schema:
        if len(category) > 1 and category[1]:
            print(f"- {category[0]} (with fields: {category[1]})")
        else:
            print(f"- {category[0]}")
    print()
    
    # Process personas with the custom schema
    for i, persona in enumerate(selected_personas[last_processed_index:], last_processed_index + 1):
        persona_id = str(hashlib.sha256(persona.encode('utf-8')).hexdigest())
        
        # Skip if this persona was already processed (for resuming interrupted processing)
        if persona_id in processed_personas and use_whole_dataset:
            print(f"Skipping already processed persona {i}/{len(selected_personas)}: {persona_id[:8]}...")
            continue
            
        print(f"Processing persona {i}/{len(selected_personas)}:\n{persona}...")
        
        # Generate structured attributes from the persona
        res = persona_kg_extractor.generate(prompt_params={"persona": persona}, json_output=True)
        print("Extracted categories:", list(res.keys()))
        
        # Canonicalize attributes
        attributes = kg.get_existing_attributes()
        try:
            # First try getting the canonicalized result
            canonized_res = persona_canonicalizer.generate(
                prompt_params={"existing_attributes": attributes, "persona_json": res}, 
                json_output=True
            )
            # If we received a string, we need to handle parsing it
            if isinstance(canonized_res, str):
                print(f"Received string response, attempting to parse as JSON...")
                import ast
                try:
                    # First try standard JSON parsing
                    canonized_res = json.loads(canonized_res)
                except json.JSONDecodeError as e:
                    print(f"Standard JSON parsing failed: {e}")
                    
                    # If it fails, try to fix Python dict syntax (single quotes to double quotes)
                    try:
                        # Use ast to safely evaluate the Python literal
                        python_dict = ast.literal_eval(canonized_res)
                        # Convert to JSON-compatible format
                        canonized_res = json.loads(json.dumps(python_dict))
                        print("Successfully converted Python dict to JSON")
                    except Exception as e2:
                        print(f"Cannot parse response as either JSON or Python dict: {e2}")
                        print("Skipping this persona due to parsing issues")
                        continue
            
            # If we get here, we have a valid JSON object to save
            kg.upsert_persona(canonized_res, persona_id)
            
            # If using the whole dataset, save progress
            if use_whole_dataset:
                processed_personas[persona_id] = canonized_res
                # Save progress every 5 personas or when we're at the end
                if i % 5 == 0 or i == len(selected_personas):
                    print(f"Saving progress at persona {i}/{len(selected_personas)}...")
                    try:
                        with open(results_file, 'w') as f:
                            json.dump({
                                'processed_personas': processed_personas,
                                'last_processed_index': i - 1,
                                'schema_hash': schema_hash,
                                'schema': schema
                            }, f)
                        print(f"Saved progress to {results_file}")
                    except Exception as e:
                        print(f"Error saving progress: {str(e)}")
                        
        except Exception as e:
            print(f"Error processing persona: {str(e)}")
            print(f"Skipping persona and continuing...")
            continue
        print(f"Processed persona {persona_id[:8]}\n")
    
    print("All personas processed successfully!")
    
    # Final save to ensure all results are stored
    if use_whole_dataset:
        try:
            with open(results_file, 'w') as f:
                json.dump({
                    'processed_personas': processed_personas,
                    'last_processed_index': len(selected_personas) - 1,
                    'schema_hash': schema_hash,
                    'schema': schema,
                    'completed': True
                }, f)
            print(f"Saved final results to {results_file}")
        except Exception as e:
            print(f"Error saving final results: {str(e)}")
    
    print("The knowledge graph now contains the personas with the new schema.")
    print("You can query the Neo4j database to explore the results.")


if __name__ == "__main__":
    main()