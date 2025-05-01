import os
import hashlib
import random

from dataset import get_dataset, get_personas
from models import LLM
from knowledge_graph import KnowledgeGraph
from prompts import *


def main():

    # -------------------------------------------------------------------------
    # CUSTOMIZE YOUR SCHEMA HERE
    # -------------------------------------------------------------------------
    # Format: [category_name, fields_list]
    # - category_name: String name for the category
    # - fields_list: For demographics/basic categories, provide a list of fields
    #               For other categories, use None
    # 
    # Example 1: Basic professional schema
    schema = [
        ["demographics", ["age", "employmentStatus", "educationStatus", "location", "occupation"]],
        ["personality", None],
        ["hobbies", None],
        ["interests", None],
        ["projects", None]
    ]
    
    # Example 2: Customer profile schema
    # schema = [
    #     ["profile", ["age", "location", "income", "household"]],
    #     ["behaviors", None],
    #     ["preferences", None],
    #     ["purchaseHistory", None],
    #     ["customerService", None]
    # ]
    
    # Example 3: Character development schema
    # schema = [
    #     ["basics", ["age", "appearance", "background", "origin"]],
    #     ["traits", None],
    #     ["motivations", None],
    #     ["relationships", None],
    #     ["skills", None],
    #     ["story", None]
    # ]
    # -------------------------------------------------------------------------
    
    # Set to True to rebuild database when schema changes
    force_rebuild = True
    
    # Number of personas to process
    num_personas = 20
    
    # Connect to Neo4j and initialize KG with custom schema
    neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
    kg = KnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password=neo4j_password, schema=schema)
    
    # Check if database needs to be rebuilt and optionally force it
    if force_rebuild:
        print("Forcing database rebuild...")
        kg.drop_database()
    
    # Initialize LLMs with prompts based on the custom schema
    persona_kg_extractor = LLM("GPT-4.1", default_prompt=kg_prompt(schema=schema))
    persona_canonicalizer = LLM("GPT-4.1", default_prompt=canonicalization_prompt())
    
    # Get personas from dataset
    dataset = get_dataset()
    personas = get_personas(dataset, "test")
    selected_personas = random.sample(personas, min(num_personas, len(personas)))
    
    # Display current schema information
    print("Using schema with these categories:")
    for category in schema:
        if len(category) > 1 and category[1]:
            print(f"- {category[0]} (with fields: {category[1]})")
        else:
            print(f"- {category[0]}")
    print()
    
    # Process personas with the custom schema
    for i, persona in enumerate(selected_personas, 1):
        persona_id = str(hashlib.sha256(persona.encode('utf-8')).hexdigest())
        
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
                import json
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
        except Exception as e:
            print(f"Error processing persona: {str(e)}")
            print(f"Skipping persona and continuing...")
            continue
        print(f"Processed persona {persona_id[:8]}\n")
    
    print("All personas processed successfully!")
    print("The knowledge graph now contains the personas with the new schema.")
    print("You can query the Neo4j database to explore the results.")


if __name__ == "__main__":
    main()