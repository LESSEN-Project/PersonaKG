import os
import hashlib
import random

from dataset import get_dataset, get_personas
from models import LLM
from knowledge_graph import KnowledgeGraph
from prompts import *

neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
kg = KnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password=neo4j_password)

persona_kg_extractor = LLM("GPT-4.1", default_prompt=kg_prompt())
persona_canonicalizer = LLM("GPT-4.1", default_prompt=canonicalization_prompt())

dataset = get_dataset()
personas = get_personas(dataset, "test")

selected_personas = random.sample(personas, 10)

for persona in selected_personas:
    persona_id = str(hashlib.sha256(persona.encode('utf-8')).hexdigest())
    
    print(f"Processing persona\n{persona}")
    res = persona_kg_extractor.generate(prompt_params={"persona": persona}, json_output=True)
    print(res)
    
    attributes = kg.get_existing_attributes()
    canonized_res = persona_canonicalizer.generate(prompt_params={"existing_attributes": attributes, "persona_json": res}, json_output=True)
    
    kg.upsert_persona(canonized_res, persona_id)