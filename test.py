import os

from dataset import get_dataset, get_personas
from models import LLM
from knowledge_graph import KnowledgeGraph

def kg_prompt():

    return [{
        "role": "system",
        "content": """You are a persona attribute extraction system. You will be given a persona’s text.
        Your task is to extract the persona’s attributes and return them in a strict JSON format, without any additional keys or text. Follow these instructions exactly:

        1. **Schema**: You must produce valid JSON with the following top-level keys:
        - demographics
        - socialConnections
        - personalityTraits
        - interestsAndHobbies
        - skillsAndAbilities
        - preferencesAndFavorites
        - goalsAndAspirations
        - beliefsAndValues
        - behavioralPatterns
        - additionalAttributes

        2. **Allowed Fields**:
        - For `demographics`, include the following fields:
            - "age"
            - "employmentStatus"
            - "educationStatus"
            - "livingSituation"
            - "placeGrewUp"  (e.g., hometown or region where the person grew up)
            If a field is not mentioned in the persona text, leave it blank (i.e., an empty string).
        - For `socialConnections`, use an array of strings to capture any significant social or early life connections (e.g., meeting a best friend in kindergarten).
        - For all the other keys (personalityTraits, interestsAndHobbies, skillsAndAbilities, preferencesAndFavorites, goalsAndAspirations, beliefsAndValues, behavioralPatterns), use arrays of strings. If the persona text doesn’t mention anything relevant, use an empty array (`[]`).
        - For `additionalAttributes`, capture any extra information or variables mentioned in the persona text that do not clearly fit into any of the above categories.

        3. **No Extra Text**: Do not add any commentary or explanation. Return **only** the JSON.

        4. **Example**  
        Input Persona Text:
        """
    },
    {
        "role": "user",
        "content": "{persona}"
    }]

neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
kg = KnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password=neo4j_password)

persona_kg_extractor = LLM("GPT-4o", default_prompt=kg_prompt())
dataset = get_dataset()
personas = get_personas(dataset, "test")

persona = personas[10]
print(persona)

res = persona_kg_extractor.generate(prompt_params={"persona": persona})
print(res)