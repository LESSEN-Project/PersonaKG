def get_cluster_themes():
    return [{
        "role": "system",
        "content": """You are a theme extraction system."""
        },
        {
            "role": "user",
            "content": """Statements coming from user personas are analyzed and clustered into groups. The statements correspond to the personal information shared by the users.
            Your task is to identify the underlying themes of the given cluster statements. These themes are going to be used to create a knowledge graph to connect the users.      
            Each cluster can have a theme or themes. If not one single theme exists, limit the number of themes to 3. If there are more than 3 themes, it means the cluster
            is not a valid cluster. Each theme can have subthemes. You are going to output a json object with the themes. Make each theme a key and the subthemes its values.
            If no theme exists, output "None". Do not include multiple themes that have similar semantic meanings. Your json object should follows this structire:
            {{"theme1": ["subtheme1", "subtheme2"], "theme2": ["subtheme3"], ...}}
            
            Here are the cluster statements:
            {statements}"""
        }
    ]


def canonicalization_prompt():
    """Generate a prompt for canonicalizing persona attributes.
    
    Returns:
        list: The prompt messages for the LLM.
    """
    system_content = """
        You are a persona attribute canonicalization system. """
    
    user_content = """You will receive persona attributes extracted from text.
        Your task is to normalize and canonicalize these attributes to ensure consistency in the knowledge graph.
        
        As input, you will receive:
        1. A JSON representation of a persona's attributes
        2. Existing normalized attributes already in the knowledge graph
        
        OUTPUT REQUIREMENTS (EXTREMELY IMPORTANT):
        1. You MUST output valid JSON with double quotes (not single quotes) for both keys and values.
        2. Your JSON must have the exact same structure and category names as the input JSON.
        3. Do not add any explanatory text before or after the JSON.
        4. Output ONLY the clean JSON object with double quotes, nothing else.
        5. Do not use single quotes in your JSON output. Always use double quotes.
        
        Important: Failure to output a valid JSON with double quotes will cause system errors.
        Here's the existing attributes in our knowledge graph: {existing_attributes}
        Here's the new persona attributes: {persona_json}
        Please canonicalize the new persona attributes, ensuring they are in the cleanest, most normalized form possible."""
    
    return [{
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }]

def kg_prompt(schema, persona):
    """Generate a prompt for extracting persona attributes based on a schema.

    Args:
        schema (dict): A custom schema for the knowledge graph in JSON format.
        persona (dict): A persona dictionary containing the persona text.

    Returns:
        list: The prompt messages for the LLM.
    """

    # Create the system content as a single non-formatted string to avoid issues
    system_content = f"You are a persona attribute extraction system. Use this schema to extract the persona's attributes.\n\n{schema}"

    user_content = f"""You will be given a persona's text.
        Your task is to extract the persona's attributes and return them in a strict JSON format, without any additional keys or text. 
        Follow these instructions exactly:

        1. **Schema**: 
        - Follow the schema to extract the persona's attributes.
        - Do not use any nodes or relationships that doesn't exist in the schema.
        - All values in the schema are optional.

        2. **Persona Extraction**:
        - Each extracted value must appear in **only one** top-level category.
        - Choose the **most appropriate** category based on context.
        - Some values can be included in multiple categories, but make sure to use the same normalized value in all categories.
        - Do not duplicate the same value inside a single category.
        - Do not extract anything related to the current emotional state of the person.
        - Ignore statements that are not realistic. (E.g. I am the queen of England, I am an alien.)
        - If a value is null, meaning that it is not mentioned in the text, do not include it in the output.
        - Some values can be inferred from the text. For example, the strength of a person's preference.
        - If there are conflicting information, choose the most likely one.

        3. **JSON FORMAT REQUIREMENTS** (EXTREMELY IMPORTANT):
        - You MUST use double quotes (not single quotes) for ALL keys and string values.
        - Your output must be a valid JSON object that can be parsed by json.loads().
        - Do not use single quotes for keys or string values.
        - Do not add any text before or after the JSON object.
        - Your response should contain ONLY the properly formatted JSON.

        4. **Attribute Normalization**:
        - All attributes should be normalized to their most concise, standardized form.
        - For example, instead of "likes going to the gym", use "goes to gym".
        - Always come up with the most normalized form that captures the core concept.
        
        Here's the persona text: {persona}"""

    return [{
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }]
