def canonicalization_prompt():

    return [{
            "role": "system",
            "content": """You are a canonicalization system. 
            You will be given:
            1) A JSON of persona attributes.
            2) A list of existing attributes for each category.

            Your task: For each attribute in the persona JSON, see if it matches or is semantically similar to an existing attribute in the same category. If yes, replace it with the existing attribute string exactly. If not, leave it as-is.

            Existing attributes:
            {existing_attributes}

            Output must be valid JSON with the same schema and keys, but with attributes replaced by their canonical forms if found.
            Do not add extra keys or text.
            """
            },
            {
                "role": "user",
                "content": """{persona_json}"""
            }
        ]

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
            - "currentLocation" (where they currently live)
            - "pets" (any pets they own)
            - "allergies" (any allergies they have)
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