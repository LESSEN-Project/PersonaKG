from neo4j import GraphDatabase
import uuid

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        # Initialize the Neo4j driver.
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Define the allowed top-level categories in the JSON.
        self.allowed_categories = {
            "demographics",
            "socialConnections",
            "personalityTraits",
            "interestsAndHobbies",
            "skillsAndAbilities",
            "preferencesAndFavorites",
            "goalsAndAspirations",
            "beliefsAndValues",
            "behavioralPatterns",
            "additionalAttributes"
        }
        # Define the allowed fields for the demographics category.
        self.allowed_demographic_fields = {
            "age", "employmentStatus", "educationStatus", "livingSituation", "placeGrewUp"
        }
        
    def close(self):
        self.driver.close()

    def create_schema(self):
        """
        Creates constraints to ensure that each Persona has a unique id and that
        each Attribute is uniquely identified by its category (and key, if applicable) plus value.
        This helps prevent unrelated or duplicate nodes from being inserted.
        """
        with self.driver.session() as session:
            # Constraint for Persona nodes: each must have a unique id.
            session.run("CREATE CONSTRAINT IF NOT EXISTS ON (p:Persona) ASSERT p.id IS UNIQUE")
            
            # Constraint for Attribute nodes: ensure uniqueness on the combination of category and value.
            # For demographic attributes, we also include the 'key' property.
            # (Adjust the syntax based on your Neo4j version.)
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS ON (a:Attribute) 
                ASSERT (a.category, a.value) IS NODE KEY
            """)
    
    def validate_json(self, persona_json):
        """
        Validates that the input JSON only contains allowed categories and,
        for demographics, only allowed demographic fields.
        """
        # Check that all top-level keys are allowed.
        for key in persona_json:
            if key not in self.allowed_categories:
                raise ValueError(f"Unrecognized category in JSON: '{key}'. Allowed categories: {self.allowed_categories}")
        # Validate that the demographics object contains only allowed fields.
        demographics = persona_json.get("demographics", {})
        for field in demographics:
            if field not in self.allowed_demographic_fields:
                raise ValueError(f"Unrecognized demographic field: '{field}'. Allowed fields: {self.allowed_demographic_fields}")

    def insert_persona(self, persona_json):
        """
        Inserts the persona JSON into the Neo4j database as a Persona node and
        creates Attribute nodes for each of the attributes. Each attribute node is
        connected to the Persona node via a relationship labeled with the category.
        """
        # Validate the JSON against the defined schema.
        self.validate_json(persona_json)
        # Generate a unique identifier for this Persona.
        persona_id = str(uuid.uuid4())
        
        with self.driver.session() as session:
            # Create the Persona node.
            session.run("CREATE (p:Persona {id: $id})", id=persona_id)
            
            # Process demographics (which is a dictionary).
            demographics = persona_json.get("demographics", {})
            for field, value in demographics.items():
                if value.strip():
                    session.run(
                        """
                        MERGE (a:Attribute {category: 'demographics', key: $field, value: $value})
                        WITH a
                        MATCH (p:Persona {id: $persona_id})
                        MERGE (p)-[:HAS_ATTRIBUTE {category: 'demographics'}]->(a)
                        """,
                        field=field,
                        value=value.strip(),
                        persona_id=persona_id
                    )
            
            # Process other categories (which are lists of strings).
            for category in self.allowed_categories:
                # Skip demographics since it's already processed.
                if category == "demographics":
                    continue
                items = persona_json.get(category, [])
                for item in items:
                    if item.strip():
                        session.run(
                            """
                            MERGE (a:Attribute {category: $category, value: $value})
                            WITH a
                            MATCH (p:Persona {id: $persona_id})
                            MERGE (p)-[:HAS_ATTRIBUTE {category: $category}]->(a)
                            """,
                            category=category,
                            value=item.strip(),
                            persona_id=persona_id
                        )
        
        return persona_id


