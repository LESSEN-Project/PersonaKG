from neo4j import GraphDatabase
import hashlib

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
            "age",
            "employmentStatus",
            "educationStatus",
            "livingSituation",
            "placeGrewUp"
        }
        
        # Map each category to a specific relationship type in Neo4j.
        self.relationship_map = {
            "demographics": "HAS_DEMOGRAPHIC",
            "socialConnections": "HAS_CONNECTION",
            "personalityTraits": "HAS_PERSONALITY",
            "interestsAndHobbies": "HAS_INTEREST",
            "skillsAndAbilities": "HAS_SKILL",
            "preferencesAndFavorites": "HAS_PREFERENCE",
            "goalsAndAspirations": "HAS_GOAL",
            "beliefsAndValues": "HAS_BELIEF",
            "behavioralPatterns": "HAS_BEHAVIOR",
            "additionalAttributes": "HAS_ADDITIONAL"
        }

    def close(self):
        self.driver.close()

    def validate_json(self, persona_json):
        """
        Validates that the input JSON only contains allowed categories and,
        for demographics, only allowed demographic fields.
        """
        # Check that all top-level keys are allowed.
        for key in persona_json:
            if key not in self.allowed_categories:
                raise ValueError(
                    f"Unrecognized category in JSON: '{key}'. "
                    f"Allowed categories: {self.allowed_categories}"
                )
        
        # Validate that the demographics object contains only allowed fields.
        demographics = persona_json.get("demographics", {})
        for field in demographics:
            if field not in self.allowed_demographic_fields:
                raise ValueError(
                    f"Unrecognized demographic field: '{field}'. "
                    f"Allowed fields: {self.allowed_demographic_fields}"
                )

    def compute_attribute_id(self, category, value, key=None):
        """
        Compute a unique id for an attribute based on its category, and value.
        For demographics, a key is also included.
        """
        if key:
            identifier = f"{category}_{key}_{value}"
        else:
            identifier = f"{category}_{value}"
        return hashlib.sha256(identifier.encode('utf-8')).hexdigest()

    def get_existing_attributes(self):
        """
        Fetches all existing Attribute nodes in the graph, grouped by category.
        Returns a dict like: { "interestsAndHobbies": ["running", "reading"], ... }
        """
        query = """
        MATCH (a:Attribute)
        RETURN a.category AS category, a.value AS value
        ORDER BY a.category, a.value
        """
        existing = {}
        with self.driver.session() as session:
            results = session.run(query)
            for record in results:
                cat = record["category"]
                val = record["value"]
                if cat not in existing:
                    existing[cat] = []
                existing[cat].append(val)
        return existing

    def upsert_persona(self, persona_json, persona_id):
        """
        Upserts a Persona node using the given persona_id.
        If the Persona exists, update its attribute relationships (by deleting them first).
        Otherwise, create a new Persona node.
        Then, upsert attribute nodes and create relationships from the Persona to each Attribute.
        """
        self.validate_json(persona_json)
        
        with self.driver.session() as session:
            result = session.run("MATCH (p:Persona {id: $id}) RETURN p", id=persona_id).single()
            if result:
                # Update: remove all existing attribute relationships.
                session.run("MATCH (p:Persona {id: $id})-[r]->() DELETE r", id=persona_id)
            else:
                # Insert: create the Persona node.
                session.run("CREATE (p:Persona {id: $id})", id=persona_id)
            
            # Process demographics (dictionary)
            demographics = persona_json.get("demographics", {})
            rel_type = self.relationship_map["demographics"]
            for field, value in demographics.items():
                clean_value = value.strip()
                if clean_value:
                    attr_id = self.compute_attribute_id("demographics", clean_value, key=field)
                    session.run(
                        f"""
                        MERGE (a:Attribute {{id: $attr_id}})
                        ON CREATE SET a.category = 'demographics', a.key = $field, a.value = $value
                        ON MATCH SET a.category = 'demographics', a.key = $field, a.value = $value
                        WITH a
                        MATCH (p:Persona {{id: $id}})
                        MERGE (p)-[:{rel_type}]->(a)
                        """,
                        attr_id=attr_id, field=field, value=clean_value, id=persona_id
                    )
            
            # Process all other categories (lists of strings)
            for category in self.allowed_categories:
                if category == "demographics":
                    continue
                items = persona_json.get(category, [])
                rel_type = self.relationship_map[category]
                for item in items:
                    clean_item = item.strip()
                    if clean_item:
                        attr_id = self.compute_attribute_id(category, clean_item)
                        session.run(
                            f"""
                            MERGE (a:Attribute {{id: $attr_id}})
                            ON CREATE SET a.category = $category, a.value = $value
                            ON MATCH SET a.category = $category, a.value = $value
                            WITH a
                            MATCH (p:Persona {{id: $id}})
                            MERGE (p)-[:{rel_type}]->(a)
                            """,
                            attr_id=attr_id, category=category, value=clean_item, id=persona_id
                        )
        return persona_id