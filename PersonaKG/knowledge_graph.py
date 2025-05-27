from neo4j import GraphDatabase
import hashlib

class KnowledgeGraph:
    def __init__(self, uri, user, password, schema=None):
        # Initialize the Neo4j driver.
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Default schema configuration if none provided
        self._set_default_schema()
        
        # Update schema if provided
        if schema:
            self.update_schema(schema)
    
    def _set_default_schema(self):
        """Set the default schema with predefined categories and fields."""
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
        
        # Initialize field categories dictionary
        self.field_categories = {}
        
        # Define the allowed fields for the demographics category
        self.field_categories["demographics"] = {
            "age",
            "employmentStatus",
            "educationStatus",
            "livingSituation",
            "placeGrewUp",
            "currentLocation",
            "pets",
            "allergies"
        }
        
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
    
    def get_current_schema_info(self):
        """Get information about the current schema from the database.
        
        Returns:
            dict: Information about the current schema, including:
                  - categories: list of category names from Attribute nodes
                  - relationship_types: list of relationship types used
                  - demographic_fields: list of demographic field keys
        """
        with self.driver.session() as session:
            # Get all categories from Attribute nodes
            categories_query = """
            MATCH (a:Attribute)
            RETURN DISTINCT a.category AS category
            """
            categories = [record["category"] for record in session.run(categories_query)]
            
            # Get all relationship types used in the graph
            rel_types_query = """
            MATCH ()-[r]->() 
            RETURN DISTINCT type(r) AS rel_type
            """
            rel_types = [record["rel_type"] for record in session.run(rel_types_query)]
            
            # Get all demographic field keys
            demographics_fields_query = """
            MATCH (a:Attribute {category: 'demographics'})
            WHERE a.key IS NOT NULL
            RETURN DISTINCT a.key AS field
            """
            demographic_fields = [record["field"] for record in session.run(demographics_fields_query)]
        
        return {
            "categories": categories,
            "relationship_types": rel_types,
            "demographic_fields": demographic_fields
        }
    
    def schema_requires_rebuild(self, new_schema):
        """Determine if the new schema is significantly different from the current schema.
        
        Args:
            new_schema (list): The new schema configuration
            
        Returns:
            bool: True if the schema change requires a database rebuild
        """
        # Get current schema info
        current_info = self.get_current_schema_info()
        
        # Extract new schema categories and demographic fields
        new_categories = set(config[0] for config in new_schema)
        new_demographic_fields = set()
        for config in new_schema:
            if config[0] == "demographics" and len(config) > 1 and config[1]:
                new_demographic_fields = set(config[1])
        
        # Check for significant changes
        categories_changed = current_info["categories"] and new_categories.difference(current_info["categories"])
        demographic_fields_changed = current_info["demographic_fields"] and new_demographic_fields.difference(current_info["demographic_fields"])
        
        return categories_changed or demographic_fields_changed
    
    def drop_database(self):
        """Drop all nodes and relationships in the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database has been cleared.")
    
    def update_schema(self, schema, force_rebuild=False, skip_schema_check=False):
        """Update the knowledge graph schema with a new configuration.
        
        Args:
            schema (list): List of lists where each inner list contains:
                           [0] - category name (str)
                           [1] - fields (list of str or None if not demographics)
            force_rebuild (bool): If True, drop and recreate the database
            skip_schema_check (bool): If True, skip the schema change detection
        """
        # Check if we need to rebuild the database
        if force_rebuild:
            print("Forcing database rebuild due to explicit request...")
            self.drop_database()
        elif not skip_schema_check and self.schema_requires_rebuild(schema):
            print("Schema change detected. Dropping and recreating the database...")
            self.drop_database()
        
        # Clear existing schema configuration
        self.allowed_categories = set()
        self.field_categories = {}  # Store field-based categories and their fields
        self.relationship_map = {}
        
        # Process the new schema
        for category_config in schema:
            category = category_config[0]
            fields = category_config[1] if len(category_config) > 1 else None
            
            # Add to allowed categories
            self.allowed_categories.add(category)
            
            # Create relationship type (convert to uppercase with HAS_ prefix)
            rel_type = f"HAS_{category.upper()}"
            if category == "demographics":
                rel_type = "HAS_DEMOGRAPHIC"
            self.relationship_map[category] = rel_type
            
            # Store field-based categories and their fields
            if fields:
                self.field_categories[category] = set(fields)

    def close(self):
        self.driver.close()

    def validate_json(self, persona_json):
        """
        Validates that the input JSON only contains allowed categories,
        and for field-based categories (like basics/demographics/profile), only allowed fields.
        """
        # Check that all top-level keys are allowed.
        for key in persona_json:
            if key not in self.allowed_categories:
                raise ValueError(
                    f"Unrecognized category in JSON: '{key}'. "
                    f"Allowed categories: {self.allowed_categories}"
                )
        
        # Validate all field-based categories (previously just demographics)
        for category, allowed_fields in getattr(self, 'field_categories', {}).items():
            if category in persona_json:
                category_data = persona_json.get(category, {})
                if isinstance(category_data, dict):
                    for field in category_data:
                        if field not in allowed_fields:
                            raise ValueError(
                                f"Unrecognized field '{field}' in '{category}' category. "
                                f"Allowed fields: {allowed_fields}"
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
            
            # Find field-based category (previously demographics) by checking if value is dict or list
            field_based_categories = []
            for category in persona_json:
                if isinstance(persona_json[category], dict):
                    field_based_categories.append(category)
                    
            # Process field-based categories (can be basics, demographics, profile, etc.)
            for field_category in field_based_categories:
                if field_category in self.relationship_map:
                    field_data = persona_json.get(field_category, {})
                    rel_type = self.relationship_map[field_category]
                    
                    # Process each field in the category
                    for field, value in field_data.items():
                        if value is None:
                            continue
                        elif isinstance(value, list):
                            # Handle list values for a field
                            for list_item in value:
                                if list_item is not None:
                                    clean_item = str(list_item).strip()
                                    if clean_item:
                                        attr_id = self.compute_attribute_id(field_category, clean_item, key=field)
                                        session.run(
                                        f"""
                                        MERGE (a:Attribute {{id: $attr_id}})
                                        ON CREATE SET a.category = $category, a.key = $field, a.value = $value
                                        ON MATCH SET a.category = $category, a.key = $field, a.value = $value
                                        WITH a
                                        MATCH (p:Persona {{id: $id}})
                                        MERGE (p)-[:{rel_type}]->(a)
                                        """,
                                        attr_id=attr_id, category=field_category, field=field, 
                                        value=clean_item, id=persona_id
                                        )
                            continue
                        
                        # Handle single string values
                        clean_value = str(value).strip()
                        if clean_value:
                            attr_id = self.compute_attribute_id(field_category, clean_value, key=field)
                            session.run(
                                f"""
                                MERGE (a:Attribute {{id: $attr_id}})
                                ON CREATE SET a.category = $category, a.key = $field, a.value = $value
                                ON MATCH SET a.category = $category, a.key = $field, a.value = $value
                                WITH a
                                MATCH (p:Persona {{id: $id}})
                                MERGE (p)-[:{rel_type}]->(a)
                                """,
                                attr_id=attr_id, category=field_category, field=field, 
                                value=clean_value, id=persona_id
                            )
            
            # Process all list-based categories (simple arrays of strings)
            for category in self.allowed_categories:
                if category in field_based_categories:
                    continue  # Skip field-based categories already processed
                
                items = persona_json.get(category, [])
                if not isinstance(items, list):
                    continue  # Skip if not a list
                    
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
        
    def get_persona_info(self, user_id):
        """
        Retrieves persona information from the knowledge graph for a specified user.
        
        Args:
            user_id (str): Identifier for the user (e.g., "user1", "user2")
            
        Returns:
            str: A formatted string containing persona information from the knowledge graph
        """
        # Since we don't have direct mapping from user_id to persona_id in this case,
        # we'll fetch the first persona in the database (or a random one)
        # In a real application, you'd want to map user_id to persona_id more explicitly
        
        with self.driver.session() as session:
            # Get the first persona or a specific one by replacing LIMIT 1 with appropriate filtering
            persona_query = """
            MATCH (p:Persona)
            RETURN p.id AS persona_id
            LIMIT 1
            """
            
            persona_result = session.run(persona_query).single()
            
            if not persona_result:
                return "No persona information available."
                
            persona_id = persona_result["persona_id"]
            
            # Get all attributes for this persona
            attributes_query = """
            MATCH (p:Persona {id: $id})-[r]->(a:Attribute)
            RETURN a.category AS category, a.key AS key, a.value AS value, type(r) AS relationship
            ORDER BY a.category, a.key
            """
            
            attributes = session.run(attributes_query, id=persona_id)
            
            # Organize attributes by category
            attribute_info = {}
            for record in attributes:
                category = record["category"]
                key = record["key"]
                value = record["value"]
                
                if category not in attribute_info:
                    attribute_info[category] = []
                    
                if key:
                    attribute_info[category].append(f"{key}: {value}")
                else:
                    attribute_info[category].append(value)
            
            # Format the information as a readable string
            info_text = f"Knowledge Graph Information for {user_id}:\n\n"
            
            for category, values in attribute_info.items():
                info_text += f"{category.capitalize()}:\n"
                for value in values:
                    info_text += f"- {value}\n"
                info_text += "\n"
            
            return info_text