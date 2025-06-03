import os
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from neo4j import GraphDatabase, basic_auth

class PersonaKG:
    """
    A class for creating and managing a Neo4j knowledge graph for persona data.
    This class processes persona data from different datasets and creates a unified knowledge graph.
    """
    
    def __init__(self, uri: str = "neo4j+s://f3ba5f50.databases.neo4j.io", 
                 username: str = "neo4j", 
                 password: str = "password",
                 database: str = "neo4j"):
        """
        Initialize the PersonaKG with Neo4j connection parameters.
        
        Args:
            uri (str): Neo4j connection URI (default: bolt://localhost:7687)
            username (str): Neo4j username (default: neo4j)
            password (str): Neo4j password (default: password)
            database (str): Neo4j database name (default: PersonaKG_Ext)
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """
        Connect to the Neo4j database.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=basic_auth(self.username, self.password)
            )
            # Verify connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            self.logger.info(f"Successfully connected to Neo4j database '{self.database}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
            return False
    
    def close(self):
        """
        Close the connection to the Neo4j database.
        """
        if self.driver:
            self.driver.close()
            self.driver = None
            self.logger.info("Neo4j connection closed")
    
    def clear_database(self):
        """
        Clear all nodes and relationships in the database.
        """
        if not self.driver:
            self.connect()
            
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        self.logger.info("Database cleared")
    
    def load_schema(self, schema_path: str) -> Dict[str, Any]:
        """
        Load the schema from a JSON file.
        
        Args:
            schema_path (str): Path to the schema JSON file
            
        Returns:
            Dict[str, Any]: The loaded schema
        """
        with open(schema_path, "r") as f:
            schema = json.load(f)
        return schema
    
    def load_personas(self, data_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load persona data from a JSON file.
        
        Args:
            data_path (str): Path to the persona data JSON file
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary containing personas by dataset
        """
        with open(data_path, "r", encoding="utf-8") as f:
            personas = json.load(f)
        return personas
    
    def create_constraints_and_indexes(self, schema: Dict[str, Any]):
        """
        Create Neo4j constraints and indexes based on the schema.
        
        Args:
            schema (Dict[str, Any]): The schema dictionary
        """
        if not self.driver:
            self.connect()
            
        with self.driver.session(database=self.database) as session:
            # Create constraints for each node type in the schema
            for node_type in schema.keys():
                # Create constraint on id property for each node type
                try:
                    session.run(
                        f"CREATE CONSTRAINT {node_type}_id IF NOT EXISTS "
                        f"FOR (n:{node_type}) REQUIRE n.id IS UNIQUE"
                    )
                    self.logger.info(f"Created constraint for {node_type} nodes")
                except Exception as e:
                    self.logger.warning(f"Could not create constraint for {node_type}: {str(e)}")
                    
                # Create index on name property if it exists in the node's properties
                properties = schema[node_type].get("properties", {})
                if "name" in properties:
                    try:
                        session.run(
                            f"CREATE INDEX {node_type}_name IF NOT EXISTS "
                            f"FOR (n:{node_type}) ON (n.name)"
                        )
                        self.logger.info(f"Created index on name for {node_type} nodes")
                    except Exception as e:
                        self.logger.warning(f"Could not create index for {node_type}.name: {str(e)}")
    
    def create_node(self, session, node_type: str, properties: Dict[str, Any]) -> str:
        """
        Create a node in the Neo4j database.
        
        Args:
            session: Neo4j session
            node_type (str): Type of the node
            properties (Dict[str, Any]): Properties of the node
            
        Returns:
            str: ID of the created node
        """
        # Filter out None values
        filtered_props = {k: v for k, v in properties.items() if v is not None}
        
        # Generate a unique ID if not provided
        if "id" not in filtered_props:
            # Use name as ID if available, otherwise generate a random ID
            if "name" in filtered_props:
                node_id = f"{node_type}_{filtered_props['name'].replace(' ', '_')}"
            else:
                import uuid
                node_id = f"{node_type}_{uuid.uuid4().hex}"
            filtered_props["id"] = node_id
            
        # Create or merge the node
        query = (
            f"MERGE (n:{node_type} {{id: $id}}) "
            f"ON CREATE SET n = $properties "
            f"ON MATCH SET n += $properties "
            f"RETURN n.id as id"
        )
        
        result = session.run(query, id=filtered_props["id"], properties=filtered_props)
        record = result.single()
        return record["id"] if record else None
    
    def create_relationship(self, session, from_id: str, to_id: str, 
                           rel_type: str, properties: Dict[str, Any] = None):
        """
        Create a relationship between two nodes in the Neo4j database.
        
        Args:
            session: Neo4j session
            from_id (str): ID of the source node
            to_id (str): ID of the target node
            rel_type (str): Type of the relationship
            properties (Dict[str, Any], optional): Properties of the relationship
        """
        if properties is None:
            properties = {}
        
        # Filter out None values
        filtered_props = {k: v for k, v in properties.items() if v is not None}
        
        query = (
            f"MATCH (a), (b) "
            f"WHERE a.id = $from_id AND b.id = $to_id "
            f"CREATE (a)-[r:{rel_type} $properties]->(b) "
            f"RETURN type(r)"
        )
        
        session.run(query, from_id=from_id, to_id=to_id, properties=filtered_props)
    
    def process_persona_node(self, session, persona: Dict[str, Any], schema: Dict[str, Any]) -> str:
        """
        Process a persona node and its relationships.
        
        Args:
            session: Neo4j session
            persona (Dict[str, Any]): The persona data
            schema (Dict[str, Any]): The schema dictionary
            
        Returns:
            str: ID of the created persona node
        """
        persona_schema = schema.get("Persona", {})
        schema_props = persona_schema.get("properties", {})
        schema_edges = persona_schema.get("edges", {})
        
        # Extract persona properties (excluding edges and special fields)
        persona_props = {}
        for prop_name in persona:
            if prop_name not in schema_edges and not isinstance(persona[prop_name], list) and prop_name != "utterances":
                persona_props[prop_name] = persona[prop_name]
        
        # Create persona node
        persona_id = self.create_node(session, "Persona", persona_props)
        
        # Process edges (relationships)
        for edge_name, edge_items in persona.items():
            if edge_name in schema_edges and isinstance(edge_items, list):
                edge_schema = schema_edges[edge_name]
                target_node_type = edge_schema.get("target", "")
                
                for edge_item in edge_items:
                    # Create the target node if it exists inline
                    target_node_id = None
                    if target_node_type in edge_item:
                        # Target node is defined inline
                        target_node = edge_item[target_node_type]
                        target_node_id = self.create_node(session, target_node_type, target_node)
                    elif "name" in edge_item:
                        # Simple case with just a name
                        target_props = {"name": edge_item["name"]}
                        target_node_id = self.create_node(session, target_node_type, target_props)
                    
                    if target_node_id:
                        # Extract relationship properties
                        rel_props = {k: v for k, v in edge_item.items() 
                                    if k != target_node_type and k != "name" and k != "target"}
                        
                        # Create relationship
                        self.create_relationship(session, persona_id, target_node_id, edge_name, rel_props)
        
        return persona_id
    
    def create_knowledge_graph(self, schema_path: str, data_path: str, clear_existing: bool = False):
        """
        Create a knowledge graph from persona data.
        
        Args:
            schema_path (str): Path to the schema JSON file
            data_path (str): Path to the persona data JSON file
            clear_existing (bool): Whether to clear existing data before creating the graph
        """
        schema = self.load_schema(schema_path)
        personas_by_dataset = self.load_personas(data_path)
        
        if not self.driver:
            success = self.connect()
            if not success:
                self.logger.error("Failed to connect to Neo4j, aborting graph creation")
                return
        
        if clear_existing:
            self.clear_database()
        
        # Create constraints and indexes
        self.create_constraints_and_indexes(schema)
        
        # Process all personas from all datasets
        with self.driver.session(database=self.database) as session:
            for dataset_name, dataset_personas in personas_by_dataset.items():
                self.logger.info(f"Processing dataset: {dataset_name}")
                
                for persona_data in dataset_personas:
                    try:
                        persona_id = self.process_persona_node(session, persona_data["Persona"], schema)
                        self.logger.info(f"Processed persona ID: {persona_id} from dataset: {dataset_name}")
                    except Exception as e:
                        persona_id = persona_data["Persona"].get("id", "unknown")
                        self.logger.error(f"Error processing persona ID: {persona_id} from dataset: {dataset_name}")
                        self.logger.error(str(e))
        
        self.logger.info("Knowledge graph creation completed")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dict[str, int]: Dictionary containing statistics
        """
        if not self.driver:
            self.connect()
            
        stats = {}
        
        with self.driver.session(database=self.database) as session:
            # Count total nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            record = result.single()
            stats["total_nodes"] = record["count"] if record else 0
            
            # Count nodes by label
            result = session.run(
                "CALL apoc.meta.stats() YIELD labels RETURN labels"
            )
            record = result.single()
            if record and "labels" in record:
                for label, count in record["labels"].items():
                    stats[f"{label}_nodes"] = count
            
            # Count total relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            record = result.single()
            stats["total_relationships"] = record["count"] if record else 0
            
            # Count relationships by type
            result = session.run(
                "CALL apoc.meta.stats() YIELD relTypes RETURN relTypes"
            )
            record = result.single()
            if record and "relTypes" in record:
                for rel_type, count in record["relTypes"].items():
                    stats[f"{rel_type}_relationships"] = count
        
        return stats

# Example usage
if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create a knowledge graph from persona data")
    parser.add_argument("-s", "--schema", type=str, default="files/default_schema.json", 
                        help="Path to the schema file")
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="Path to the persona data file")
    parser.add_argument("-u", "--uri", type=str, default="neo4j+s://f3ba5f50.databases.neo4j.io",
                        help="Neo4j connection URI")
    parser.add_argument("-n", "--username", type=str, default="neo4j",
                        help="Neo4j username")
    parser.add_argument("-db", "--database", type=str, default="neo4j",
                        help="Neo4j database name")
    parser.add_argument("-c", "--clear", action="store_true",
                        help="Clear existing data before creating the graph")
    
    args = parser.parse_args()
    
    password = os.getenv("NEO4J_PKG_EXT_PASSWORD")
    # Create knowledge graph
    kg = PersonaKG(
        uri=args.uri,
        username=args.username,
        password=password,
        database=args.database
    )
    
    kg.create_knowledge_graph(args.schema, args.data, args.clear)
    
    # Print statistics
    stats = kg.get_statistics()
    print("\nKnowledge Graph Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Close connection
    kg.close()
