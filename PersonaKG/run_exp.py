import os
import json
import argparse
import hashlib
import datetime
from tqdm import tqdm
from dataset import get_dataset
from evaluate import load
import re
from prompts import get_next_utterance_prompt

from construct_kg import load_knowledge_graph_from_file
from models import LLM
from knowledge_graph import KnowledgeGraph

def setup_args():
    parser = argparse.ArgumentParser(description='Run next utterance prediction experiment')
    parser.add_argument('--model', "-m", type=str, default='GPT-4.1-mini')
    parser.add_argument('--split', "-s", type=str, default='test', 
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to use for evaluation')
    parser.add_argument('--knowledge_graph', "-kg", type=str, default=None,
                        help='Path to a knowledge graph file. If provided, will use the KG for persona information')
    parser.add_argument('--max_neighbors', "-mn", type=int, default=3,
                        help='Maximum number of neighboring personas to include in KG context')
    parser.add_argument('--output_dir', "-od", type=str, default='results',
                        help='Directory to save results')
    return parser.parse_args()

def parse_conversation(conversation):
    lines = conversation.strip().split('\n')
    utterances = []
    
    for line in lines:
        if not line.strip():
            continue
        parts = line.split(':', 1)
        if len(parts) == 2:
            speaker = parts[0].strip()
            text = parts[1].strip()
            utterances.append({"speaker": speaker, "text": text})
    
    return utterances

def create_prediction_samples(data, split='test'):
    samples = []
    dataset_split = data[split]
    
    # Process all items sequentially
    for idx in range(len(dataset_split)):
        if dataset_split[idx]['Best Generated Conversation']:
            user1_persona = dataset_split[idx]['user 1 personas']
            user2_persona = dataset_split[idx]['user 2 personas']
            conversation = dataset_split[idx]['Best Generated Conversation']
            
            utterances = parse_conversation(conversation)
            
            for i in range(len(utterances)):
                history = utterances[:i]
                target = utterances[i]['text'] if i < len(utterances) else None
                
                if target is not None:
                    samples.append({
                        'user1_persona': user1_persona,
                        'user2_persona': user2_persona,
                        'history': history,
                        'target_speaker': utterances[i]['speaker'],
                        'target': target
                    })
    
    return samples

def predict_next_utterance(sample, llm, kg=None, kg_extractor=None, canonicalizer=None, max_neighbors=3, return_prompt=False):
    user1_persona = sample['user1_persona']
    user2_persona = sample['user2_persona']
    history = sample['history']
    target_speaker = sample['target_speaker']
    
    # Process personas through knowledge graph if provided
    kg_info = None
    if kg is not None and kg_extractor is not None and canonicalizer is not None:
        # Generate unique IDs for personas
        user1_id = 'user1_' + hashlib.md5(user1_persona.encode()).hexdigest()
        user2_id = 'user2_' + hashlib.md5(user2_persona.encode()).hexdigest()
        
        with kg.driver.session() as session:
            # Check if personas already exist
            user1_exists = session.run("MATCH (p:Persona {id: $id}) RETURN p", id=user1_id).single()
            user2_exists = session.run("MATCH (p:Persona {id: $id}) RETURN p", id=user2_id).single()
            
            # Process user1 persona if it doesn't exist
            if not user1_exists:
                try:
                    print(f"Adding User 1 persona to knowledge graph...")
                    # Extract attributes using LLM
                    extracted_attrs = kg_extractor.generate(prompt_params={"persona": user1_persona}, json_output=True)
                    
                    # Get existing attributes for canonicalization
                    existing_attrs = kg.get_existing_attributes()
                    
                    # Canonicalize extracted attributes
                    canonized_attrs = canonicalizer.generate(
                        prompt_params={"existing_attributes": existing_attrs, "persona_json": extracted_attrs},
                        json_output=True
                    )
                    
                    # Handle string response if needed
                    if isinstance(canonized_attrs, str):
                        try:
                            canonized_attrs = json.loads(canonized_attrs)
                        except json.JSONDecodeError:
                            import ast
                            try:
                                python_dict = ast.literal_eval(canonized_attrs)
                                canonized_attrs = json.loads(json.dumps(python_dict))
                            except Exception:
                                print("Error parsing canonized attributes")
                                canonized_attrs = extracted_attrs
                    
                    # Add to knowledge graph directly - the KG already has the right schema
                    kg.upsert_persona(canonized_attrs, user1_id)
                    print(f"Added User 1 persona to knowledge graph")
                except Exception as e:
                    print(f"Error adding User 1 persona to knowledge graph: {str(e)}")

            else:
                print("User 1 persona already exists in knowledge graph")
            
            # Process user2 persona if it doesn't exist
            if not user2_exists:
                try:
                    print(f"Adding User 2 persona to knowledge graph...")
                    # Extract attributes using LLM
                    extracted_attrs = kg_extractor.generate(prompt_params={"persona": user2_persona}, json_output=True)
                    
                    # Get existing attributes for canonicalization
                    existing_attrs = kg.get_existing_attributes()
                    
                    # Canonicalize extracted attributes
                    canonized_attrs = canonicalizer.generate(
                        prompt_params={"existing_attributes": existing_attrs, "persona_json": extracted_attrs},
                        json_output=True
                    )
                    
                    # Handle string response if needed
                    if isinstance(canonized_attrs, str):
                        try:
                            canonized_attrs = json.loads(canonized_attrs)
                        except json.JSONDecodeError:
                            import ast
                            try:
                                python_dict = ast.literal_eval(canonized_attrs)
                                canonized_attrs = json.loads(json.dumps(python_dict))
                            except Exception:
                                print("Error parsing canonized attributes")
                                canonized_attrs = extracted_attrs
                    
                    # Add to knowledge graph directly - the KG already has the right schema
                    kg.upsert_persona(canonized_attrs, user2_id)
                    print(f"Added User 2 persona to knowledge graph")
                except Exception as e:
                    print(f"Error adding User 2 persona to knowledge graph: {str(e)}")
            else:
                print("User 2 persona already exists in knowledge graph")
            
        # Get KG information for the target persona and neighbors
        kg_info = ""
        target_id = user1_id if target_speaker == "User 1" else user2_id
        
        with kg.driver.session() as session:
            # 1. First get direct attributes of the target persona
            kg_info = f"Knowledge Graph Context for {target_speaker}:\n\n"
            kg_info += f"== {target_speaker}'s Attributes ==\n"
            
            attributes_query = """
            MATCH (p:Persona {id: $id})-[r]->(a:Attribute)
            RETURN a.category AS category, a.key AS key, a.value AS value
            ORDER BY a.category, a.key
            """
            
            attributes = session.run(attributes_query, id=target_id)
            
            # Format KG information by category
            category_data = {}
            
            # Group by category first
            for record in attributes:
                category = record["category"]
                key = record["key"]
                value = record["value"]
                
                if category not in category_data:
                    category_data[category] = []
                
                if key:
                    category_data[category].append(f"{key}: {value}")
                else:
                    category_data[category].append(value)
            
            # Format by category
            for category, values in category_data.items():
                kg_info += f"\n{category.capitalize()}:\n"
                for value in values:
                    kg_info += f"- {value}\n"
            
            # 2. Find neighbors with shared attributes
            kg_info += f"\n== Personas with Similar Attributes ==\n"
            
            neighbors_query = """
            // Find personas that share attributes with our target persona
            MATCH (p1:Persona {id: $id})-[r1]->(a:Attribute)<-[r2]-(p2:Persona)
            WHERE p1 <> p2
            WITH p2, count(a) AS shared_count, collect(a.value) AS shared_values
            ORDER BY shared_count DESC
            LIMIT $max_neighbors
            RETURN p2.id AS neighbor_id, shared_count, shared_values
            """
            
            neighbors = session.run(neighbors_query, id=target_id, max_neighbors=max_neighbors)
            
            for neighbor in neighbors:
                neighbor_id = neighbor["neighbor_id"]
                shared_count = neighbor["shared_count"]
                shared_values = neighbor["shared_values"]
                
                # Short display of neighbor ID
                short_id = neighbor_id[:8]
                
                kg_info += f"\nNeighbor {short_id} shares {shared_count} attributes:\n"
                for value in shared_values:
                    kg_info += f"- {value}\n"
                
                # Get specific information about this neighbor
                neighbor_info_query = """
                MATCH (p:Persona {id: $id})-[r]->(a:Attribute)
                RETURN a.category AS category, a.key AS key, a.value AS value
                ORDER BY a.category, a.key
                LIMIT 5
                """
                
                neighbor_attributes = session.run(neighbor_info_query, id=neighbor_id)
                
                # Add a few specific attributes from this neighbor
                kg_info += "Other attributes:\n"
                for record in neighbor_attributes:
                    value = record["value"]
                    kg_info += f"- {value}\n"
                    
            # 3. Add some statistical information about common attribute patterns
            kg_info += f"\n== Attribute Patterns ==\n"
            
            patterns_query = """
            // Find common attribute patterns in the same categories
            MATCH (p1:Persona)-[r1]->(a1:Attribute {category: $category})
            MATCH (p1)-[r2]->(a2:Attribute {category: $category})
            WHERE a1 <> a2
            WITH a1.value AS value1, a2.value AS value2, count(*) AS frequency
            ORDER BY frequency DESC
            LIMIT 3
            RETURN value1, value2, frequency
            """
            
            # Get the main category from the target persona's attributes
            main_categories = list(category_data.keys())
            if main_categories:
                main_category = main_categories[0]
                patterns = session.run(patterns_query, category=main_category)
                
                kg_info += f"Common {main_category} patterns:\n"
                for pattern in patterns:
                    value1 = pattern["value1"]
                    value2 = pattern["value2"]
                    frequency = pattern["frequency"]
                    kg_info += f"- People with '{value1}' often also have '{value2}' ({frequency} occurrences)\n"
    
    formatted_history = ""
    for utterance in history:
        formatted_history += f"{utterance['speaker']}: {utterance['text']}\n"
    
    prompt = get_next_utterance_prompt(
        user1_persona=user1_persona,
        user2_persona=user2_persona,
        conversation_history=formatted_history,
        target_speaker=target_speaker,
        kg_info=kg_info
    )
    
    prediction = llm.generate(prompt)
    
    prediction = re.sub(r'^.*?:', '', prediction).strip()
    
    if return_prompt:
        return prediction, prompt
    else:
        return prediction

def evaluate_predictions(predictions, targets):
    if not predictions or not targets:
        return {'bleu': 0, 'rouge': {'precision': 0, 'recall': 0, 'f1': 0}}
    
    bleu_metric = load("bleu")
    rouge_metric = load("rouge")
    
    references = [[t] for t in targets]
    
    bleu_result = bleu_metric.compute(predictions=predictions, references=references)
    bleu_score = bleu_result["bleu"]

    rouge_result = rouge_metric.compute(predictions=predictions, references=targets)
    
    return {
        'bleu': bleu_score,
        'rouge': rouge_result
    }

def create_experiment_id(args):
    """Create a unique identifier for an experiment based on its parameters"""
    params = {
        'model': args.model,
        'split': args.split,
        'knowledge_graph': args.knowledge_graph,
        'max_neighbors': args.max_neighbors
    }
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()

def run_experiment(args):
    data = get_dataset()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a unique experiment ID based on parameters
    experiment_id = create_experiment_id(args)
    checkpoint_file = os.path.join(args.output_dir, f"checkpoint_{experiment_id}.json")
    eval_file = os.path.join(args.output_dir, f"eval_{experiment_id}.json")
    final_output_file = os.path.join(args.output_dir, f"results_{experiment_id}.json")
    
    # Check if this experiment has already been completed
    if os.path.exists(final_output_file):
        print(f"Experiment already completed. Results available at {final_output_file}")
        with open(final_output_file, 'r') as f:
            results = json.load(f)
        print(f"BLEU Score: {results['metrics']['bleu']}")
        print(f"ROUGE-F1 Score: {results['metrics']['rouge']}")
        return results
    
    # Check if there's a checkpoint to resume from
    completed_samples = []
    predictions = []
    targets = []
    start_idx = 0
    
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file. Resuming experiment from checkpoint.")
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            completed_samples = checkpoint_data.get('completed_samples', [])
            predictions = checkpoint_data.get('predictions', [])
            targets = checkpoint_data.get('targets', [])
            start_idx = len(completed_samples)
        print(f"Resuming from sample {start_idx}")
    else:
        print(f"Starting new experiment with ID: {experiment_id}")
    
    # Initialize knowledge graph if specified
    kg = None
    schema = None
    kg_extractor = None
    canonicalizer = None
    
    if args.knowledge_graph:
        print(f"Loading knowledge graph from file: {args.knowledge_graph}")
        neo4j_password = os.environ.get("NEO4J_PKG_PASSWORD")
        if not neo4j_password:
            print("Error: NEO4J_PKG_PASSWORD environment variable not set. Knowledge graph will not be used.")
        else:
            # Determine the file path
            if not os.path.exists(args.knowledge_graph):
                # Try prefixing with graphs/
                alt_path = os.path.join("graphs", args.knowledge_graph)
                if os.path.exists(alt_path):
                    args.knowledge_graph = alt_path
                else:
                    print(f"Knowledge graph file not found: {args.knowledge_graph}")
                    return
                    
            # Use the load_knowledge_graph_from_file function
            print(f"Using load_knowledge_graph_from_file with {args.knowledge_graph}")
            try:
                # Load schema from the file
                with open(args.knowledge_graph, 'r') as f:
                    saved_data = json.load(f)
                
                schema = saved_data.get('schema', [])
                if not schema:
                    print("No schema found in the file. Cannot proceed.")
                    return
                
                # Create a temporary KG object to check the current schema
                temp_kg = KnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password=neo4j_password)
                
                # Check if the schema in the file matches the existing KG
                rebuild_needed = False
                try:
                    # First check if there's any data in the KG
                    with temp_kg.driver.session() as session:
                        result = session.run("MATCH (n) RETURN count(n) as count").single()
                        if result and result["count"] > 0:
                            # There's data, check if schema matches
                            print("Found existing knowledge graph. Checking schema compatibility...")
                            if temp_kg.schema_requires_rebuild(schema):
                                print("Schema mismatch detected. Rebuilding knowledge graph...")
                                rebuild_needed = True
                            else:
                                print("Existing schema is compatible. Using existing knowledge graph.")
                        else:
                            # Empty database, needs rebuild
                            print("Knowledge graph is empty. Building from file...")
                            rebuild_needed = True
                except Exception as e:
                    print(f"Error checking schema: {str(e)}")
                    rebuild_needed = True
                
                # If rebuild is needed, use load_knowledge_graph_from_file
                if rebuild_needed:
                    print(f"Loading knowledge graph from file: {args.knowledge_graph}")
                    success = load_knowledge_graph_from_file(args.knowledge_graph, neo4j_password)
                    if not success:
                        print("Failed to load knowledge graph from file")
                        return
                
                # Now create our own KG object to use for queries
                # The database already has the correct schema from load_knowledge_graph_from_file
                kg = KnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password=neo4j_password)
                
                # Load schema from the file for our LLMs
                with open(args.knowledge_graph, 'r') as f:
                    saved_data = json.load(f)
                
                schema = saved_data.get('schema', [])
                if not schema:
                    print("No schema found in the file. Cannot proceed.")
                    return
                
                # Make sure our KG instance has the same schema
                kg.update_schema(schema, force_rebuild=False, skip_schema_check=True)
                
                # Display schema information
                print("Using schema with these categories:")
                for category in schema:
                    if len(category) > 1 and category[1]:
                        print(f"- {category[0]} (with fields: {category[1]})")
                    else:
                        print(f"- {category[0]}")
                
                # Initialize LLMs for persona extraction and canonicalization
                from prompts import kg_prompt, canonicalization_prompt
                kg_extractor = LLM("GPT-4.1-mini", default_prompt=kg_prompt(schema=schema))
                canonicalizer = LLM("GPT-4.1-mini", default_prompt=canonicalization_prompt())
                
                print("Knowledge graph initialized with schema")
            except Exception as e:
                print(f"Error initializing knowledge graph: {str(e)}")
                kg = None
    
    print(f"Creating prediction samples from {args.split} split...")
    samples = create_prediction_samples(
        data, 
        split=args.split
    )
    
    # If we've already processed some samples, skip those
    if start_idx > 0:
        print(f"Skipping {start_idx} already processed samples")
        samples_to_process = samples[start_idx:]
    else:
        samples_to_process = samples
    
    if not samples_to_process:
        print("All samples have been processed. Proceeding to evaluation.")
    else:
        print(f"Running predictions with model {args.model} on {len(samples_to_process)} samples...")
        llm = LLM(args.model, gen_params={
            "temperature": 0.7,
            "max_tokens": 128
        })
    
    # Only process samples that haven't been processed yet
    for i, sample in enumerate(tqdm(samples_to_process)):
        try:
            # Get prediction and prompt
            prediction, prompt = predict_next_utterance(
                sample, 
                llm=llm,
                kg=kg,
                kg_extractor=kg_extractor,
                canonicalizer=canonicalizer,
                max_neighbors=args.max_neighbors,
                return_prompt=True
            )
            
            predictions.append(prediction)
            targets.append(sample['target'])
            
            # Record this sample as completed
            completed_samples.append({
                'index': start_idx + i,
                'user1_persona': sample['user1_persona'],
                'user2_persona': sample['user2_persona'],
                'history': sample['history'],
                'target': sample['target'],
                'prediction': prediction,
                'prompt': prompt
            })
            
            # Save checkpoint every 5 samples
            if (i + 1) % 5 == 0 or i == len(samples_to_process) - 1:
                checkpoint_data = {
                    'args': vars(args),
                    'experiment_id': experiment_id,
                    'completed_samples': completed_samples,
                    'timestamp': str(datetime.datetime.now()),
                    'progress': f"{len(completed_samples)}/{len(samples)}"
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"\nCheckpoint saved at sample {start_idx + i + 1}/{len(samples)}")
            
            # Run intermediate evaluation every 10 samples
            if (start_idx + i + 1) % 10 == 0 or i == len(samples_to_process) - 1:
                print(f"\nRunning intermediate evaluation at sample {start_idx + i + 1}...")
                # Get predictions and targets from completed samples
                interim_predictions = [s['prediction'] for s in completed_samples]
                interim_targets = [s['target'] for s in completed_samples]
                
                # Run evaluation
                interim_results = evaluate_predictions(interim_predictions, interim_targets)
                
                # Create interim result data
                interim_result_data = {
                    'args': vars(args),
                    'experiment_id': experiment_id,
                    'metrics': interim_results,
                    'samples': completed_samples,  # This now includes prompts
                    'timestamp': str(datetime.datetime.now()),
                    'total_samples': len(samples),
                    'processed_samples': len(completed_samples),
                    'is_interim': True
                }
                
                # Save to a single evaluation file that gets updated each time
                with open(eval_file, 'w') as f:
                    json.dump(interim_result_data, f, indent=2)
                    
                print(f"Evaluation results updated in {eval_file}")
                print(f"Interim BLEU Score: {interim_results['bleu']}")
                print(f"Interim ROUGE-F1 Score: {interim_results['rouge']}")
        except Exception as e:
            print(f"Error processing sample {start_idx + i}: {str(e)}")
            # Save checkpoint on error as well
            checkpoint_data = {
                'args': vars(args),
                'experiment_id': experiment_id,
                'completed_samples': completed_samples,
                'timestamp': str(datetime.datetime.now()),
                'progress': f"{len(completed_samples)}/{len(samples)}",
                'error': str(e)
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"\nCheckpoint saved before error at sample {start_idx + i + 1}/{len(samples)}")
            
            # Run evaluation on what we have so far
            if completed_samples:
                print("\nRunning evaluation on samples completed before error...")
                # Get predictions and targets from completed samples
                interim_predictions = [s['prediction'] for s in completed_samples]
                interim_targets = [s['target'] for s in completed_samples]
                
                # Run evaluation
                interim_results = evaluate_predictions(interim_predictions, interim_targets)
                
                # Save results
                interim_result_data = {
                    'args': vars(args),
                    'experiment_id': experiment_id,
                    'metrics': interim_results,
                    'samples': completed_samples,  # This now includes prompts
                    'timestamp': str(datetime.datetime.now()),
                    'total_samples': len(samples),
                    'processed_samples': len(completed_samples),
                    'is_interim': True,
                    'error': str(e)
                }
                
                # Save to the evaluation file with error flag
                with open(eval_file, 'w') as f:
                    json.dump(interim_result_data, f, indent=2)
                    
                print(f"Error evaluation saved to {eval_file}")
    
    print("Evaluating predictions...")
    results = evaluate_predictions(predictions, targets)
    
    # Create result data with all the details
    result_data = {
        'args': vars(args),
        'experiment_id': experiment_id,
        'metrics': results,
        'samples': completed_samples,  # This now includes prompts, predictions, everything
        'timestamp': str(datetime.datetime.now()),
        'total_samples': len(samples),
        'processed_samples': len(predictions),
        'is_final': True
    }
    
    # Save to final output file
    with open(final_output_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    # If we have a checkpoint file and the experiment is complete, we can remove it
    if os.path.exists(checkpoint_file) and len(predictions) == len(samples):
        os.remove(checkpoint_file)
        print(f"Checkpoint file removed as experiment is complete.")
    
    print(f"Results saved to {final_output_file}")
    print(f"BLEU Score: {results['bleu']}")
    print(f"ROUGE-F1 Score: {results['rouge']}")
    
    return results

if __name__ == "__main__":
    args = setup_args()
    run_experiment(args)