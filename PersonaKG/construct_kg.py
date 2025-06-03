import os
import hashlib
import random
import json
import pathlib
import argparse
from typing import Dict, Any, List, Tuple, Optional
import sys

from models import LLM
from persona_dataset import PersonaDataset
from prompts import *

def encode_filename(schema_path, args_dict):
    schema_name = os.path.basename(schema_path)
    schema_name = schema_name.split(".")[0]
    with open(schema_path, "rb") as f:
        schema_hash = hashlib.md5(f.read()).hexdigest()[:8]
    args_str = json.dumps(args_dict, sort_keys=True)
    args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
    return f"kg_{schema_name}_{schema_hash}_{args_hash}.json"

argparser = argparse.ArgumentParser(description="PersonaKG")
argparser.add_argument("-d", "--datasets", nargs='+', type=str, default="all", help="Dataset(s) to use")
argparser.add_argument("-m", "--model", type=str, default="GPT-4.1", help="Model to use")
argparser.add_argument("-s", "--schema", type=str, default="files/default_schema.json", help="Schema file")
argparser.add_argument("-sp", "-split", "--split", type=str, default="train", choices=["train", "validation", "test"], help="Dataset split to use")
argparser.add_argument("-ss", "-sample_size", "--sample_size", type=int, default=-1, help="Sample size for each dataset")

args = argparser.parse_args()
args_dict = {
    "datasets": args.datasets,
    "model": args.model,
    "split": args.split,
    "sample_size": args.sample_size
}
out_file = encode_filename(args.schema, args_dict)
out_file = os.path.join("files", out_file)
log_file = out_file.replace(".json", ".log")
if os.path.exists(log_file):
    os.remove(log_file)

def get_schema(path):
    with open(path, "r") as f:
        schema = json.load(f)
    return schema

def validate_json_against_schema(json_obj: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []
    if "Persona" not in json_obj:
        errors.append("Missing 'Persona' node in the generated JSON")
        return False, errors
    persona = json_obj["Persona"]
    persona_schema = schema.get("Persona", {})
    schema_props = persona_schema.get("properties", {})
    for prop_name, prop_schema in schema_props.items():
        if prop_name in persona:
            prop_type = prop_schema.get("type", "str")
            prop_value = persona[prop_name]
            if "enum" in prop_schema and prop_value is not None and prop_value not in prop_schema["enum"]:
                errors.append(f"Property '{prop_name}' has invalid enum value: {prop_value}")
            if prop_type == "int" and not (isinstance(prop_value, int) or (isinstance(prop_value, str) and prop_value.isdigit())):
                errors.append(f"Property '{prop_name}' should be an integer, got '{prop_value}'")
    schema_edges = persona_schema.get("edges", {})
    for edge_name, edge_schema in schema_edges.items():
        if edge_name in persona:
            target_node_type = edge_schema.get("target", "")
            if target_node_type not in schema:
                errors.append(f"Edge '{edge_name}' references non-existent node type '{target_node_type}'")
                continue
            if not isinstance(persona[edge_name], list):
                errors.append(f"Edge '{edge_name}' should be a list, got {type(persona[edge_name]).__name__}")
                continue
            edge_prop_schema = edge_schema.get("properties", {})
            for i, edge_item in enumerate(persona[edge_name]):
                if not isinstance(edge_item, dict):
                    errors.append(f"Edge item at index {i} in '{edge_name}' should be an object")
                    continue
                for prop_name, prop_schema in edge_prop_schema.items():
                    if prop_name in edge_item and "enum" in prop_schema:
                        if edge_item[prop_name] not in prop_schema["enum"]:
                            errors.append(f"Property '{prop_name}' in edge '{edge_name}' at index {i} has invalid enum value: {edge_item[prop_name]}")
    for edge_name in persona:
        if edge_name not in schema_props and edge_name not in schema_edges and edge_name not in ["id", "dataset_id", "dataset", "dataset_type", "utterances", "persona_statements"]:
            errors.append(f"Unexpected edge or property '{edge_name}' in Persona node")
    return len(errors) == 0, errors

def robust_llm_json_generate(llm, prompt, gen_params, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            result = llm.generate(prompt=prompt, gen_params=gen_params, json_output=True)
            if isinstance(result, dict):
                return result
            elif isinstance(result, str):
                return json.loads(result)
        except Exception:
            pass
    try:
        result = llm.generate(prompt=prompt, gen_params=gen_params, json_output=True)
        return result if isinstance(result, dict) else {"raw_output": result}
    except Exception as e:
        return {"error": str(e)}

class TeeLogger:
    def __init__(self, log_path):
        self.log = open(log_path, "a", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, msg):
        self.log.write(msg)
        self.stdout.write(msg)
    def flush(self):
        self.log.flush()
        self.stdout.flush()
    def close(self):
        self.log.close()

schema = get_schema(args.schema)
llm = LLM(args.model)
dataset_config = PersonaDataset().get_config()
if args.datasets == "all":
    personas = PersonaDataset().load_all_personas(args.split, args.sample_size)
else:
    personas = {}
    for dataset in args.datasets:
        personas[dataset] = PersonaDataset().load_personas(dataset, args.split, args.sample_size)

all_personas = {}

logger = TeeLogger(log_file)
sys.stdout = logger

try:
    for dataset in personas:
        
        print(f"\nProcessing dataset: {dataset}")
        dataset_personas = []
        for persona in personas[dataset]:
            try:
                print(f"\nProcessing persona ID: {persona['id']} from dataset: {dataset}")
                prompt = kg_prompt(schema, persona["persona_statements"])
                kg_persona = robust_llm_json_generate(llm, prompt, gen_params={"max_tokens": 2048})
                print("Validating generated JSON against schema...")
                is_valid, errors = validate_json_against_schema(kg_persona, schema)
                if not is_valid:
                    print(f"Found {len(errors)} schema validation errors:")
                    for error in errors:
                        print(f"- {error}")
                    print("\nAttempting to correct the JSON using the LLM...")
                    error_str = "\n".join(f"- {error}" for error in errors)
                    correction_prompt = json_correction_prompt(schema, kg_persona, error_str)
                    kg_persona = robust_llm_json_generate(llm, correction_prompt, gen_params={"max_tokens": 2048})
                    is_valid, errors = validate_json_against_schema(kg_persona, schema)
                    if not is_valid:
                        print(f"Corrected JSON still has {len(errors)} validation errors:")
                        for error in errors:
                            print(f"- {error}")
                    else:
                        print("JSON successfully corrected!")
                else:
                    print("JSON validation successful - no errors found.")
                kg_persona["Persona"]["id"] = persona["id"]
                kg_persona["Persona"]["dataset_id"] = persona["dataset_id"]
                kg_persona["Persona"]["dataset"] = dataset
                kg_persona["Persona"]["dataset_source"] = dataset_config[dataset]["origin"]
                kg_persona["Persona"]["utterances"] = persona["utterances"]
                dataset_personas.append(kg_persona)
            except Exception as e:
                print(f"Error processing persona ID: {persona['id']} from dataset: {dataset}")
                print(e)
                continue
            all_personas[dataset] = dataset_personas

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_personas, f, ensure_ascii=False, indent=2)
finally:
    sys.stdout = logger.stdout
    logger.close()
