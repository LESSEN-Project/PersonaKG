import os
import hashlib
import random
import json
import pathlib
import argparse

from models import LLM
from persona_dataset import PersonaDataset
from prompts import *

argparser = argparse.ArgumentParser(description="PersonaKG")
argparser.add_argument("-d", "--datasets", nargs='+', type=str, default="all", help="Dataset(s) to use")
argparser.add_argument("-m", "--model", type=str, default="GPT-4.1", help="Model to use")
argparser.add_argument("-s", "--schema", type=str, default="default_schema.json", help="Schema file")
argparser.add_argument("-sp", "-split", "--split", type=str, default="train", choices=["train", "validation", "test"], help="Dataset split to use")
argparser.add_argument("-ss", "-sample_size", "--sample_size", type=int, default=-1, help="Sample size for each dataset")

args = argparser.parse_args()

def get_schema(path):
    with open(path, "r") as f:
        schema = json.load(f)
    return schema

schema = get_schema(args.schema)
llm = LLM(args.model)

dataset_config = PersonaDataset().get_config()
personas = PersonaDataset().load_all_personas(args.split, args.sample_size)
dataset = "PersonaChat"
sample_persona = personas[dataset][2]

prompt = kg_prompt(schema, sample_persona["persona_statements"])

kg_persona = llm.generate(prompt=prompt, json_output=True)

kg_persona["Persona"]["id"] = sample_persona["id"]
kg_persona["Persona"]["dataset_id"] = sample_persona["dataset_id"]
kg_persona["Persona"]["dataset"] = dataset
kg_persona["Persona"]["dataset_type"] = dataset_config[dataset]["origin"]
kg_persona["Persona"]["utterances"] = sample_persona["utterances"]

print(kg_persona)