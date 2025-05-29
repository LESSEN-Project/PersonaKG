import argparse
import os
import re
import random

from PersonaKG.prompts import get_cluster_themes
from PersonaKG.models import LLM

def get_sample_statements(cluster_path):

    with open(cluster_path, "r") as f:
        cluster_data = f.readlines()
    statement_lines = [line[line.find(".")+1:] for line in cluster_data if re.match(r"^\d+\. ", line)]
    
    return statement_lines

parser = argparse.ArgumentParser(description="Analyze clusters")
add = parser.add_argument
add("-m", "--mode", type=str, default="combined", choices=["combined", "separate"], help="Clustering mode: combined (all datasets together), or separate (per dataset)")
add("-l", "--llm", type=str, default="QWEN-3-32B-GGUF", help="LLM to use for analysis")
add("-ca", "--clustering_algorithm", type=str, default="hdbscan", choices=["agglomerative", "hdbscan"], help="Which clustering algorithm to analyze")
add("-o", "--output_dir", type=str, default="clustering/cluster_analysis", help="Output directory for results")
args = parser.parse_args()

output_dir = os.path.join(args.output_dir, args.clustering_algorithm, args.mode)
os.makedirs(output_dir, exist_ok=True)
llm = LLM(model_name=args.llm, default_prompt=get_cluster_themes())

if llm.cfg.get("reason"):
    gen_params={
    "max_new_tokens": 4096, 
    "temperature":0.6, 
    "top_p":0.95, 
    "top_k":20, 
    "min_p":0
    }
else:
    gen_params={
        "max_new_tokens": 128
    }

input_dir = os.path.join(f"clustering/{args.clustering_algorithm}_clusters", args.mode)
max_statements = 200

clusters = [cluster for cluster in os.listdir(input_dir) if cluster.startswith("cluster_")]
print(f"Found {len(clusters)} clusters")

for i, cluster in enumerate(clusters):
    statements = get_sample_statements(os.path.join(input_dir, cluster))
    print(f"Found {len(statements)} statements in {cluster.split('.')[0]}\n")
    if len(statements) > max_statements:
        statements = random.sample(statements, max_statements)

    themes = llm.generate(prompt_params={"statements": "\n".join(statements)}, gen_params=gen_params)
    print(themes)
    output_path = os.path.join(output_dir, f"{cluster}_themes.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(themes)

if args.clustering_algorithm == "hdbscan":
    
    noise_file = os.path.join(input_dir, "noise_points.txt")
    noise_statements = get_sample_statements(noise_file)
    if len(noise_statements) > max_statements:
        noise_categories = len(noise_statements) // max_statements

        for i in range(noise_categories):
            themes = llm.generate(prompt_params={"statements": "\n".join(noise_statements[i*max_statements:(i+1)*max_statements])}, gen_params=gen_params)
            print(themes)
            output_path = os.path.join(output_dir, f"noise_points_themes_{i+1}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(themes)