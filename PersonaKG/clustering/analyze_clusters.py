import argparse
import os
import re
import random
import json

from PersonaKG.prompts import get_cluster_themes
from PersonaKG.models import LLM

def get_sample_statements(cluster_path):
    with open(cluster_path, "r", encoding="utf-8") as f:
        cluster_data = f.readlines()
    statement_lines = [line[line.find(".")+1:].strip() for line in cluster_data if re.match(r"^\d+\. ", line)]
    return statement_lines

def robust_llm_json_generate(llm, prompt_params, gen_params, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            result = llm.generate(prompt_params=prompt_params, gen_params=gen_params, json_output=True)
            if isinstance(result, dict):
                return result
            elif isinstance(result, str):
                return json.loads(result)
        except Exception:
            pass
    # fallback: try one last time, return as string if all else fails
    try:
        result = llm.generate(prompt_params=prompt_params, gen_params=gen_params, json_output=True)
        return result if isinstance(result, dict) else {"raw_output": result}
    except Exception as e:
        return {"error": str(e)}

parser = argparse.ArgumentParser(description="Analyze clusters")
add = parser.add_argument
add("-m", "--mode", type=str, default="combined", choices=["combined", "separate"], help="Clustering mode: combined (all datasets together), or separate (per dataset)")
add("-l", "--llm", type=str, default="QWEN-3-32B-GGUF", help="LLM to use for analysis")
add("-ca", "--clustering_algorithm", type=str, default="hdbscan", choices=["agglomerative", "hdbscan"], help="Which clustering algorithm to analyze")
add("-o", "--output_dir", type=str, default="clustering/cluster_analysis", help="Output directory for results")
args = parser.parse_args()

output_dir = os.path.join(args.output_dir, args.clustering_algorithm, args.mode)
os.makedirs(output_dir, exist_ok=True)
json_path = os.path.join(output_dir, "cluster_themes.json")

llm = LLM(model_name=args.llm, default_prompt=get_cluster_themes())

if llm.cfg.get("reason"):
    gen_params = {
        "max_new_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0
    }
else:
    gen_params = {
        "max_new_tokens": 512
    }

input_dir = os.path.join(f"clustering/{args.clustering_algorithm}_clusters", args.mode)
max_statements = 200

clusters = [cluster for cluster in os.listdir(input_dir) if cluster.startswith("cluster_")]
print(f"Found {len(clusters)} clusters")

results = {}

for cluster in clusters:
    cluster_num_match = re.search(r"cluster_(\d+)", cluster)
    cluster_num = cluster_num_match.group(1) if cluster_num_match else cluster
    statements = get_sample_statements(os.path.join(input_dir, cluster))
    print(f"Found {len(statements)} statements in {cluster.split('.')[0]}\n")
    if len(statements) > max_statements:
        statements = random.sample(statements, max_statements)
    themes = robust_llm_json_generate(llm, {"statements": "\n".join(statements)}, gen_params)
    results[cluster_num] = {
        "themes": themes
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if args.clustering_algorithm == "hdbscan":
    noise_file = os.path.join(input_dir, "noise_points.txt")
    print("Processing noise points")
    if os.path.exists(noise_file):
        noise_statements = get_sample_statements(noise_file)
        if len(noise_statements) > max_statements:
            noise_categories = len(noise_statements) // max_statements
            for i in range(noise_categories):
                print(f"Processing noise category {i+1}/{noise_categories}")
                chunk = noise_statements[i*max_statements:(i+1)*max_statements]
                themes = robust_llm_json_generate(llm, {"statements": "\n".join(chunk)}, gen_params)
                results[f"noise_{i+1}"] = {
                    "themes": themes
                }
            remainder = noise_statements[noise_categories*max_statements:]
            if remainder:
                themes = robust_llm_json_generate(llm, {"statements": "\n".join(remainder)}, gen_params)
                results[f"noise_{noise_categories+1}"] = {
                    "themes": themes
                }
        else:
            themes = robust_llm_json_generate(llm, {"statements": "\n".join(noise_statements)}, gen_params)
            results["noise"] = {
                "themes": themes
            }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)