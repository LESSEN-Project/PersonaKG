import os
import json
import random
import argparse
from tqdm import tqdm
from dataset import get_dataset
from evaluate import load
import re
from prompts import get_next_utterance_prompt

from construct_kg import load_knowledge_graph_from_file
from models import LLM

def setup_args():
    parser = argparse.ArgumentParser(description='Run next utterance prediction experiment')
    parser.add_argument('--model', type=str, default='GPT-4.1-mini')
    parser.add_argument('--split', type=str, default='test', 
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to use for evaluation')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Number of samples to use for evaluation. -1 for all')

    parser.add_argument('--output_dir', type=str, default='results',
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

def create_prediction_samples(data, split='test', num_samples=-1):
    samples = []
    dataset_split = data[split]
    
    sample_size = len(dataset_split) if num_samples == -1 else min(num_samples, len(dataset_split))
    indices = random.sample(range(len(dataset_split)), sample_size)
    
    for idx in indices:
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

def predict_next_utterance(sample, llm):
    user1_persona = sample['user1_persona']
    user2_persona = sample['user2_persona']
    history = sample['history']
    target_speaker = sample['target_speaker']
    
    formatted_history = ""
    for utterance in history:
        formatted_history += f"{utterance['speaker']}: {utterance['text']}\n"
    
    prompt = get_next_utterance_prompt(
        user1_persona=user1_persona,
        user2_persona=user2_persona,
        conversation_history=formatted_history,
        target_speaker=target_speaker
    )
    
    prediction = llm.generate(prompt)
    
    prediction = re.sub(r'^.*?:', '', prediction).strip()
    
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

def run_experiment(args):
    data = get_dataset()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Creating prediction samples from {args.split} split...")
    samples = create_prediction_samples(
        data, 
        split=args.split, 
        num_samples=args.num_samples
    )
    samples = samples[:20]
    
    print(f"Running predictions with model {args.model}...")
    llm = LLM(args.model, gen_params={
        "temperature": 0.7,
        "max_tokens": 128
    })
    predictions = []
    targets = []
    
    for sample in tqdm(samples):
        prediction = predict_next_utterance(
            sample, 
            llm=llm
        )
        
        predictions.append(prediction)
        targets.append(sample['target'])
    
    print("Evaluating predictions...")
    results = evaluate_predictions(predictions, targets)
    
    output_file = os.path.join(
        args.output_dir, 
        f"results_{args.split}.json"
    )
    
    result_data = {
        'args': vars(args),
        'metrics': results,
        'samples': [
            {
                'history': [f"{u['speaker']}: {u['text']}" for u in s['history']],
                'target': s['target'],
                'prediction': p
            }
            for s, p in zip(samples, predictions)
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print(f"BLEU Score: {results['bleu']}")
    print(f"ROUGE-F1 Score: {results['rouge']}")
    
    return results

if __name__ == "__main__":
    # kg = load_knowledge_graph_from_file("saved_results/canonized_results_69f0eb0369fca5b1acff7a9964224e74.json", os.environ.get("NEO4J_PKG_PASSWORD"))
    args = setup_args()
    run_experiment(args)