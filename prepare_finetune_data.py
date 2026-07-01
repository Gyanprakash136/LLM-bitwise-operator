import json
import random
import argparse
from typing import List, Dict

# Dummy function to represent Gemini API call for synthetic data
def generate_synthetic_pairs(num_pairs: int) -> List[Dict]:
    print(f"Generating {num_pairs} synthetic contrastive pairs via Gemini API...")
    # In a real scenario, this would call Gemini API with JD and Candidate Schema
    # to create borderline candidates and honeypots.
    synthetic_data = []
    for i in range(num_pairs):
        synthetic_data.append({
            "instruction": "Given this job description and candidate profile, score fit 0-100 and explain in 1-2 sentences.",
            "input": f"JD: {{structured_jd}}\nCandidate: Synthetic_{i}, AI Engineer, 6 yrs, Skills: [Python, FAISS, LangChain], GitHub: 80, Response Rate: 90%, Last Active: 1 days ago",
            "output": f"{random.randint(60, 95)} | Strong match with solid semantic retrieval experience and active platform engagement."
        })
    return synthetic_data

def prepare_finetune_data(candidates_file: str, gold_pairs_file: str, output_file: str, num_synthetic: int = 500):
    print("Loading gold pairs...")
    training_examples = []
    
    # In a real scenario, we'd read gold_pairs_file here.
    # For now, we simulate finding 2000 pairs.
    print(f"Simulating extraction of 2000 contrastive pairs from {gold_pairs_file}")
    
    for i in range(2000):
        training_examples.append({
            "instruction": "Given this job description and candidate profile, score fit 0-100 and explain in 1-2 sentences.",
            "input": f"JD: {{structured_jd}}\nCandidate: Gold_{i}, Senior ML Engineer, 7 yrs, Skills: [PyTorch, Pinecone, AWS], GitHub: 95, Response Rate: 98%, Last Active: 2 days ago",
            "output": f"{random.randint(70, 99)} | Excellent fit for Senior ML role with deep production embedding experience and high responsiveness."
        })
        
    if num_synthetic > 0:
        synthetic_examples = generate_synthetic_pairs(num_synthetic)
        training_examples.extend(synthetic_examples)
        
    random.shuffle(training_examples)
    
    print(f"Writing {len(training_examples)} training examples to {output_file}...")
    with open(output_file, 'w') as f:
        for ex in training_examples:
            f.write(json.dumps(ex) + "\n")
            
    print("Data preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare finetuning data for ranking model.")
    parser.add_argument("--candidates", default="India_runs_data_and_ai_challenge/candidates.jsonl", help="Path to candidates file")
    parser.add_argument("--gold_pairs", default="India_runs_data_and_ai_challenge/gold_ranked_pairs.csv", help="Path to competition provided ranked pairs")
    parser.add_argument("--out", default="India_runs_data_and_ai_challenge/finetune_dataset.jsonl", help="Output JSONL for training")
    parser.add_argument("--synthetic", type=int, default=500, help="Number of synthetic pairs to generate")
    
    args = parser.parse_args()
    prepare_finetune_data(args.candidates, args.gold_pairs, args.out, args.synthetic)
