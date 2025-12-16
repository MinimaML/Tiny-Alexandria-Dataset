import argparse
import os
import json
import time
import random
import glob
from typing import List, Dict


# STRICT IMPORTS
try:
    from datasets import load_dataset
    import torch
    from vllm import LLM, SamplingParams
    import tqdm
except ImportError as e:
    print(f"[!] Critical Import Error: {e}")
    sys.exit(1)

# --- HARDWARE CHECK ---
if not torch.cuda.is_available():
    print("[!] CRITICAL ERROR: CUDA is not available.")
    sys.exit(1)

# --- CONFIGURATION ---
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct" # Updated default
OUTPUT_DIR = "data/alexandria"
STATE_FILE = "curation_state.json"
TEMPLATE_DIR = "templates"

TEMPLATES = {
    "academic": "refine_academic.md",
    "instruction": "refine_instruction.md",
    "creative": "refine_creative.md",
    "code": "refine_code.md",
    "memory": "refine_memory.md"
}

BUCKET_MAP = {
    "academic": "bucket_a", 
    "instruction": "bucket_b",
    "code": "bucket_c",
    "creative": "bucket_d",
    "memory": "bucket_mem"
}

def load_template(name):
    path = os.path.join(TEMPLATE_DIR, name)
    if not os.path.exists(path):
        # Fallback if templates not moved correctly
        path = os.path.join("alexandria", "templates", name)
        
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_source_stream(source_type, skip_n=0):
    """
    Returns a generator created from the dataset.
    Skip N is approximate since random sampling makes perfect resume hard.
    But for streaming datasets, we can use skip(n) to advance iterator.
    """
    print(f"  -> Initializing stream for {source_type} (Skipping {skip_n})...")
    try:
        if source_type == "academic":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True).skip(skip_n)
            for row in ds: yield row['text']
        elif source_type == "instruction":
            ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).skip(skip_n)
            for row in ds:
                messages = row['messages']
                if len(messages) > 0 and messages[0]['role'] == 'user':
                    yield messages[0]['content']
        elif source_type == "creative":
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True).skip(skip_n)
            for row in ds: yield row['text']
        elif source_type == "code":
             ds = load_dataset("mbpp", split="train", streaming=True).skip(skip_n) # Smaller, might loop
             while True: # Loop if small
                 for row in ds: yield row['text']
        elif source_type == "memory":
             ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True).skip(skip_n)
             for row in ds:
                 if len(row['text']) > 2000: yield row['text']
        else:
            yield "Unknown Source"
    except Exception as e:
        print(f"[!] Error creating stream {source_type}: {e}")
        raise e

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {k: 0 for k in TEMPLATES.keys()}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def curate_alexandria(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Init Checkpoint State
    state = load_state()
    total_existing = sum(state.values())
    target_samples = args.total_samples
    
    print(f"=== Alexndria Curation ===")
    print(f"Target: {target_samples}")
    print(f"Resuming from: {total_existing}")
    
    if total_existing >= target_samples:
        print("Target already reached!")
        return

    # Init LLM
    print(f"Initializing Model: {args.model} (TP={args.tensor_parallel})...")
    # Suggestion: enforce max_model_len to avoid OOM on 80GB cards with 72B/80B models if context is huge.
    # 8192 is plenty for training data generation.
    llm = LLM(
        model=args.model, 
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=8192, 
        gpu_memory_utilization=0.90, # Reduced to leave room for Sampler Buffer
        max_num_seqs=256, # Limit concurrent sequences to save overhead
        quantization=args.quantization
    )
    
    # Load Templates
    loaded_templates = {k: load_template(v) for k, v in TEMPLATES.items()}
    
    # Init Streams with resume skip
    streams = {
        k: get_source_stream(k, skip_n=state[k]) for k in TEMPLATES.keys()
    }
    
    pbar = tqdm.tqdm(total=target_samples, initial=total_existing, desc="Curating")
    
    while total_existing < target_samples:
        # Mix Weights
        category = random.choices(
            ["academic", "instruction", "creative", "code", "memory"],
            weights=[35, 25, 15, 10, 15],
            k=1
        )[0]
        
        batch_size = 50
        # If nearly done, reduce batch
        remaining = target_samples - total_existing
        if remaining < batch_size:
            batch_size = remaining
            
        batch_prompts = []
        batch_sources = []
        
        # Fetch Data
        stream = streams[category]
        fetched = 0
        try:
            for _ in range(batch_size):
                text = next(stream)
                # Length Filter
                if len(text) > 50 and len(text) < 6000:
                   tmpl = loaded_templates[category]
                   prompt = tmpl.replace("{text}", text)
                   batch_prompts.append(prompt)
                   batch_sources.append(text)
                   fetched += 1
                state[category] += 1 # Count skipped ones too to keep stream sync roughly
        except StopIteration:
            print(f"\n[!] Stream exhausted: {category}")
            continue
            
        if not batch_prompts:
            continue
            
        # Refine
        outputs = llm.generate(batch_prompts, SamplingParams(temperature=0.7, max_tokens=2048, stop=["<|eot_id|>"]))
        
        # Save
        bucket_file = os.path.join(OUTPUT_DIR, f"{BUCKET_MAP[category]}.jsonl")
        with open(bucket_file, "a", encoding="utf-8") as f:
            for i, out in enumerate(outputs):
                entry = {
                    "text": f"{batch_sources[i]}\n\n{out.outputs[0].text}", # Pre-training Standard
                    "prompt": batch_sources[i],
                    "completion": out.outputs[0].text,
                    "category": category,
                    "curated": True
                }
                f.write(json.dumps(entry) + "\n")
        
        # Verbose Logging (Sample 1 per batch)
        if random.random() < 0.2: # 20% chance to show a sample log
            tqdm.tqdm.write(f"\n[Preview - {category.upper()}]")
            tqdm.tqdm.write(f"Source: {batch_sources[0][:100]}...")
            tqdm.tqdm.write(f"Result: {outputs[0].outputs[0].text[:100]}...\n")

        total_existing += len(batch_prompts)
        pbar.update(len(batch_prompts))
        save_state(state) # Checkpoint line position

    pbar.close()
    print("\n[Success] Curation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--total_samples", type=int, default=100000)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--quantization", type=str, default=None, help="e.g. 'fp8' or 'bitsandbytes'")
    args = parser.parse_args()
    
    curate_alexandria(args)
