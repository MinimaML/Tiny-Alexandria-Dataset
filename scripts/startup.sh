#!/bin/bash
set -e

# Bulletproof Startup Script
echo "Starting Setup..."

# 1. System Updates
apt-get update -qq
apt-get install -y git zip python3-pip

# 2. Clone Repository
# Uses GITHUB_REPO env var if present (e.g. MinimaML/Tiny-Alexandria-Dataset)
if [ ! -z "$GITHUB_REPO" ]; then
    REPO_URL="https://github.com/${GITHUB_REPO}.git"
    REPO_DIR=$(basename "$GITHUB_REPO")
else
    REPO_URL="https://github.com/MinimaML/Tiny-Alexandria-Dataset.git"
    REPO_DIR="Tiny-Alexandria-Dataset"
fi

if [ -d "$REPO_DIR" ]; then
    echo "Directory $REPO_DIR exists. Pulling latest..."
    cd $REPO_DIR
    git pull
else
    echo "Cloning $REPO_URL..."
    git clone $REPO_URL
    cd $REPO_DIR
fi

# 3. Python Dependencies (Explicit Install)
echo "Installing Python Libraries..."
python3 -m pip install --upgrade pip
python3 -m pip install vllm datasets huggingface_hub torch transformers accelerate tqdm bitsandbytes scipy

# --- TEMPLATE INJECTION ---
# Ensure templates exist (self-healing)
mkdir -p alexandria/templates

echo "Generating Templates..."

cat <<EOF > alexandria/templates/refine_academic.md

# ROLE
You are an expert educator and logician.

# ATTENTION
You will be given a raw text excerpt (SOURCE). Your goal is to **rewrite** and **expand** this content into a **Reasoning-Dense Textbook Section**.

# INSTRUCTION
1.  **Analyze** the key concepts in the SOURCE.
2.  **Rewrite** the content as if you are explaining it to a brilliant student.
3.  **Crucial**: You must include an **Internal Monologue** or **Thought Process** block before your final explanation, where you break down the logic, check for connections, and verify facts.
4.  If the source is factual, ensure you explain *why* it is true.
5.  If the source is a problem, solve it step-by-step.

# FORMAT
<|begin_of_thought|>
(Your internal reasoning, connecting concepts, checking prerequisites, planning the explanation)
<|end_of_thought|>

# (Title of Concept)
(Your expanded, clear, and educational explanation)

# SOURCE
{text}
EOF

cat <<EOF > templates/refine_code.md

# ROLE
You are a Staff Principal Engineer.

# INSTRUCTION
You will be given a CODE SNIPPET or PROBLEM (SOURCE).
Your task is to **explain**, **optimize**, or **complete** it.
You must demonstrate rigorous logical thinking.

# FORMAT
<|begin_of_thought|>
(Analyze the code complexity. Check for bugs or edge cases. Plan the optimization or explanation strategy. pseudo-code solution.)
<|end_of_thought|>

(The final code and explanation)

# SOURCE
{text}
EOF

cat <<EOF > templates/refine_creative.md

# ROLE
You are an award-winning author and creative writing coach.

# INSTRUCTION
You will be given a SHORT STORY START or CONCEPT (SOURCE).
Your task is to **write a complete, engaging story** based on it.
**However**, you must first plan the narrative arc.

# FORMAT
<|begin_of_thought|>
(Outline the plot points. Define the characters' motivations. Plan the twist or climax. Ensure the tone is consistent.)
<|end_of_thought|>

(The story)

# SOURCE
{text}
EOF

cat <<EOF > templates/refine_instruction.md

# ROLE
You are an AI Assistant capable of deep internal reasoning.

# INSTRUCTION
You will be given a USER INSTRUCTION (SOURCE).
Your task is to provide the **Best Possible Response**, but you must **Thinking Aloud** first.
You must explore edge cases, plan your response, and ensure safety/helpfulness before outputting the final answer.

# FORMAT
<|begin_of_thought|>
(Analyze the user's intent. Check for ambiguity. Plan the structure of the response. Draft code mentall if needed.)
<|end_of_thought|>

(The actual helpful response to the user)

# SOURCE
{text}
EOF

cat <<EOF > templates/refine_memory.md

# ROLE
You are a Memory Specialist designed to test Long-Context Recall.

# INSTRUCTION
You will be given a text (SOURCE).
Your task is to generate a **Long-Context Retrieval Task**.
1.  Identify a specific, small detail in the text (the "Needle").
2.  Create a Question that requires finding that detail.
3.  Rewrite the text or pad it to ensure the detail is buried in the middle.
4.  The Goal is to force the model to look back and retrieve the exact information.

# FORMAT
<|begin_of_thought|>
(Identify the needle. Plan the distraction/padding text. Formulate the question.)
<|end_of_thought|>

Context:
(The long text with the buried needle)

Question: (The question asking for the needle)
Answer: (The exact detail)

# SOURCE
{text}
EOF

echo "Templates Generated."

# 4. Validation
if [ ! -f "scripts/curate_alexandria.py" ]; then
    echo "Error: scripts/curate_alexandria.py not found in $(pwd)"
    ls -R
    exit 1
fi

# 5. Execution
echo "Running Curation..."
# Note: curate_alexandria.py output dir is relative to CWD.
# Switched to bitsandbytes (4-bit) to ensure 80B MoE fits in 141GB VRAM.
python3 scripts/curate_alexandria.py --total_samples 100000 --model "Qwen/Qwen3-Next-80B-A3B-Instruct" --tensor_parallel 1 --quantization bitsandbytes

# 6. Archiving
echo "Zipping Data..."
if [ -d "data/alexandria" ]; then
    zip -r alexandria_100k.zip data/alexandria
    echo "Done. File: alexandria_100k.zip"
    
    # Optional: Auto-Upload to HF
    if [ ! -z "$HF_TOKEN" ] && [ ! -z "$HF_REPO_ID" ]; then
        echo "Uploading to HuggingFace ($HF_REPO_ID)..."
        huggingface-cli login --token "$HF_TOKEN"
        huggingface-cli upload "$HF_REPO_ID" alexandria_100k.zip --repo-type dataset
        echo "Upload Complete."
    fi
else
    echo "Error: No data directory found"
    ls -R
    exit 1
fi
