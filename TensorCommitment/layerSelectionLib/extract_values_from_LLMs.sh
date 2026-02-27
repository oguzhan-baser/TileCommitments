conda init
cd layer_selection_integration
conda activate alphaprun
# Models of interest:
# - OPT-125M
# - LLaMA-7B
# - LLaMA-13B
# - LLaMA-30B
# - LLaMA-65B
# - LLaMA-2-7B
# - LLaMA-2-13B
# - LLaMA-2-70B

mkdir -p ../scores
mkdir -p ../plots
# Extract scores for OPT-125M
python extract_layer_scores.py \
    --model facebook/opt-125m \
    --output ../scores/opt125m_scores.json \
    --pretty

# Visualize single model
python visualize_scores.py \
    --mode json_single \
    --json_file ../scores/opt125m_scores.json \
    --output ../plots/opt125m_plot.png

# Extract scores for LLaMA-7B
python extract_layer_scores.py \
    --model meta-llama/Llama-7B-hf \
    --output ../scores/llama7b_scores.json \
    --pretty

# Visualize single model
python visualize_scores.py \
    --mode json_single \
    --json_file ../scores/llama7b_scores.json \
    --output ../plots/llama7b_plot.png


# Extract scores for LLaMA-13B
python extract_layer_scores.py \
    --model meta-llama/Llama-13B-hf \
    --output ../scores/llama13b_scores.json \
    --pretty

# Visualize single model
python visualize_scores.py \
    --mode json_single \
    --json_file ../scores/llama13b_scores.json \
    --output ../plots/llama13b_plot.png

# Extract scores for LLaMA-30B
python extract_layer_scores.py \
    --model meta-llama/Llama-30B-hf \
    --output ../scores/llama30b_scores.json \
    --pretty

# Visualize single model
python visualize_scores.py \
    --mode json_single \
    --json_file ../scores/llama30b_scores.json \
    --output ../plots/llama30b_plot.png

# Extract scores for LLaMA-65B
python extract_layer_scores.py \
    --model meta-llama/Llama-65B-hf \
    --output ../scores/llama65b_scores.json \
    --pretty

# Visualize single model
python visualize_scores.py \
    --mode json_single \
    --json_file ../scores/llama65b_scores.json \
    --output ../plots/llama65b_plot.png


# Extract scores for LLaMA-2-7B
python extract_layer_scores.py \
    --model meta-llama/Llama2-7B-hf \
    --output ../scores/llama2-7b_scores.json \
    --pretty

# Visualize single model
python visualize_scores.py \
    --mode json_single \
    --json_file ../scores/llama2-7b_scores.json \
    --output ../plots/llama2-7b_plot.png

#  Extract scores for LLaMA-2-13B
python extract_layer_scores.py \
    --model meta-llama/Llama2-13B-hf \
    --output ../scores/llama2-13b_scores.json \
    --pretty

# Visualize single model
python visualize_scores.py \
    --mode json_single \
    --json_file ../scores/llama2-13b_scores.json \
    --output ../plots/llama2-13b_plot.png

# Extract scores for LLaMA-2-70B
python extract_layer_scores.py \
    --model meta-llama/Llama2-70B-hf \
    --output ../scores/llama2-70b_scores.json \
    --pretty

# Visualize single model
python visualize_scores.py \
    --mode json_single \
    --json_file ../scores/llama2-70b_scores.json \
    --output ../plots/llama2-70b_plot.png
    

# # Visualize ALL models from data folder
python visualize_scores.py \
    --mode data_dir \
    --data_dir ../AlphaPruning/data/ \
    --metric alpha_peak \
    --output ../plots/all_models_comparison.png