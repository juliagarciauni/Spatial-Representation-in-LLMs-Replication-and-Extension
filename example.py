
import os
import torch
from data_loader import load_worldcities_dataset, load_paper_dataset
from embeddings import extract_embeddings, extraer_embeddings
from probes import entrenar_desde_h5, train
import plotting as plot


# 1. DATASET PREPARATION

print("--- [Phase 1/4] Loading Datasets ---")
dataset_paper = load_paper_dataset()       # For replication, linearity and Haversine
dataset_regional = load_worldcities_dataset() # For the regional bias study


# 2. MODEL CONFIGURATION AND EXTRACTION

# We define the models and their quantization configurations
modelos_info = {
    'Llama3':  {"repo": "meta-llama/Meta-Llama-3-8B", "quant": "4bit"},
    'Mistral': {"repo": "mistralai/Mistral-7B-v0.1", "quant": "4bit"},
    'Qwen':    {"repo": "Qwen/Qwen2.5-7B",           "quant": "8bit"} # 8-bit for better precision
}

print("\n--- [Phase 2/4] Extracting Embeddings ---")
for nombre, info in modelos_info.items():
    print(f"Processing {nombre}...")
    # If we already have the .h5 files saved, the function will skip this automatically.
    extract_embeddings(info["repo"], dataset_paper, config=(True if info["quant"]=="4bit" else None))


# 3. PROBE EXECUTION

# Dictionary where we will accumulate everything for the plots
results = {'Llama3': {}, 'Mistral': {}, 'Qwen': {}}

print("\n--- [Phase 3/4] Training Linear and Non-Linear Probes ---")
for nombre, info in modelos_info.items():
    ruta_h5 = f"matriz_{info['repo'].replace('/', '_')}.h5"

    # PAPER REPLICATION, HAVERSINE, NO_LINEAL
    # Here we activate non-linear to verify the paper's hypothesis. We also activate haversine to get the error in kilometers.
    res_paper = train( ruta_h5, dataset_paper, dataset_regional=False, haversine=True, no_lineal=True )
   
    # REGIONAL BIAS (USA, Europe, China)
    # Here we are only interested in the linear probe (Ridge) to compare regions and we use the regional dataset. 
    res_regional = train( ruta_h5, dataset_regional, dataset_regional=True, haversine=False, no_lineal=False)
        
    # We save everything in the global dictionary
    results[nombre].update(res_paper)
    results[nombre].update(res_regional)



# 4. VISUAL RESULTS GENERATION

print("\n--- [Phase 4/4] Generating Visualizations ---")

# --- STEP 1: Global Performance (Paper Dataset) ---
# We see how the 3 models perform on a global scale
plot.plot_paper_comparison(results)

# --- STEP 2: Linearity Test ---
# We verify if Ridge is similar to MLP in each model separately
plot.plot_check_linearity(results)

# --- STEP 3: Haversine Metric ---
# We translate the error into physical kilometers
plot.plot_haversine_error(results)

# --- STEP 4: Regional Analysis ---
# Layer-by-layer evolution in USA, Europe, and China
plot.plot_evolution_regions(results)

# --- STEP 5: Bias Plot (Maximums) ---
# Final bar comparison to see the spatial bias by model origin
plot.plot_maximums(results)

print("\n--- Experiment successfully completed ---")