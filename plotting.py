import matplotlib.pyplot as plt
import numpy as np

def plot_paper_comparison(results):
    # GRAPH 1: Overall Performance on Paper Dataset (Linear Probes only)
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['Llama3']['Dataset_Paper']['lin'], marker='o', color='b', linewidth=2, label="Llama-3-8B")
    plt.plot(results['Mistral']['Dataset_Paper']['lin'], marker='s', color='darkorange', linewidth=2, label="Mistral-7B")
   
    plt.title("Global Spatial Representation Accuracy (Paper Dataset)") 
    plt.xlabel("Layer Index")
    plt.ylabel("Prediction Accuracy $R^2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_check_linearity(results):
    # GRAPH 2A - Linearity Test: Llama 3
    plt.figure(figsize=(10, 6))
    plt.plot(results['Llama3']['Dataset_Paper']['lin'], marker='o', color='b', linewidth=2, label="Llama-3 (Linear)")
    plt.plot(results['Llama3']['Dataset_Paper']['non'], linestyle='--', color='b', alpha=0.5, linewidth=2, label="Llama-3 (Non-Linear)")
    plt.title("Linearity Test: Llama-3-8B (Paper Dataset)") 
    plt.xlabel("Layer Index")
    plt.ylabel("Prediction Accuracy $R^2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # GRAPH 2B - Linearity Test: Mistral
    plt.figure(figsize=(10, 6))
    plt.plot(results['Mistral']['Dataset_Paper']['lin'], marker='s', color='darkorange', linewidth=2, label="Mistral (Linear)")
    plt.plot(results['Mistral']['Dataset_Paper']['non'], linestyle='--', color='darkorange', alpha=0.5, linewidth=2, label="Mistral (Non-Linear)")
    plt.title("Linearity Test: Mistral-7B (Paper Dataset)") 
    plt.xlabel("Layer Index")
    plt.ylabel("Prediction Accuracy $R^2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_haversine_error(results):
    # GRAPH 3 - Haversine Distance (Llama, Mistral)
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['Llama3']['Dataset_Paper']['km'], marker='o', color='b', linewidth=2, label="Llama-3-8B")
    plt.plot(results['Mistral']['Dataset_Paper']['km'], marker='s', color='darkorange', linewidth=2, label="Mistral-7B")
    
    plt.title("Median Spatial Prediction Error (Haversine Distance)")
    plt.xlabel("Layer Depth")
    plt.ylabel("Median Error in Kilometers (Lower is Better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0)
    plt.show()

def plot_evolution_regions(results):
    # GRAPH 4: Evolution EUROPE
    plt.figure(figsize=(10, 6))
    plt.plot(results['Llama3']['Europa']['km'], marker='o', color='b', linewidth=2, label="Llama-3-8B")
    plt.plot(results['Mistral']['Europa']['km'], marker='s', color='darkorange', linewidth=2, label="Mistral-7B")
    plt.plot(results['Qwen']['Europa']['km'], marker='^', color='green', linewidth=2, label="Qwen-2.5-7B") 
    
    plt.title("Spatial Error by Layer: EUROPE")
    plt.xlabel("Layer Index")
    plt.ylabel("Median Error in Kilometers (Lower is better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0) 
    plt.show()

    # GRAPH 5: Evolution USA
    plt.figure(figsize=(10, 6))
    plt.plot(results['Llama3']['USA']['km'], marker='o', color='b', linewidth=2, label="Llama-3-8B")
    plt.plot(results['Mistral']['USA']['km'], marker='s', color='darkorange', linewidth=2, label="Mistral-7B")
    plt.plot(results['Qwen']['USA']['km'], marker='^', color='green', linewidth=2, label="Qwen-2.5-7B")
    
    plt.title("Spatial Error by Layer: USA")
    plt.xlabel("Layer Index")
    plt.ylabel("Median Error in Kilometers (Lower is better)") 
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0) 
    plt.show()

    # GRAPH 6: Evolution CHINA
    plt.figure(figsize=(10, 6))
    plt.plot(results['Llama3']['China']['km'], marker='o', color='b', linewidth=2, label="Llama-3-8B")
    plt.plot(results['Mistral']['China']['km'], marker='s', color='darkorange', linewidth=2, label="Mistral-7B")
    plt.plot(results['Qwen']['China']['km'], marker='^', color='green', linewidth=2, label="Qwen-2.5-7B")
    
    plt.title("Spatial Error by Layer: CHINA")
    plt.xlabel("Layer Index")
    plt.ylabel("Median Error in Kilometers (Lower is better)") 
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0) 
    plt.show()


def plot_maximums(results):
    # GRAPH 7: Comparison of maximums by region (best results from all layers)
    regiones = ['USA', 'Europa', 'China']
    
    llama_scores = [min(results['Llama3']['USA']['km']), min(results['Llama3']['Europa']['km']), min(results['Llama3']['China']['km'])]
    mistral_scores = [min(results['Mistral']['USA']['km']), min(results['Mistral']['Europa']['km']), min(results['Mistral']['China']['km'])]
    qwen_scores = [min(results['Qwen']['USA']['km']), min(results['Qwen']['Europa']['km']), min(results['Qwen']['China']['km'])]

    x = np.arange(len(regiones))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, llama_scores, width, label='Llama-3', color='blue')
    plt.bar(x, mistral_scores, width, label='Mistral', color='darkorange')
    plt.bar(x + width, qwen_scores, width, label='Qwen', color='green')

    plt.title("Best Spatial Accuracy (Minimum Error in km) by Region")
    plt.xlabel("Evaluated Region")
    plt.ylabel("Best Result (Kilometers of Error)")
    plt.xticks(x, regiones)
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.show()
