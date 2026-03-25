import matplotlib.pyplot as plt
import numpy as np

def plot_evolution_regions(results):
    # GRAPH 1: Evolution EUROPE
    plt.figure(figsize=(10, 6))
    plt.plot(results['Llama3']['Europa']['lin'], marker='o', color='b', linewidth=2, label="Llama-3-8B")
    plt.plot(results['Mistral']['Europa']['lin'], marker='s', color='darkorange', linewidth=2, label="Mistral-7B")
    plt.plot(results['Qwen']['Europa']['lin'], marker='^', color='green', linewidth=2, label="Qwen-2.5-7B")
    plt.title("Spatial Prediction Accuracy by Layer: EUROPE")
    plt.xlabel("Layer Index")
    plt.ylabel("Prediction Accuracy $R^2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # GRAPH 2: Evolution USA
    plt.figure(figsize=(10, 6))
    plt.plot(results['Llama3']['USA']['lin'], marker='o', color='b', linewidth=2, label="Llama-3-8B")
    plt.plot(results['Mistral']['USA']['lin'], marker='s', color='darkorange', linewidth=2, label="Mistral-7B")
    plt.plot(results['Qwen']['USA']['lin'], marker='^', color='green', linewidth=2, label="Qwen-2.5-7B")
    plt.title("Spatial Prediction Accuracy by Layer: USA")
    plt.xlabel("Layer Index")
    plt.ylabel("Prediction Accuracy $R^2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # GRAPH 3: Evolution CHINA
    plt.figure(figsize=(10, 6))
    plt.plot(results['Llama3']['China']['lin'], marker='o', color='b', linewidth=2, label="Llama-3-8B")
    plt.plot(results['Mistral']['China']['lin'], marker='s', color='darkorange', linewidth=2, label="Mistral-7B")
    plt.plot(results['Qwen']['China']['lin'], marker='^', color='green', linewidth=2, label="Qwen-2.5-7B")
    plt.title("Spatial Prediction Accuracy by Layer: CHINA")
    plt.xlabel("Layer Index")
    plt.ylabel("Prediction Accuracy $R^2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_maximums(results):
    # GRAPH 4: Comparison of maximums by region (best results from all layers)
    regiones = ['USA', 'Europa', 'China']
    llama_scores = [max(results['Llama3']['USA']['lin']), max(results['Llama3']['Europa']['lin']), max(results['Llama3']['China']['lin'])]
    mistral_scores = [max(results['Mistral']['USA']['lin']), max(results['Mistral']['Europa']['lin']), max(results['Mistral']['China']['lin'])]
    qwen_scores = [max(results['Qwen']['USA']['lin']), max(results['Qwen']['Europa']['lin']), max(results['Qwen']['China']['lin'])]

    x = np.arange(len(regiones))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, llama_scores, width, label='Llama-3', color='blue')
    plt.bar(x, mistral_scores, width, label='Mistral', color='darkorange')
    plt.bar(x + width, qwen_scores, width, label='Qwen', color='green')

    plt.title("Maximum Prediction Accuracy ($R^2$) by Region")
    plt.xlabel("Evaluated Region")
    plt.ylabel("Best $R^2$ Score")
    plt.xticks(x, regiones)
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.show()


def plot_paper_comparison(results):
    # GRAPH 5 - Overall Performance on Paper Dataset (Linear Probes only)
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['Llama3']['Dataset_Paper']['lin'], marker='o', color='b', linewidth=2, label="Llama-3-8B")
    plt.plot(results['Mistral']['Dataset_Paper']['lin'], marker='s', color='darkorange', linewidth=2, label="Mistral-7B")
    plt.plot(results['Qwen']['Dataset_Paper']['lin'], marker='^', color='green', linewidth=2, label="Qwen-2.5-7B")

    plt.title("Global Spatial Representation Accuracy (Paper Dataset)") 
    plt.xlabel("Layer Index")
    plt.ylabel("Prediction Accuracy $R^2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_check_linearity(results):
    # GRAPH 6A - Linearity Test: Llama 3
    plt.figure(figsize=(10, 6))
    plt.plot(results['Llama3']['Dataset_Paper']['lin'], marker='o', color='b', linewidth=2, label="Llama-3 (Linear)")
    plt.plot(results['Llama3']['Dataset_Paper']['non'], linestyle='--', color='b', alpha=0.5, linewidth=2, label="Llama-3 (Non-Linear)")
    plt.title("Linearity Test: Llama-3-8B (Paper Dataset)") 
    plt.xlabel("Layer Index")
    plt.ylabel("Prediction Accuracy $R^2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # GRAPH 6B - Linearity Test: Mistral
    plt.figure(figsize=(10, 6))
    plt.plot(results['Mistral']['Dataset_Paper']['lin'], marker='s', color='darkorange', linewidth=2, label="Mistral (Linear)")
    plt.plot(results['Mistral']['Dataset_Paper']['non'], linestyle='--', color='darkorange', alpha=0.5, linewidth=2, label="Mistral (Non-Linear)")
    plt.title("Linearity Test: Mistral-7B (Paper Dataset)") 
    plt.xlabel("Layer Index")
    plt.ylabel("Prediction Accuracy $R^2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # GRAPH 6C - Linearity Test: Qwen
    plt.figure(figsize=(10, 6))
    plt.plot(results['Qwen']['Dataset_Paper']['lin'], marker='^', color='green', linewidth=2, label="Qwen (Linear)")
    plt.plot(results['Qwen']['Dataset_Paper']['non'], linestyle='--', color='green', alpha=0.5, linewidth=2, label="Qwen (Non-Linear)")
    plt.title("Linearity Test: Qwen-2.5-7B (Paper Dataset)") 
    plt.xlabel("Layer Index")
    plt.ylabel("Prediction Accuracy $R^2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_haversine_error(results):
    # GRAPH 7 - Haversine Distance (Llama, Mistral, Qwen)
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['Llama3']['Dataset_Paper']['km'], marker='o', color='b', linewidth=2, label="Llama-3-8B")
    plt.plot(results['Mistral']['Dataset_Paper']['km'], marker='s', color='darkorange', linewidth=2, label="Mistral-7B")
    plt.plot(results['Qwen']['Dataset_Paper']['km'], marker='^', color='green', linewidth=2, label="Qwen-2.5-7B")
    
    plt.title("Median Spatial Prediction Error (Haversine Distance)")
    plt.xlabel("Layer Depth")
    plt.ylabel("Median Error in Kilometers (Lower is Better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0)
    plt.show()