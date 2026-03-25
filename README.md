# Spatial Representation in LLMs: Replication and Extension

This repository contains a replication and extension of the paper *"Language Models Represent Space and Time"* by Wes Gurnee and Max Tegmark.

The original study demonstrates that Large Language Models (LLMs) internally learn linear representations of geographic coordinates and temporal events. While the original authors focused primarily on the Llama-2 family, this project aims to verify their hypothesis on more modern architectures and extends the research with new physical metrics and geographical bias analyses.

## Replication
Firstly, we replicated the methodology of the original paper. Using one of the official datasets, we extracted the hidden states of **Llama-3 (8B)** and trained both linear (Ridge regression) and non-linear (MLP) probes to predict the latitude and longitude of 5,000 global cities. Our results successfully confirm the original authors' hypothesis: the model encodes spatial geography in a highly linear manner, as the linear probe performs just as well as the non-linear one in the mid-to-late layers.

## Main Contributions
This project introduces the following contributions:

* **Multi-Model Testing:** We ran the same experiment on Mistral (7B) and Qwen-2.5 (7B), checking both the models' performance and the linearity.
* **Physical Error Tracking (Haversine Distance):** To give the statistical accuracy ($R^2$) a physical meaning, we implemented the Haversine formula. This allows us to measure the median spatial prediction error in actual kilometers, providing a more intuitive understanding of the models' internal maps. 
* **Geographical Bias Analysis:** We investigated if a model's spatial representation is biased by its origin. From the `worldcities` dataset, we classified cities into three regions (USA, Europe, and China) to evaluate if models perform significantly better at mapping cities located in their creators' geographic regions. 

## Project Structure

* **`data_loader.py`:** Processes the paper's dataset (filtering to 5,000 cities) and the worldcities dataset, creating three new regional datasets by extracting cities from USA, Europe, and China. 
* **`embeddings.py`:** Connects to HuggingFace to extract and store the hidden states (activations) of the models layer by layer using `.h5` files.
* **`probes.py`:** Contains the core logic for training the Ridge and MLP regressors, as well as calculating the Haversine physical distance.
* **`plotting.py`:** Generates all the evaluation plots. 
* **`main_experiment.py`:** The main script that runs the data loading, extraction, probing and plotting sequentially. 

## Installation and Setup
To run this project locally or in Google Colab, you will need Python 3.8+ and the following dependencies. First, clone the repository:

```bash
git clone [https://github.com/juliagarciauni/Spatial-Representation-in-LLMs-Replication-and-Extension.git](https://github.com/juliagarciauni/Spatial-Representation-in-LLMs-Replication-and-Extension.git)
cd Spatial-Representation-in-LLMs-Replication-and-Extension
```
Then, install the required libraries:
```bash
pip install torch transformers pandas numpy scikit-learn scikit-learn-intelex h5py matplotlib tqdm
````
**Note:**  To extract the embeddings, you will need access to a GPU and enough RAM/VRAM depending on the model quantization.

## Usage
Once the dependencies are installed and the datasets are in the root folder, simply run the main script:

```bash
python main_experiment.py
```
The script will automatically process the datasets, extract the embeddings, train the linear and non-linear probes across all layers, and generate the final evaluation plots.

## References
* Wes Gurnee and Max Tegmark (2024). Language Models Represent Space and Time.
