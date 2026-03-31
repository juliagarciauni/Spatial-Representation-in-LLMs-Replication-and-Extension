# Spatial Representation in LLMs: Replication and Extension

This repository contains a replication and extension of the paper *"Language Models Represent Space and Time"* by Wes Gurnee and Max Tegmark, ICLR 2024.

The original study demonstrates that Large Language Models (LLMs) internally learn linear representations of geographic coordinates and temporal events. While the original authors focused primarily on the Llama-2 family, this project aims to verify their hypothesis on more modern architectures and extends the research with new physical metrics and geographical bias analyses.

## Replication
Firstly, we replicated the methodology of the original paper. Using one of the official datasets, we extracted the hidden states of **Llama-3 (8B)** and **Mistral (7B)** and trained both linear (Ridge regression) and non-linear (MLP) probes to predict the latitude and longitude of 5,000 global cities. Our results successfully confirm the original authors' hypothesis: the model encodes spatial geography in a highly linear manner, as the linear probe performs just as well as the non-linear one in the mid-to-late layers.

## Main Contributions
This project introduces the following contributions:

* **Multi-Model Testing:** We ran the same experiment on Mistral (7B) and LLama-3 (8B).
* **Physical Error Tracking (Haversine Distance):** To give the statistical accuracy ($R^2$) a physical meaning, we implemented the Haversine formula. This allows us to measure the median spatial prediction error in actual kilometers, providing a more intuitive understanding of the models' internal maps. 
* **Geographical Bias Analysis:** We investigated if a model's spatial representation is biased by its origin. From the `worldcities` dataset, we classified cities into three regions (USA, Europe, and China) to evaluate if models perform significantly better at mapping cities located in their creators' geographic regions. 

## Project Structure

* **`data_loader.py`:** Processes the paper's dataset (filtering to 5,000 cities) and the worldcities dataset.
* **`embeddings.py`:** Extracts and persists the hidden states (activations) from the models' layers into .h5 files.
* **`probes.py`:** Contains the core logic for training the Ridge and MLP regressors, as well as calculating the Haversine physical distance.
* **`plotting.py`:** Generates all the evaluation plots. 
* **`main_experiment.py`:** The main script that runs the data loading, extraction, probing and plotting sequentially. 

## Results and Evaluation
This section presents the empirical findings from our replication and extension experiments.

### 1. Global Spatial Representation (Replication)
<img width="702" height="453" alt="Captura de pantalla 2026-03-29 162110" src="https://github.com/user-attachments/assets/8d104993-77a0-4460-92d3-98e31c7fe78f" />

Following the methodology of the original study, we extracted the activations from the last token of each entity to evaluate the models' spatial representations. The results confirm the original thesis: both Llama-3 and Mistral develop genuine internal maps, achieving an $R^2 \approx 0.8$ coefficient in their middle layers and showing nearly identical behavior. 

This decline likely occurs because the final layers heavily prioritize next-token prediction, shifting away from maintaining a neatly organized internal spatial map. To output the correct word, the model compresses the feature space, collapsing the entire representation space into a narrow cone, a phenomenon known as anisotropy (Ethayarajh, "How Contextual are Contextualized Word Representations?", 2019). As the vectors lose their uniform spatial distribution, the embedded geographic features become linearly inseparable, causing the linear probe's performance to drop.

### 2. The Linearity Hypothesis
<img width="1153" height="693" alt="Captura de pantalla 2026-03-29 145157" src="https://github.com/user-attachments/assets/5314f4c9-fd7b-4653-8dbc-5af80ecf0962" />

<img width="1118" height="712" alt="Captura de pantalla 2026-03-29 145208" src="https://github.com/user-attachments/assets/5a30f928-ce40-410b-b7ac-ee7608153556" />

To verify how this spatial knowledge is structured, we compared the standard linear probe (Ridge) against a non-linear probe based on a Multilayer Perceptron (MLP).

In the middle layers, both probes show nearly identical results ($R^2 \approx 0.8$), confirming that geographic information is organized linearly. If a complex structure like an MLP does not significantly outperform a simple Ridge regression, it is because the intrinsic relationship in the data is already linear. 

Interestingly, in the final layers, the MLP is capable of adapting to the spatial deformation and maintaining high precision. This demonstrates that the information remains present at the end of the processing, but an anisotropic deformation makes it inaccessible through purely linear methods.

### 3. Physical Error Interpretation (Haversine Distance)
<img width="717" height="456" alt="Captura de pantalla 2026-03-29 162320" src="https://github.com/user-attachments/assets/4401e191-63e0-49cf-8e2c-75370680e89d" />

To provide a real-world dimension beyond the $R^2$ coefficient, we implemented the Haversine formula to calculate the actual distance in kilometers between the predicted and actual coordinates.

In their optimal layers, the models achieved median errors between 1,500 and 2,000 km. While this margin may seem high at a human scale, it is a remarkable result considering the immensity of the Earth's surface (510 million square kilometers), reinforcing the existence of a structured mental map rather than mere coordinate memorization.

### 4. Regional Bias and Data Volume Paradox
<img width="770" height="482" alt="Captura de pantalla 2026-03-29 153938" src="https://github.com/user-attachments/assets/48b1964e-16a5-46a6-8cd7-a17f32466ff2" />

To evaluate geographic biases, we constructed a balanced dataset comparing the USA, Europe, and China, and incorporated Qwen-2.5-7B into the analysis.

Firstly, we evaluated the models' layer-by-layer Haversine error across three distinct regions:

#### Europe
<img width="767" height="479" alt="Captura de pantalla 2026-03-29 153913" src="https://github.com/user-attachments/assets/6f413105-b575-4bd4-a8ba-183f8d9ef6f2" />

#### United States (USA)
<img width="762" height="480" alt="Captura de pantalla 2026-03-29 153922" src="https://github.com/user-attachments/assets/d255f8f3-acde-4e21-8e09-e004510993b4" />

#### China
<img width="747" height="478" alt="Captura de pantalla 2026-03-29 153932" src="https://github.com/user-attachments/assets/ff407bc6-e55f-49a7-8cf5-e94c7c4f3523" />


Then, we compared the best results of each model in each region. We observed no clear evidence of a geographic bias favoring the models' respective regions of origin. Instead, we observed a common geographic trend: all three models achieved their lowest average error in Europe. Beyond Europe, Llama-3 and Qwen performed slightly better in China than in the US, while Mistral showed better results in the US than in China. To understand why physical distance metrics are heavily dictated by the evaluated region rather than the model's origin, we identified two key factors:

* **Topographic Metric Sensitivity:** The Haversine metric is more sensitive in vast territories. In compact regions like Europe, small mistakes result in low kilometer penalties. In the vast expanses of the US or China, the same conceptual deviation translates into much higher errors. This could explain why all models show higher physical precision in Europe.
* **Demographic Granularity:**  Intuitively, given the Western origins of Llama-3 and Mistral, one might expect a clear dominance in the US over China. However, China’s results are highly competitive, even surpassing the US for Llama-3. To understand this, we must also take into account a crucial factor: city size. The median population in our China sample (68,670) is triple that of the US (19,235). Naturally, it is easier for any model to locate prominent urban centers with abundant digital references than smaller rural towns.

## Installation and Setup
To run this project locally or in Google Colab, you will need Python 3.8+ and the following dependencies. First, clone the repository:

```bash
git clone https://github.com/juliagarciauni/Spatial-Representation-in-LLMs-Replication-and-Extension.git
cd Spatial-Representation-in-LLMs-Replication-and-Extension
```
Then, install the required libraries:
```bash
pip install torch transformers pandas numpy scikit-learn scikit-learn-intelex h5py matplotlib tqdm bitsandbytes accelerate```
**Note:**  To extract the embeddings, you will need access to a GPU and enough RAM/VRAM.

## Usage
Once the dependencies are installed and the datasets are in the root folder, simply run the main script:

```bash
python main_experiment.py
```
The script will automatically process the datasets, extract the embeddings, train the linear and non-linear probes across all layers, and generate the final evaluation plots.

## References
* Wes Gurnee and Max Tegmark (2024). Language Models Represent Space and Time.
* Ethayarajh (2019). How Contextual are Contextualized Word Representations?
