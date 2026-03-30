import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import h5py
from tqdm import tqdm
import os
import gc

# The idea is that we are going to obtain the list of numbers that represents each city
# as understood by the model in each of the layers
def get_embeddings(city_name, model, tokenizer):
    info = f"{city_name}"
    # We obtain the token corresponding to the phrase
    token = tokenizer.encode(info)
    # It is necessary to add a new dimension, since the model works with batches of phrases, not a single phrase.
    tensor = torch.tensor(token).unsqueeze(0).to(model.device)


    # We use torch.no_grad() to avoid saving the intermediate calculations that are not necessary for our purpose.
    with torch.no_grad():
        output = model(tensor, output_hidden_states=True)
        citiesLayers = []
        for layer in output.hidden_states: # We iterate through the layers
            # We extract the obtained data. We take the last token (-1). This is because the model processes the
            # input token by token, and the last token will have the most complete information about the city.
            vector = layer[0, -1, :].float().cpu().numpy()
            citiesLayers.append(vector)
    # We return an array with the versions we have been getting in each layer
    return np.array(citiesLayers, dtype=np.float16)

def extract_embeddings(name_model, dataset, suffix, config=None):
    name_file = f"matriz_{name_model.replace('/', '_')}{suffix}.h5"

    # If it already exists, we exit the function doing nothing, since we already have the embeddings extracted and
    # saved in the .h5 file.
    if os.path.exists(name_file):
        return

    # We load the tokenizer and the model. The tokenizer is responsible for converting the city name into a format
    # that the model can understand (tokens).
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    tokenizer.pad_token = tokenizer.eos_token # We set the padding token to be the same as the end of sequence token,
    #since some models do not have a specific padding token.

    # Depending on the model, we load it with 4-bit or 8-bit quantization to save memory and speed up inference. As mentionced
    # in the document, Qwen gave us bad results using 4-bit quantization, so we started using 8-bit for that model.
    if config is not None:

        model = AutoModelForCausalLM.from_pretrained(
            name_model, device_map={"": 0},
            quantization_config= BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"),
            torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name_model, device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    model.eval()

    #We obtain the embedding of the first city to determine the number of layers and dimensions of the embeddings.
    emb = get_embeddings(dataset["City"].iloc[0], model, tokenizer)
    num_layers, num_dim = emb.shape

    #We create the .h5 file and prepare it with the appropriate shape to store the embeddings of all the cities.
    archivo_h5 = h5py.File(name_file, 'w')
    dset = archivo_h5.create_dataset('embeddings', shape=(len(dataset), num_layers, num_dim), dtype='float16')

    #We iterate through the cities in the dataset, extract their embeddings and save them in the .h5 file.
    #We use tqdm to show a progress bar.
    for i, city in enumerate(tqdm(dataset["City"])):
        dset[i, :, :] = get_embeddings(city, model, tokenizer)

    archivo_h5.close()

    #We free the memory used by the model and the tokenizer, and we clear the GPU cache to avoid memory issues in subsequent runs.
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
