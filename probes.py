
from sklearnex import patch_sklearn #To speed up sklearn with Intel's optimizations
patch_sklearn()
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#We implement the Haversine formula to calculate the distance in kilometers between the predicted coordinates 
#(y_pred) and the true coordinates (y_true).
def haversine_distance(y_true, y_pred):
    r = 6371.0 # Earth radius in kilometers

    # y_true and y_pred contain [Latitude, Longitude] of each city
    # We need to convert them to radians for the formula
    lat1, lon1 = np.radians(y_true[:, 0]), np.radians(y_true[:, 1])
    lat2, lon2 = np.radians(y_pred[:, 0]), np.radians(y_pred[:, 1])

    dlat = lat2 - lat1 # We get the difference of latitudes in radians
    dlon = lon2 - lon1 # We get the difference of longitudes in radians

    # We apply the mathematical formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * r * np.arcsin(np.sqrt(a))

    return c # We return a list with the kilometers of error of each city


# In order to probe the linearity the paper defends, we will use a Ridge regression as linear probe and an MLPRegressor as non-linear probe.
# We will use the dataset from the paper, because it belongs to the part of the replication of the paper. 

# On the other hand, my contribution to the paper, as mentioned in the report, are the following:

# 1. I have tried another two models to compare with the paper's results: Qwen and Mistral. Therefore, we will have the results of three different 
# models (Llama3, Mistral and Qwen) to compare with the paper's results. We will use the dataset from the paper and probe the linearity. 

# 2. I have added the option to calculate the Haversine distance in kilometers between the predicted coordinates 
# and the true coordinates, to give it a physical meaning. This is controlled by the boolean parameter "haversine". 
# We will use the dataset from the paper for this experiment and we will use the three models (Llama3, Mistral and Qwen) to compare the results.

# 3. Lastly, we will check if there is any spatial representation bias depending on the geographical origin of the models. 
# To do this, we will use a different, regionally-balanced dataset (grouping cities into USA, Europe, and China) to evaluate 
# if a model performs significantly better in its region of origin compared to the others. 


def train(ruta_h5, dataset, dataset_regional, haversine, no_lineal):
    
    #We load the embeddings from the .h5 file
    with h5py.File(ruta_h5, 'r') as archivo_h5:
        dset = archivo_h5['embeddings'][:].astype(np.float32)

    num_layers = dset.shape[1]
    
    indices_split = {}
    
    #In case of using the regional dataset, we will split the data by region.
    if dataset_regional:
        names_reg = ['USA', 'Europa', 'China']
        columns = ['is_USA', 'is_Europa', 'is_China']
       
        for col, reg_name in zip(columns, names_reg):
            # We create a boolean mask which has a True or a False depending on if the city belongs to the region or not
            mask = dataset[col].values
            # We get the indices of the cities that belong to the region (the ones where the mask is True)
            indices_reg = np.where(mask)[0] 
            # We split the indices of the cities that belong to the region into a train set and a test set, with a fixed random seed for reproducibility
            idx_tr, idx_te = train_test_split(indices_reg, test_size=0.2, random_state=31) 
            indices_split[reg_name] = (idx_tr, idx_te) #We save the train and test indices for each region in a dictionary
    else:
        # If we are using the dataset from the paper, we will use all the cities given in the dataset.
        # We are using names_reg = ['Dataset_Paper'] just to be able to use the same function for both experiments. Because the logic of 
        # the training is almost the same. 
        names_reg = ['Dataset_Paper']
        indices_reg = np.arange(len(dataset)) 
        idx_tr, idx_te = train_test_split(indices_reg, test_size=0.2, random_state=31)
        indices_split['Dataset_Paper'] = (idx_tr, idx_te)


    #In order to be consistent, we will create the same structure of results for both experiments, even if we are not going to use all
    #the options in each experiment. 
    results = {reg: {'lin': [], 'non': [], 'km': []} for reg in names_reg}

    #We process all the layers
    for layer_idx in tqdm(range(num_layers)):
        #In the case of the regional dataset, we will train a model for each region. That is why our reg_name has the name of each region. In the case 
        #of the dataset from the paper, we will train only one model with all the cities. That is why our reg_name only has one value: 'Dataset_Paper'.
        for reg_name in names_reg:
            idx_tr, idx_te = indices_split[reg_name] 
            
            #Training and test sets of embeddings for this layer and this region
            X_tr = dset[idx_tr, layer_idx, :]
            X_te = dset[idx_te, layer_idx, :]
            
            #We get the latitude and longitude of the cities in the train and test sets
            y_tr = dataset.iloc[idx_tr][["Lat", "Lon"]].values
            y_te = dataset.iloc[idx_te][["Lat", "Lon"]].values

            # LINEAR PROBE (Ridge)
            model_lin = Ridge(alpha=1.0)
            model_lin.fit(X_tr, y_tr) #training
            predictions_linear = model_lin.predict(X_te) #test
            results[reg_name]['lin'].append(r2_score(y_te, predictions_linear))
            
            # HAVERSINE DISTANCE IN KILOMETERS
            if haversine:
                mistake_distance = haversine_distance(y_te, predictions_linear)
                mistake_layer = np.median(mistake_distance)
                results[reg_name]['km'].append(mistake_layer)

            # NO LINEAR PROBE (MLP)
            if no_lineal:
                model_non = MLPRegressor(
                    hidden_layer_sizes=(256,), 
                    activation='relu', 
                    max_iter=200, #So that our computer is able to train the model in a reasonable time.
                    early_stopping=True, 
                    random_state=42
                )
                model_non.fit(X_tr, y_tr)
                results[reg_name]['non'].append(r2_score(y_te, model_non.predict(X_te)))

    return results
