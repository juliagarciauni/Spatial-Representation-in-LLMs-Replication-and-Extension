import pandas as pd
import numpy as np

def load_worldcities_dataset():
    # We read the CSV 
    df_completo = pd.read_csv("worldcities.csv").dropna(subset=['lat', 'lng', 'city_ascii'])

    # European countries list
    countries_europa = [
        "Spain", "France", "Germany", "Italy", "United Kingdom",
        "Portugal", "Netherlands", "Belgium", "Switzerland",
        "Austria", "Sweden", "Norway", "Denmark", "Finland", "Poland"
    ]

    # We classify cities into USA, China, Europa, or Otro.
    def label_region(pais):
        if pais == 'United States': return 'USA'
        elif pais == 'China': return 'China'
        elif pais in countries_europa: return 'Europa'
        else: return 'Otro'

    # Assign region labels to the complete dataset (we add a new column 'Region')
    df_completo['Region'] = df_completo['country'].apply(label_region)

    # We look for how many cities the smallest region has, to be fair 
    max_ciudades_justas = min(
        len(df_completo[df_completo['Region'] == 'USA']),
        len(df_completo[df_completo['Region'] == 'Europa']),
        len(df_completo[df_completo['Region'] == 'China'])
    )
    
    # We extract the indices for the random balanced dataset (USA, Europa, China) with a fixed seed for reproducibility
    idx_usa = df_completo[df_completo['Region'] == 'USA'].sample(n=max_ciudades_justas, random_state=42).index
    idx_eur = df_completo[df_completo['Region'] == 'Europa'].sample(n=max_ciudades_justas, random_state=42).index
    idx_chi = df_completo[df_completo['Region'] == 'China'].sample(n=max_ciudades_justas, random_state=42).index

    # We join the indices of the three regions into a single list (Since regions are mutually exclusive, there are no overlapping cities)
    all_index = idx_usa.union(idx_eur).union(idx_chi)

    # We filter the dataset with those indices
    dataSet = df_completo.loc[all_index].copy()

    # We create new boolean columns asking if the city index was in the original groups of USA, Europa, China. 
    # If it was in that group, it will be True, otherwise False. This way we can keep track of the original region of each city in the final dataset.
    dataSet['is_USA'] = dataSet.index.isin(idx_usa)
    dataSet['is_Europa'] = dataSet.index.isin(idx_eur)
    dataSet['is_China'] = dataSet.index.isin(idx_chi)

    # We take only the columns that we need and we rename them to match the expected format (City, Lat, Lon, Region, country, is_USA, is_Europa, is_China)
    dataSet = dataSet[['city_ascii', 'lat', 'lng', 'Region', 'country', 'is_USA', 'is_Europa', 'is_China']]
    dataSet = dataSet.rename(columns={'city_ascii': 'City', 'lat': 'Lat', 'lng': 'Lon'}).reset_index(drop=True)
    
    return dataSet 

def load_paper_dataset():
    # Download the official paper dataset
    data = pd.read_csv("https://raw.githubusercontent.com/wesg52/world-models/refs/heads/main/data/entity_datasets/world_place.csv")
    
    # Rename original CSV columns to match our code
    dataSet = data.rename(columns={"name": "City", "latitude": "Lat", "longitude": "Lon"})
    
    # We reduce to 5000 cities due to GPU memory constraints, using a fixed seed for reproducibility.
    dataSet = dataSet.sample(n=5000, random_state=42).reset_index(drop=True)
    
    return dataSet 
