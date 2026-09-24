import numpy as np
import pandas as pd 
def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    return pd.DataFrame(data.tolist())

def npy_preprocessor(filename):
    df = read_data(filename)
    return df


def scale_x_coordinates(X_split, scaler):
    if len(X_split) == 0:
        return []
        
    X_stacked = np.stack(X_split) 
    
    coords = X_stacked[:, :, :3]
    features = X_stacked[:, :, 3:]
    
    coords_scaled_flat = scaler.transform(coords.reshape(-1, 3))
    coords_scaled = coords_scaled_flat.reshape(X_stacked.shape[0], X_stacked.shape[1], 3)
    
    X_scaled_stacked = np.concatenate((coords_scaled, features), axis=2)
    
    return [X_scaled_stacked[i, ...] for i in range(X_scaled_stacked.shape[0])]


