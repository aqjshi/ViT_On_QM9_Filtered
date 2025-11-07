import numpy as np
from tqdm import tqdm
def rotate_molecule(xyz_data, angle, axis='z'):
    """Rotate the first 3 columns (x, y, z) around the chosen axis by 'angle'."""
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    else:  # 'z'
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])

    coords_3d = xyz_data[:, :3]    # (27,3)
    other_feats = xyz_data[:, 3:]  # (27,5) or however many remain
    rotated_coords = np.dot(coords_3d, rotation_matrix.T)  # (27,3)

    rotated_xyz_data = np.hstack((rotated_coords, other_feats))  # Still (27,8)
    return rotated_xyz_data

def translate_molecule(xyz_data, magnitude=0.02):
    """Translate the first 3 columns (x, y, z) by a random magnitude along a random axis."""
    translation_vector = np.zeros(3)
    random_axis = np.random.choice([0, 1, 2])  # Choose x (0), y (1), or z (2)
    translation_vector[random_axis] = magnitude

    coords_3d = xyz_data[:, :3]    # (27,3)
    other_feats = xyz_data[:, 3:]  # (27,5) or however many remain
    translated_coords = coords_3d + translation_vector  # Apply translation

    translated_xyz_data = np.hstack((translated_coords, other_feats))  # Still (27,8)
    return translated_xyz_data

def reflect_molecule(xyz_data):
    """Reflect the first 3 columns (x, y, z) across a randomly chosen axis."""
    reflection_matrix = np.eye(3)
    random_axis = np.random.choice([0, 1, 2])  # Choose x (0), y (1), or z (2)
    reflection_matrix[random_axis, random_axis] = -1  # Reflect across the chosen axis

    coords_3d = xyz_data[:, :3]    # (27,3)
    other_feats = xyz_data[:, 3:]  # (27,5) or however many remain
    reflected_coords = np.dot(coords_3d, reflection_matrix.T)  # Apply reflection

    reflected_xyz_data = np.hstack((reflected_coords, other_feats))  # Still (27,8)
    return reflected_xyz_data

def augment_data(X_train, y_train, num_samples):
    X_train_stacked = np.stack(X_train)
    
    rotated_X, rotated_y = [], []
    original_num_samples = len(X_train_stacked)
    
    for _ in tqdm(range(num_samples), desc="Augmenting data"):
        idx = np.random.randint(0, original_num_samples)
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.choice(['x', 'y', 'z'])
        
        # Use the stacked array for augmentation
        aug_molecule = rotate_molecule(X_train_stacked[idx], angle, axis=axis)
        aug2_molecule = translate_molecule(X_train_stacked[idx])
        rotated_X.append(aug_molecule)
        rotated_y.append(y_train[idx])
        rotated_X.append(aug2_molecule)
        rotated_y.append(y_train[idx])
        
    # Concatenate the original stacked data with the new augmented data
    X_augmented = np.concatenate((X_train_stacked, np.array(rotated_X)), axis=0)
    y_augmented = np.concatenate((y_train, np.array(rotated_y)), axis=0)
    
    print("Augmentation complete.")
    return X_augmented, y_augmented
