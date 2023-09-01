import pandas as pd
import numpy as np

# Generate synthetic data
num_samples = 1000

data = {
    'Element1': np.random.choice(['Fe', 'Ni', 'Co', 'Cu'], size=num_samples),
    'Element2': np.random.choice(['Mn', 'O', 'Fe', 'Co', 'Ag'], size=num_samples),
    'Element3': np.random.choice(['', 'Fe', 'Ag', 'Mn'], size=num_samples),
    'Numb1': np.random.randint(1, 6, size=num_samples),
    'Numb2': np.random.randint(1, 6, size=num_samples),
    'Numb3': np.random.randint(1, 6, size=num_samples),
    'Lattice_a': np.random.uniform(2.0, 4.0, size=num_samples),
    'Lattice_b': np.random.uniform(2.0, 4.0, size=num_samples),
    'Lattice_c': np.random.uniform(2.0, 4.0, size=num_samples),
    'Magnetic_Moment': np.random.uniform(0.0, 3.0, size=num_samples),
    'Band_Gap': np.random.uniform(0.5, 2.5, size=num_samples),
    'Heat_Capacity': np.random.uniform(15.0, 35.0, size=num_samples),
    'Curie_Temperature': np.random.randint(500, 1500, size=num_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Generate compound names
df['Compound_Name'] = df['Element1'] + df['Element2'] + df['Element3']

# Save the DataFrame to a CSV file
df.to_csv('synthetic_dataset_with_names.csv', index=False)
