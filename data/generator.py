import numpy as np

def generate_test_matrix(matrix_type='dense', size = 50, seed = 42):
    "Generating cost matrices here with specific properties that might be needed for benchmarking."
    np.random.seed(seed)
    