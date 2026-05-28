import numpy as np

def generate_test_matrix(matrix_type='dense', size = 50, seed = 42):
    "Generating cost matrices here with specific properties that might be needed for benchmarking."
    np.random.seed(seed)

    if matrix_type == 'dense':
        return np.random.uniform(1, 10000, size=(size, size))

    elif matrix_type == 'skewed':
        base = np.random.uniform(500, 10000, size=(size, size))
        rows = np.arange(size)
        cols = np.random.permutation(size)
        base[rows, cols] = np.random.uniform(1, 100, size=size)
        return base
    
    elif matrix_type == 'sparse':
        base = np.random.uniform(1, 10000, size=(size, size))
        mask = np.random.rand(size, size) < 0.3
        base[mask] = float('inf')
        return base
    
    else:
        raise ValueError(f"Unkown matrix type: {matrix_type}")