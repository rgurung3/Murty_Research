import time
import os
import sys
import numpy as np
import pandas as pd
import traceback
import math


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

np.math = math

MURTY_DIR = os.path.join(PROJECT_ROOT, 'implementations', 'motrom_fast')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

sys.path.append(MURTY_DIR)
sys.path.append(DATA_DIR)

from generator import generate_test_matrix
import mhtdaClink
from mhtdaClink import mhtda, allocateWorkvarsforDA, sparsifyByRow


def run_benchmark():
    sizes = [10, 20, 30]
    k_values = [100, 500]
    matrix_types = ['dense', 'skewed', 'sparse']
    nums_runs = 5

    results = []
    print("=" * 65)
    print(f"{'Matrix Type':<15} | {'Size':<6} | {'K':<5} | {'Avg Time (s)':<12} | {'Status':<8}")
    print("=" * 65)

    for mat_Type in matrix_types:
        for size in sizes:
            for k in k_values:
                execution_times = []
                failed = False

                for seed in range(nums_runs):
                    matrix = generate_test_matrix(matrix_type = mat_Type, size=size, seed=seed)

                    if mat_Type == 'sparse':
                        matrix[matrix==float('inf')] = 1e6

                    cost_matrix = np.ascontiguousarray(matrix.astype(np.float64))
                    nrows, ncolumns = cost_matrix.shape

                    if mhtdaClink.sparse:
                        cost_matrix_to_use = sparsifyByRow(cost_matrix, ncolumns)
                    else:
                        cost_matrix_to_use = cost_matrix
                    row_priors = np.ones((1, nrows), dtype=np.bool)
                    col_priors = np.ones((1, ncolumns), dtype=np.bool)
                    row_prior_weights = np.zeros(1)
                    col_prior_weights = np.zeros(1)
                    out_costs = np.zeros(k)
                    out_associations = np.zeros((k, nrows + ncolumns, 2), dtype=np.int32)
                    workvars = allocateWorkvarsforDA(nrows, ncolumns, k)
                    start_time = time.time()
                    
                    try:
                        workvars = allocateWorkvarsforDA(nrows, ncolumns, k)
                        start_time = time.time()
                        mhtda(
                            cost_matrix_to_use, row_priors, row_prior_weights, col_priors, col_prior_weights,
                            out_associations, out_costs, workvars
                        )
                        end_time = time.time()
                        execution_times.append(start_time - end_time)
                    except Exception as e:
                        print("Crashed here.")
                        traceback.print_exc()
                        failed = True
                        break

                    if failed or not execution_times:
                        avg_time = "Crashed"
                        status = "Failed"
                    else:
                        avg_time = f"{np.mean(execution_times): .4f}"
                        status = "Success"
                    print(f"{mat_Type: <15} | {size:<6} | {k:<5} | {avg_time:<12} | {status:<8}")
                    results.append({
                        'Matrix Type': mat_Type,
                        'Size': size,
                        'K': k,
                        'Avg Time': avg_time,
                        'Status': status
                    })
    df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, 'fast_murty_benchmark_results.csv')
    df.to_csv(output_path, index=False)
    print("=" * 65)
    print(f"Benchmark complete! Saved to: {output_path}")

if __name__ == '__main__':
    run_benchmark()
