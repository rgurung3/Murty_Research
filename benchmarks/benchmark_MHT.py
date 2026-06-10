import time
import os
import sys
import numpy as np
import pandas as pd
import traceback
import math
from mht import hypgen

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

np.math = math

MHT_CORE_DIR = os.path.join(PROJECT_ROOT, 'implementations', 'mht', 'mht')
MHT_MURTY_DIR = os.path.join(PROJECT_ROOT, 'implementations', 'mht', 'murty')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

sys.path.append(MHT_CORE_DIR)
sys.path.append(MHT_MURTY_DIR)
sys.path.append(DATA_DIR)

from generator import generate_test_matrix
# Importing the internal murty module from the cloned repo framework
import murty as mht_murty

def run_benchmark():
    sizes = [50, 75, 100]
    k_values = [1000, 5000]
    matrix_types = ['dense', 'skewed', 'sparse']
    num_runs = 5

    results = []
    print("=" * 65)
    print(f"{'Matrix Type':<15} | {'Size':<6} | {'K':<5} | {'Avg Time (s)':<12} | {'Status':<8}")
    print("=" * 65)

    for mat_type in matrix_types:
        for size in sizes:
            for k in k_values:
                execution_times = []
                failed = False

                for seed in range(num_runs):
                    matrix = generate_test_matrix(matrix_type=mat_type, size=size, seed=seed)

                    # Handle values for sparse matrices consistently
                    matrix[matrix == float('inf')] = 100000
                    matrix[matrix == np.inf] = 100000
                    cost_matrix = np.matrix(matrix, dtype=np.int32)

                    try:
                        start_time = time.perf_counter()
                        
                        solutions = []
                        
                        for res in hypgen.murty(cost_matrix):
                            solutions.append(res)
                            if len(solutions) >= k:
                                break
                        end_time = time.perf_counter()
                        execution_times.append(end_time - start_time)
                    except Exception as e:
                        print("Crash detected..")
                        traceback.print_exc()
                        failed = True
                        break
                
                if failed or not execution_times:
                    avg_time = "CRASHED"
                    status = "FAILED"
                else:
                    avg_time = f"{np.mean(execution_times):.4f}"
                    status = "Success"
                
                print(f"{mat_type:<15} | {size:<6} | {k:<5} | {avg_time:<12} | {status:<8}")

                results.append({
                    'Matrix Type': mat_type,
                    'Size': size,
                    'K': k,
                    'Avg Time': avg_time,
                    'Status': status
                })

    df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, 'mht_murty_benchmark_results.csv')
    df.to_csv(output_path, index=False)
    print("=" * 65)
    print(f"Benchmark complete! Saved to: {output_path}")

if __name__ == '__main__':
    run_benchmark()
