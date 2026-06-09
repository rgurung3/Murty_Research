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

LAUZIERE_DIR = os.path.join(PROJECT_ROOT, 'implementations', 'lauziere_naive')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

sys.path.append(LAUZIERE_DIR)
sys.path.append(DATA_DIR)

from generator import generate_test_matrix
from getkBestNoRankHung import getkBestNoRankHung

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

                    if mat_type=='sparse':
                        matrix[matrix==float('inf')] = 1e6
                    
                    start_time = time.time()
                    try:
                        solutions, cost = getkBestNoRankHung(matrix, k)
                        end_time = time.time()
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
    output_path = os.path.join(RESULTS_DIR, 'lauziere_benchmark_results.csv')
    df.to_csv(output_path, index=False)
    print("=" * 65)
    print(f"Benchmark complete! Saved to: {output_path}")

if __name__=='__main__':
    run_benchmark()
                
