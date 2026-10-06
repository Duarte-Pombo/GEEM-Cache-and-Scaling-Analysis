import resultsData
import os
import pandas as pd
import matplotlib.pyplot as plt


output_dir = 'graphImgs'
os.makedirs(output_dir, exist_ok=True)

# Part 1 performance metrics ---------------------------------------

df = pd.DataFrame(resultsData.single_core_data, columns=['Algorithm', 'Language', 'Size', 'Time', 'L1_Misses', 'L2_Misses'])

# Normalized misses = L1/L2 cache misses / (num of op)³

def size_norm (matrix_size):
    if '-' in (matrix_size):
        return int (matrix_size.split('-')[0])
    return int(matrix_size)

df['N'] = df['Size'].apply(size_norm)
df['N3'] = df['N'] ** 3
df['Norm_L1'] = df['L1_Misses'] / df['N3']
df['Norm_L2'] = df['L2_Misses'] / df['N3']

def graph_gen (df, algo, metric, ylabel, title, filename):
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        return

    plt.figure(figsize=(10, 6))
    subset = df[df['Algorithm'] == algo]
    languages = subset['Language'].unique()
    
    if metric == 'Misses':
        for lang in languages:
            lang_data = subset[subset['Language'] == lang]
            plt.plot(lang_data['Size'], lang_data['Norm_L1'], marker='o', linestyle=':', label=f'{lang} (L1)')
            plt.plot(lang_data['Size'], lang_data['Norm_L2'], marker='s', linestyle=':', label=f'{lang} (L2)')
    else:
        for lang in languages:
            lang_data = subset[subset['Language'] == lang]
            plt.plot(lang_data['Size'], lang_data[metric], marker='o', linestyle=':', label=lang)
            
    plt.xlabel('Matrix Size' if algo != 'Block' else 'Matrix Size - Block Size')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, filename) 
    plt.savefig(filename)
    plt.close()


graph_gen(df, 'Basic', 'Time', 'Time (seconds)', 'Basic Algorithm: Runtime vs Matrix Size', 'time_basic.png')
graph_gen(df, 'Line', 'Time', 'Time (seconds)', 'Line-by-Line Algorithm: Runtime vs Matrix Size', 'time_line.png')
graph_gen(df, 'Block', 'Time', 'Time (seconds)', 'Block Algorithm: Runtime vs (Matrix Size - Block Size)', 'time_block.png')

graph_gen(df, 'Basic', 'Misses', 'Cache Misses Count', 'Basic Algorithm: L1 & L2 Cache Misses vs Matrix Size', 'misses_basic.png')
graph_gen(df, 'Line', 'Misses', 'Cache Misses Count', 'Line-by-Line Algorithm: L1 & L2 Cache Misses vs Matrix Size', 'misses_line.png')
graph_gen(df, 'Block', 'Misses', 'Cache Misses Count', 'Block Algorithm: L1 & L2 Cache Misses vs (Matrix Size - Block Size)', 'misses_block.png')

# Part 2 performance metrics ---------------------------------------

df_multi = pd.DataFrame(resultsData.multi_core_data, columns=['Algorithm', 'Threads', 'Size', 'Time'])

df_multi['Threads'] = pd.to_numeric(df_multi['Threads'])
df_multi['Size'] = pd.to_numeric(df_multi['Size'])
df_multi['Time'] = pd.to_numeric(df_multi['Time'])

df_multi['GFlops'] = (2 * (df_multi['Size'] ** 3)) / df_multi['Time'] / (10**9)

df_step1 = df_multi[df_multi['Size'] <= 3072].copy()
df_step2 = df_multi[df_multi['Size'] == 8192].copy()

baseline_time = df_step2[(df_step2['Algorithm'] == 'Parallel Line') & (df_step2['Threads'] == 4)]['Time'].values[0]
df_step2['Speedup'] = baseline_time / df_step2['Time']

def graph_multi_gen(df, x_col, y_col, xlabel, ylabel, title, filename):
    filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        return

    plt.figure(figsize=(10, 6))
    algorithms = df['Algorithm'].unique()
    
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo].sort_values(by=x_col)
        plt.plot(algo_data[x_col], algo_data[y_col], marker='o', linestyle='-', label=algo)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(filepath)
    plt.close()
    

graph_multi_gen(df_step1, 'Size', 'GFlops', 'Matrix Size', 'GFlop/s', 'GFlop/s vs Matrix Size (4 Threads)', 'gflops_step1.png')
graph_multi_gen(df_step2, 'Threads', 'GFlops', 'Number of Threads', 'GFlop/s', 'GFlop/s vs Threads (8192x8192)', 'gflops_step2.png')
graph_multi_gen(df_step2, 'Threads', 'Speedup', 'Number of Threads', 'Speedup (Baseline: 4 Threads)', 'Speedup vs Threads (8192x8192)', 'speedup_step2.png')
graph_multi_gen(df_step1, 'Size', 'Time', 'Matrix Size', 'Execution Time (seconds)', 'Execution Time vs Matrix Size (4 Threads)', 'time_step1.png')
graph_multi_gen(df_step2, 'Threads', 'Time', 'Number of Threads', 'Execution Time (seconds)', 'Execution Time vs Threads (8192x8192)', 'time_step2.png')
