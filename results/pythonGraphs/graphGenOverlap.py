import overlap_data
import os
import pandas as pd
import matplotlib.pyplot as plt


output_dir = 'graphImgs'
os.makedirs(output_dir, exist_ok=True)

# Part 1 performance metrics ---------------------------------------

df = pd.DataFrame(overlap_data.single_core_data_overlap_simple_line, columns=['Algorithm', 'Language', 'Size', 'Time', 'L1_Misses', 'L2_Misses'])

# Normalized misses = L1/L2 cache misses / (num of op)³

def size_norm (matrix_size):
    if '-' in (matrix_size):
        return int (matrix_size.split('-')[0])
    return int(matrix_size)

df['N'] = df['Size'].apply(size_norm)
df['N3'] = df['N'] ** 3
df['Norm_L1'] = df['L1_Misses'] / df['N3']
df['Norm_L2'] = df['L2_Misses'] / df['N3']

def graph_gen (df, metric, ylabel, title, filename):
    filepath = os.path.join(output_dir, filename)

    if os.path.exists(filepath):
        return

    plt.figure(figsize=(10, 6))
    subset_basic = df[df['Algorithm'] == 'Basic']
    subset_line = df[df['Algorithm'] == 'Line']
    languages = subset_basic['Language'].unique()
    
    if metric == 'Misses':
        for lang in languages:
            lang_data = subset_basic[subset_basic['Language'] == lang]
            plt.plot(lang_data['Size'], lang_data['Norm_L1'], marker='o', linestyle=':', label=f'{lang} Basic (L1)')
            plt.plot(lang_data['Size'], lang_data['Norm_L2'], marker='s', linestyle=':', label=f'{lang} Basic (L2)')

            lang_data = subset_line[subset_line['Language'] == lang]
            plt.plot(lang_data['Size'], lang_data['Norm_L1'], marker='o', linestyle=':', label=f'{lang} Line (L1)')
            plt.plot(lang_data['Size'], lang_data['Norm_L2'], marker='s', linestyle=':', label=f'{lang} Line (L2)')
    else:
        for lang in languages:
            lang_data = subset_basic[subset_basic['Language'] == lang]
            plt.plot(lang_data['Size'], lang_data[metric], marker='o', linestyle=':', label=f'{lang} Basic')

            lang_data = subset_line[subset_line['Language'] == lang]
            plt.plot(lang_data['Size'], lang_data[metric], marker='o', linestyle=':', label=f'{lang} Line')
            
    plt.xlabel('Matrix Size')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, filename) 
    plt.savefig(filename)
    plt.close()


graph_gen(df, 'Time', 'Time (seconds)', 'Basic Algorithm: Runtime vs Matrix Size', 'time_overlap.png')

graph_gen(df, 'Misses', 'Cache Misses Count', 'Basic Algorithm: L1 & L2 Cache Misses vs Matrix Size', 'misses_overlap.png')
