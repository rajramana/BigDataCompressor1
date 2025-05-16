import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib.cm import get_cmap
import data_generator as dg
import compression_algorithms as ca
import utils

def compare_compression_ratios(data_size=10240):
    """
    Compare compression ratios of different algorithms on different data types
    """
    # Generate different types of data
    text_data = dg.generate_text_data(data_size // 10)  # Each word is roughly 10 bytes
    time_series = dg.generate_time_series(data_size // 8)  # 8 bytes per float64
    categorical = dg.generate_categorical_data(data_size // 10, 20, 'skewed')
    binary_low = dg.generate_binary_data(data_size, 'low')
    binary_high = dg.generate_binary_data(data_size, 'high')
    
    # Data types and their respective data
    data_types = {
        'Text': text_data,
        'Time Series': time_series,
        'Categorical': categorical,
        'Binary (Low Entropy)': binary_low,
        'Binary (High Entropy)': binary_high
    }
    
    # Algorithms to compare
    algorithms = [
        ('Huffman', lambda x: ca.huffman_coding_demo(x)[1]),
        ('Delta', lambda x: utils.calculate_delta_compression_ratio(x)),
        ('LZW', lambda x: utils.calculate_lzw_compression_ratio(x)),
        ('RLE', lambda x: utils.calculate_rle_compression_ratio(x)),
        ('Dictionary', lambda x: utils.calculate_dict_compression_ratio(x))
    ]
    
    # Results dictionary
    results = {algo_name: [] for algo_name, _ in algorithms}
    results['Data Type'] = []
    
    # Compare algorithms on each data type
    for data_type, data in data_types.items():
        results['Data Type'].append(data_type)
        
        for algo_name, algo_func in algorithms:
            try:
                # Some algorithms may not work on all data types
                # We'll handle binary data types separately
                if data_type.startswith('Binary') and algo_name in ['Delta', 'Dictionary']:
                    ratio = 0  # Not applicable
                else:
                    ratio = algo_func(data)
                results[algo_name].append(ratio)
            except Exception as e:
                print(f"Error applying {algo_name} to {data_type}: {e}")
                results[algo_name].append(0)  # Mark as failed/not applicable
    
    # Create a DataFrame for plotting
    df = pd.DataFrame(results)
    
    # Create the visualization
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar chart
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Data Type', y='value', hue='Algorithm', 
                     data=pd.melt(df, id_vars=['Data Type'], 
                                  value_vars=[name for name, _ in algorithms],
                                  var_name='Algorithm', value_name='value'))
    
    plt.title('Compression Ratio Comparison by Data Type', fontsize=16)
    plt.xlabel('Data Type', fontsize=14)
    plt.ylabel('Compression Ratio (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)  # Compression ratio from 0-100%
    
    # Add a horizontal line at 0% compression
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    return plt.gcf()

def compare_compression_times():
    """
    Compare compression and decompression speeds of different algorithms
    """
    # Generate a standard dataset for testing
    data_size = 100000
    text_data = dg.generate_text_data(data_size // 10)
    
    # Define algorithms to test
    algorithms = [
        'Huffman',
        'Delta',
        'LZW',
        'RLE',
        'Dictionary'
    ]
    
    # Measure compression and decompression times
    comp_times = []
    decomp_times = []
    
    for algo in algorithms:
        comp_time, decomp_time = utils.measure_compression_time(algo, text_data)
        comp_times.append(comp_time)
        decomp_times.append(decomp_time)
    
    # Create the visualization
    plt.figure(figsize=(12, 6))
    
    # Plot compression times
    width = 0.35
    x = np.arange(len(algorithms))
    
    plt.bar(x - width/2, comp_times, width, label='Compression Time')
    plt.bar(x + width/2, decomp_times, width, label='Decompression Time')
    
    plt.title('Compression and Decompression Time Comparison', fontsize=16)
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.xticks(x, algorithms)
    plt.legend()
    
    # Add speed comparison annotation
    for i, (ct, dt) in enumerate(zip(comp_times, decomp_times)):
        if dt > 0:  # Avoid division by zero
            ratio = ct / dt
            plt.annotate(f"{ratio:.1f}x", 
                         xy=(i, max(ct, dt) + 0.01),
                         ha='center', va='bottom',
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    plt.tight_layout()
    
    return plt.gcf()

def space_time_tradeoff():
    """
    Visualize the tradeoff between compression ratio and speed
    """
    # Define algorithms and their characteristics
    algorithms = [
        'LZ4',
        'Snappy',
        'GZIP',
        'Deflate',
        'ZSTD',
        'Brotli',
        'LZMA'
    ]
    
    # Approximate values based on general performance characteristics
    # Compression ratio (higher is better)
    compression_ratios = [50, 55, 65, 65, 70, 75, 80]
    
    # Compression speed (higher is better)
    compression_speeds = [900, 700, 150, 160, 400, 100, 30]
    
    # Decompression speed (higher is better)
    decompression_speeds = [2000, 1800, 400, 430, 1000, 350, 100]
    
    # Create the visualization
    plt.figure(figsize=(12, 10))
    
    # Create a colormap for the points
    colors = get_cmap('viridis')(np.linspace(0, 1, len(algorithms)))
    
    # Plot speed vs. ratio as a scatter plot
    plt.scatter(compression_ratios, compression_speeds, s=200, c=colors, alpha=0.7, label='Compression Speed')
    plt.scatter(compression_ratios, decompression_speeds, s=200, c=colors, marker='^', alpha=0.7, label='Decompression Speed')
    
    # Connect related points with lines
    for i in range(len(algorithms)):
        plt.plot([compression_ratios[i], compression_ratios[i]], 
                 [compression_speeds[i], decompression_speeds[i]], 
                 'k--', alpha=0.3)
    
    # Add algorithm labels
    for i, algo in enumerate(algorithms):
        plt.annotate(algo, 
                     xy=(compression_ratios[i], compression_speeds[i]),
                     xytext=(0, -15),
                     textcoords='offset points',
                     ha='center', va='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Add quadrant labels
    plt.text(52, 1800, "Fast but Low Compression", 
             ha='left', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.3))
    
    plt.text(78, 1800, "Ideal\n(Fast and High Compression)", 
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", alpha=0.3))
    
    plt.text(52, 100, "Worst Case\n(Slow and Low Compression)", 
             ha='left', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", fc="lightcoral", alpha=0.3))
    
    plt.text(78, 100, "High Compression but Slow", 
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.3))
    
    # Add dividing lines for quadrants
    plt.axvline(x=65, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=400, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Compression Space-Time Tradeoff', fontsize=16)
    plt.xlabel('Compression Ratio (%)', fontsize=14)
    plt.ylabel('Speed (MB/s)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Use logarithmic scale for speed to better visualize differences
    plt.yscale('log')
    
    # Set axis limits
    plt.xlim(45, 85)
    plt.ylim(20, 3000)
    
    plt.tight_layout()
    
    return plt.gcf()

def data_type_performance(data_type):
    """
    Analyze how different algorithms perform on a specific data type
    
    Parameters:
    -----------
    data_type : str
        The type of data to analyze ('Text', 'Numerical', 'Mixed', 'Binary')
    """
    # Generate appropriate data based on the data type
    if data_type == "Text":
        data = dg.generate_text_data(5000)
        subtitle = "Analysis of Text Data Compression"
    elif data_type == "Numerical":
        data = dg.generate_time_series(5000)
        subtitle = "Analysis of Numerical Data Compression"
    elif data_type == "Mixed":
        mixed_data = dg.generate_mixed_dataset(1000)
        # Convert to string representation for testing
        data = str(mixed_data)
        subtitle = "Analysis of Mixed Data Compression"
    else:  # Binary
        data = dg.generate_binary_data(10000, 'medium')
        subtitle = "Analysis of Binary Data Compression"
    
    # Define algorithms to test with appropriate parameters
    algorithms = [
        ('Huffman', {'name': 'Huffman', 'color': '#1f77b4', 'params': {}}),
        ('LZW', {'name': 'LZW', 'color': '#ff7f0e', 'params': {}}),
        ('Delta', {'name': 'Delta', 'color': '#2ca02c', 'params': {}}),
        ('RLE', {'name': 'RLE', 'color': '#d62728', 'params': {}}),
        ('Dictionary', {'name': 'Dictionary', 'color': '#9467bd', 'params': {}}),
        ('FOR', {'name': 'FOR', 'color': '#8c564b', 'params': {}})
    ]
    
    # Collect metrics for each algorithm
    metrics = {
        'Algorithm': [],
        'Compression Ratio (%)': [],
        'Compression Time (ms)': [],
        'Decompression Time (ms)': [],
        'Memory Usage (MB)': []
    }
    
    for algo_name, algo_info in algorithms:
        try:
            # Measure performance
            ratio, comp_time, decomp_time, memory = utils.measure_algorithm_performance(
                algo_name, data, **algo_info['params'])
            
            # Store metrics
            metrics['Algorithm'].append(algo_info['name'])
            metrics['Compression Ratio (%)'].append(ratio)
            metrics['Compression Time (ms)'].append(comp_time * 1000)  # Convert to ms
            metrics['Decompression Time (ms)'].append(decomp_time * 1000)  # Convert to ms
            metrics['Memory Usage (MB)'].append(memory)
        except Exception as e:
            print(f"Error measuring {algo_name}: {e}")
            # Skip this algorithm
    
    # Create DataFrame for plotting
    df = pd.DataFrame(metrics)
    
    # Create the visualization with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Compression Performance for {data_type} Data", fontsize=18)
    
    # Plot compression ratio
    axes[0, 0].bar(df['Algorithm'], df['Compression Ratio (%)'], color=[algo_info['color'] for _, algo_info in algorithms if algo_info['name'] in df['Algorithm'].values])
    axes[0, 0].set_title('Compression Ratio', fontsize=14)
    axes[0, 0].set_ylabel('Compression Ratio (%)', fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim(0, 100)
    
    # Plot compression time
    axes[0, 1].bar(df['Algorithm'], df['Compression Time (ms)'], color=[algo_info['color'] for _, algo_info in algorithms if algo_info['name'] in df['Algorithm'].values])
    axes[0, 1].set_title('Compression Time', fontsize=14)
    axes[0, 1].set_ylabel('Time (ms)', fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot decompression time
    axes[1, 0].bar(df['Algorithm'], df['Decompression Time (ms)'], color=[algo_info['color'] for _, algo_info in algorithms if algo_info['name'] in df['Algorithm'].values])
    axes[1, 0].set_title('Decompression Time', fontsize=14)
    axes[1, 0].set_ylabel('Time (ms)', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot memory usage
    axes[1, 1].bar(df['Algorithm'], df['Memory Usage (MB)'], color=[algo_info['color'] for _, algo_info in algorithms if algo_info['name'] in df['Algorithm'].values])
    axes[1, 1].set_title('Memory Usage', fontsize=14)
    axes[1, 1].set_ylabel('Memory (MB)', fontsize=12)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    for ax in axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add a subtitle
    plt.figtext(0.5, 0.92, subtitle, fontsize=16, ha='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    return fig
