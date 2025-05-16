import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import streamlit as st
import compression_algorithms as ca
import data_generator as dg
import utils

def benchmark_algorithms(data, algorithms=None):
    """
    Benchmark multiple compression algorithms on the given data
    
    Parameters:
    -----------
    data : str, list, np.ndarray, or bytes
        The data to compress
    algorithms : list
        List of algorithm names to benchmark. If None, uses all available algorithms.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with benchmark results
    """
    if algorithms is None:
        algorithms = ['Huffman', 'Delta', 'LZW', 'RLE', 'Dictionary', 'FOR']
    
    results = []
    
    for algo_name in algorithms:
        try:
            # Get original size
            original_size = utils.get_data_size(data)
            
            # Measure compression time
            start_time = time.time()
            compressed_data = utils.compress_with_algorithm(algo_name, data)
            compression_time = time.time() - start_time
            
            # Get compressed size
            compressed_size = utils.get_data_size(compressed_data)
            
            # Measure decompression time
            start_time = time.time()
            decompressed_data = utils.decompress_with_algorithm(algo_name, compressed_data)
            decompression_time = time.time() - start_time
            
            # Verify correctness (not always possible due to data types)
            try:
                is_correct = utils.verify_decompression(data, decompressed_data)
            except:
                is_correct = "N/A"
            
            # Calculate compression ratio
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            results.append({
                'Algorithm': algo_name,
                'Original Size (bytes)': original_size,
                'Compressed Size (bytes)': compressed_size,
                'Compression Ratio (%)': compression_ratio,
                'Compression Time (s)': compression_time,
                'Decompression Time (s)': decompression_time,
                'Correctness': is_correct
            })
        except Exception as e:
            # Some algorithms may not work on certain data types
            results.append({
                'Algorithm': algo_name,
                'Original Size (bytes)': utils.get_data_size(data),
                'Compressed Size (bytes)': 'N/A',
                'Compression Ratio (%)': 'N/A',
                'Compression Time (s)': 'N/A',
                'Decompression Time (s)': 'N/A',
                'Correctness': 'N/A',
                'Error': str(e)
            })
    
    return pd.DataFrame(results)

def compare_across_data_types():
    """
    Compare compression performance across different data types
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with results for different data types
    """
    # Generate different types of data
    data_types = {
        'Text': dg.generate_text_data(1000),
        'Time Series': dg.generate_time_series(1000),
        'Categorical': dg.generate_categorical_data(1000, 20, 'skewed'),
        'Binary (Low Entropy)': dg.generate_binary_data(1000, 'low'),
        'Binary (High Entropy)': dg.generate_binary_data(1000, 'high')
    }
    
    algorithms = ['Huffman', 'Delta', 'LZW', 'RLE', 'Dictionary']
    all_results = []
    
    for data_type, data in data_types.items():
        # Benchmark each algorithm on this data type
        results = benchmark_algorithms(data, algorithms)
        results['Data Type'] = data_type
        all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)

def compare_compression_speeds(data_sizes=None):
    """
    Compare compression speeds across different data sizes
    
    Parameters:
    -----------
    data_sizes : list
        List of data sizes to test. If None, uses default sizes.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with compression speeds for different data sizes
    """
    if data_sizes is None:
        data_sizes = [1000, 10000, 100000]
    
    algorithms = ['Huffman', 'Delta', 'LZW', 'RLE', 'Dictionary']
    all_results = []
    
    for size in data_sizes:
        # Generate text data of the specified size
        data = dg.generate_text_data(size // 10)  # Each word is roughly 10 bytes
        
        # Benchmark each algorithm
        results = benchmark_algorithms(data, algorithms)
        results['Data Size'] = size
        all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)

def visualize_compression_comparison(results):
    """
    Create visualizations for compression benchmark results
    
    Parameters:
    -----------
    results : pd.DataFrame
        DataFrame with benchmark results
        
    Returns:
    --------
    fig
        Matplotlib figure with visualizations
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot compression ratio
    if 'Data Type' in results.columns:
        # Grouped by data type and algorithm
        pivot_ratio = results.pivot_table(
            values='Compression Ratio (%)', 
            index='Data Type', 
            columns='Algorithm',
            aggfunc='mean'
        )
        pivot_ratio.plot(kind='bar', ax=axes[0, 0], rot=45)
    else:
        # Just by algorithm
        results.plot(
            x='Algorithm', 
            y='Compression Ratio (%)', 
            kind='bar', 
            ax=axes[0, 0]
        )
    
    axes[0, 0].set_title('Compression Ratio by Algorithm')
    axes[0, 0].set_ylabel('Compression Ratio (%)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot compression time
    if 'Data Type' in results.columns:
        pivot_time = results.pivot_table(
            values='Compression Time (s)', 
            index='Data Type', 
            columns='Algorithm',
            aggfunc='mean'
        )
        pivot_time.plot(kind='bar', ax=axes[0, 1], rot=45)
    else:
        results.plot(
            x='Algorithm', 
            y='Compression Time (s)', 
            kind='bar', 
            ax=axes[0, 1]
        )
    
    axes[0, 1].set_title('Compression Time by Algorithm')
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot decompression time
    if 'Data Type' in results.columns:
        pivot_decomp = results.pivot_table(
            values='Decompression Time (s)', 
            index='Data Type', 
            columns='Algorithm',
            aggfunc='mean'
        )
        pivot_decomp.plot(kind='bar', ax=axes[1, 0], rot=45)
    else:
        results.plot(
            x='Algorithm', 
            y='Decompression Time (s)', 
            kind='bar', 
            ax=axes[1, 0]
        )
    
    axes[1, 0].set_title('Decompression Time by Algorithm')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot compression vs. decompression time
    if 'Data Type' not in results.columns:
        results.plot(
            x='Compression Time (s)', 
            y='Decompression Time (s)', 
            kind='scatter', 
            ax=axes[1, 1]
        )
        
        for i, row in results.iterrows():
            axes[1, 1].annotate(
                row['Algorithm'], 
                (row['Compression Time (s)'], row['Decompression Time (s)']),
                xytext=(5, 5),
                textcoords='offset points'
            )
    else:
        # Alternative plot for data type comparison
        pivot_size = results.pivot_table(
            values='Compressed Size (bytes)', 
            index='Data Type', 
            columns='Algorithm',
            aggfunc='mean'
        )
        pivot_size.plot(kind='bar', ax=axes[1, 1], rot=45)
        axes[1, 1].set_title('Compressed Size by Algorithm')
        axes[1, 1].set_ylabel('Size (bytes)')
    
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_algorithm_use_cases():
    """
    Analyze and recommend algorithms for different use cases
    
    Returns:
    --------
    dict
        Dictionary with use cases and recommended algorithms
    """
    use_cases = {
        'Real-time Data Streaming': {
            'Description': 'Applications requiring real-time compression/decompression with minimal latency',
            'Key Requirements': 'Fast compression/decompression, moderate compression ratio',
            'Recommended Algorithms': ['LZ4', 'Snappy', 'Delta encoding'],
            'Not Recommended': ['LZMA', 'Huffman (alone)'],
            'Implementation Notes': 'Use chunked data processing to minimize latency'
        },
        'Time Series Data': {
            'Description': 'Sensor data, financial markets, IoT measurements',
            'Key Requirements': 'Exploit temporal patterns and correlations',
            'Recommended Algorithms': ['Delta encoding', 'Delta-of-delta', 'Gorilla'],
            'Not Recommended': ['Dictionary encoding', 'LZW'],
            'Implementation Notes': 'Consider different compression for different columns/metrics'
        },
        'Log Files & Text Data': {
            'Description': 'Server logs, text documents, configuration files',
            'Key Requirements': 'Handle variable-length text efficiently',
            'Recommended Algorithms': ['LZW', 'Huffman', 'Deflate/GZIP'],
            'Not Recommended': ['RLE', 'Delta encoding'],
            'Implementation Notes': 'Preprocess to normalize formats for better compression'
        },
        'Database Columnar Storage': {
            'Description': 'Column-oriented databases and warehouses',
            'Key Requirements': 'Efficient compression of similar data types',
            'Recommended Algorithms': [
                'Dictionary encoding (low cardinality)', 
                'RLE (repeated values)', 
                'FOR/PFOR (numerical data)'
            ],
            'Not Recommended': ['One-size-fits-all approach'],
            'Implementation Notes': 'Use different algorithms for different column types'
        },
        'Distributed File Systems': {
            'Description': 'Hadoop, S3, distributed filesystems',
            'Key Requirements': 'Balance compression ratio and CPU usage',
            'Recommended Algorithms': ['ZSTD', 'Brotli', 'GZIP'],
            'Not Recommended': ['CPU-intensive algorithms for hot data'],
            'Implementation Notes': 'Consider tiered compression strategies based on access patterns'
        },
        'Archival Storage': {
            'Description': 'Long-term storage with infrequent access',
            'Key Requirements': 'Maximum compression ratio, decompression speed more important than compression speed',
            'Recommended Algorithms': ['LZMA', 'BZIP2', 'Combined algorithms'],
            'Not Recommended': ['Speed-optimized algorithms'],
            'Implementation Notes': 'Pre-process data to identify and exploit patterns'
        }
    }
    
    return use_cases

def benchmark_on_distributed_scenario():
    """
    Simulate and benchmark compression in a distributed system scenario
    
    Returns:
    --------
    dict
        Dictionary with distributed metrics
    """
    # Simulate different nodes with different data characteristics
    nodes = {
        'Node1': {'data_type': 'Time Series', 'size': 50000, 'pattern': 'smooth'},
        'Node2': {'data_type': 'Text', 'size': 100000, 'pattern': 'repetitive'},
        'Node3': {'data_type': 'Categorical', 'size': 75000, 'pattern': 'skewed'},
        'Node4': {'data_type': 'Binary', 'size': 80000, 'pattern': 'low'},
        'Node5': {'data_type': 'Mixed', 'size': 60000, 'pattern': 'varied'}
    }
    
    # Generate network bandwidth scenarios (MB/s)
    network_scenarios = {
        'Local Network': 1000,  # 1 GB/s
        'WAN': 100,           # 100 MB/s
        'Internet': 10,       # 10 MB/s
        'Mobile': 1           # 1 MB/s
    }
    
    algorithms = ['LZW', 'Huffman', 'Delta', 'RLE']
    
    # Calculate transfer times with and without compression
    results = []
    
    for node_name, node_info in nodes.items():
        # Generate appropriate data
        if node_info['data_type'] == 'Time Series':
            data = dg.generate_time_series(node_info['size'] // 8)  # 8 bytes per float64
        elif node_info['data_type'] == 'Text':
            data = dg.generate_text_data(node_info['size'] // 10)  # ~10 bytes per word
        elif node_info['data_type'] == 'Categorical':
            data = dg.generate_categorical_data(node_info['size'] // 10, 20, node_info['pattern'])
        elif node_info['data_type'] == 'Binary':
            data = dg.generate_binary_data(node_info['size'], node_info['pattern'])
        else:  # Mixed
            mixed_data = dg.generate_mixed_dataset(node_info['size'] // 100)  # Rough estimate
            data = str(mixed_data)  # Convert to string for testing
        
        # Calculate uncompressed transfer time
        original_size = utils.get_data_size(data)
        
        for algo_name in algorithms:
            try:
                # Compress the data
                start_time = time.time()
                compressed_data = utils.compress_with_algorithm(algo_name, data)
                compression_time = time.time() - start_time
                
                # Calculate compressed size
                compressed_size = utils.get_data_size(compressed_data)
                
                # Calculate compression ratio
                compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
                
                # Calculate network transfer times for each scenario
                for scenario, bandwidth in network_scenarios.items():
                    # Convert sizes to MB for bandwidth calculation
                    orig_size_mb = original_size / (1024 * 1024)
                    comp_size_mb = compressed_size / (1024 * 1024)
                    
                    # Transfer times in seconds
                    uncompressed_transfer = orig_size_mb / bandwidth
                    compressed_transfer = comp_size_mb / bandwidth
                    
                    # Total time including compression
                    total_time_with_compression = compression_time + compressed_transfer
                    
                    # Time saved
                    time_saved = uncompressed_transfer - total_time_with_compression
                    time_saved_percent = (time_saved / uncompressed_transfer) * 100 if uncompressed_transfer > 0 else 0
                    
                    results.append({
                        'Node': node_name,
                        'Data Type': node_info['data_type'],
                        'Algorithm': algo_name,
                        'Network Scenario': scenario,
                        'Original Size (MB)': orig_size_mb,
                        'Compressed Size (MB)': comp_size_mb,
                        'Compression Ratio (%)': compression_ratio,
                        'Compression Time (s)': compression_time,
                        'Uncompressed Transfer Time (s)': uncompressed_transfer,
                        'Compressed Transfer Time (s)': compressed_transfer,
                        'Total Time With Compression (s)': total_time_with_compression,
                        'Time Saved (s)': time_saved,
                        'Time Saved (%)': time_saved_percent,
                        'Worth Compressing': time_saved > 0
                    })
            except Exception as e:
                # Skip failed algorithm for this data type
                print(f"Error with {algo_name} on {node_name}: {e}")
    
    return pd.DataFrame(results)

def get_algorithm_recommendations(data_type, data_size, access_pattern, network_speed):
    """
    Get algorithm recommendations based on data characteristics
    
    Parameters:
    -----------
    data_type : str
        Type of data ('Text', 'Time Series', 'Categorical', 'Binary', 'Mixed')
    data_size : str
        Size of data ('Small', 'Medium', 'Large')
    access_pattern : str
        Data access pattern ('Frequent', 'Moderate', 'Rare')
    network_speed : str
        Network connection speed ('Fast', 'Medium', 'Slow')
        
    Returns:
    --------
    dict
        Dictionary with algorithm recommendations
    """
    recommendations = {}
    
    # Primary recommendation based on data type
    if data_type == 'Text':
        recommendations['primary'] = {
            'algorithm': 'LZW or Deflate (GZIP)',
            'reason': 'Effective for text with recurring patterns and phrases'
        }
    elif data_type == 'Time Series':
        recommendations['primary'] = {
            'algorithm': 'Delta Encoding',
            'reason': 'Excellent for time series data where consecutive values are often similar'
        }
    elif data_type == 'Categorical':
        recommendations['primary'] = {
            'algorithm': 'Dictionary Encoding',
            'reason': 'Optimal for categorical data with limited unique values'
        }
    elif data_type == 'Binary':
        recommendations['primary'] = {
            'algorithm': 'Huffman or Arithmetic Coding',
            'reason': 'Adapts well to binary data distributions'
        }
    else:  # Mixed
        recommendations['primary'] = {
            'algorithm': 'Combined approach',
            'reason': 'Use specialized algorithms for each column/data type'
        }
    
    # Adjust for data size
    if data_size == 'Large':
        if access_pattern == 'Frequent':
            recommendations['size_adjustment'] = {
                'algorithm': 'Focus on speed: LZ4, Snappy',
                'reason': 'Large datasets with frequent access require fast compression/decompression'
            }
        else:
            recommendations['size_adjustment'] = {
                'algorithm': 'Consider chunking with parallel compression',
                'reason': 'Break large data into manageable chunks that can be processed in parallel'
            }
    
    # Adjust for access pattern
    if access_pattern == 'Rare':
        recommendations['access_adjustment'] = {
            'algorithm': 'LZMA, BZIP2',
            'reason': 'For rarely accessed data, prioritize compression ratio over speed'
        }
    
    # Adjust for network speed
    if network_speed == 'Slow':
        recommendations['network_adjustment'] = {
            'algorithm': 'Higher compression ratio recommended',
            'reason': 'In slow networks, the time saved in transfer outweighs compression time'
        }
    elif network_speed == 'Fast':
        recommendations['network_adjustment'] = {
            'algorithm': 'Consider if compression is necessary',
            'reason': 'In very fast networks, compression might not save overall time'
        }
    
    # Implementation recommendations
    if data_type == 'Time Series':
        recommendations['implementation'] = {
            'strategy': 'Consider delta-of-delta for smoother series',
            'distribution_approach': 'Pre-compress at source before distribution'
        }
    elif data_type == 'Text':
        recommendations['implementation'] = {
            'strategy': 'Consider preprocessing to normalize text patterns',
            'distribution_approach': 'Compress once, distribute to many recipients'
        }
    else:
        recommendations['implementation'] = {
            'strategy': 'Profile your specific data distribution',
            'distribution_approach': 'Benchmark multiple algorithms on sample data'
        }
    
    return recommendations
