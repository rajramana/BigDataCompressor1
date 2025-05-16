import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random

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
    # Default to all algorithms if none specified
    if algorithms is None:
        algorithms = ['huffman', 'delta', 'delta_of_delta', 'lzw', 'rle', 'dictionary', 'for']
    
    # Validate input
    valid_algorithms = ['huffman', 'delta', 'delta_of_delta', 'lzw', 'rle', 'dictionary', 'for']
    algorithms = [algo for algo in algorithms if algo in valid_algorithms]
    
    # Store results
    results = []
    
    # Benchmark each algorithm
    for algorithm in algorithms:
        try:
            result = ca.benchmark_compression(algorithm, data)
            results.append(result)
        except Exception as e:
            # Add failed result
            results.append({
                'algorithm': algorithm,
                'original_size': None,
                'compressed_size': None,
                'compression_ratio': None,
                'compression_time': None,
                'decompression_time': None,
                'error': str(e)
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add speed columns (bytes per second)
    df['compression_speed'] = df.apply(
        lambda row: row['original_size'] / row['compression_time'] if row['compression_time'] else None, 
        axis=1
    )
    df['decompression_speed'] = df.apply(
        lambda row: row['compressed_size'] / row['decompression_time'] if row['decompression_time'] else None, 
        axis=1
    )
    
    return df

def compare_across_data_types():
    """
    Compare compression performance across different data types
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with results for different data types
    """
    # Generate different types of data
    time_series = dg.generate_time_series(1000)
    text = dg.generate_text_data(500)
    categorical = dg.generate_categorical_data(1000, 20, "skewed")
    binary_low = dg.generate_binary_data(1000, "low")
    binary_high = dg.generate_binary_data(1000, "high")
    
    # Define data types and corresponding datasets
    data_types = {
        "Time Series": time_series,
        "Text": text,
        "Categorical": categorical,
        "Binary (Low Entropy)": binary_low,
        "Binary (High Entropy)": binary_high
    }
    
    # Store results
    all_results = []
    
    # Benchmark each data type
    for data_type, data in data_types.items():
        # Run benchmarks
        df = benchmark_algorithms(data)
        
        # Add data type column
        df['data_type'] = data_type
        
        # Append to results
        all_results.append(df)
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    return combined_df

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
    # Default sizes if none provided
    if data_sizes is None:
        data_sizes = [1000, 10000, 100000]
    
    # Define algorithms to test
    algorithms = ['huffman', 'delta', 'lzw', 'rle', 'dictionary', 'for']
    
    # Store results
    all_results = []
    
    # Test with text data
    for size in data_sizes:
        # Generate text data
        data = dg.generate_text_data(size // 10)  # Each word is roughly 10 chars
        
        # Run benchmarks
        df = benchmark_algorithms(data, algorithms)
        
        # Add size column
        df['data_size'] = size
        
        # Append to results
        all_results.append(df)
    
    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    return combined_df

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
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Compression Ratio Comparison (by algorithm)
    axes[0, 0].set_title('Compression Ratio by Algorithm')
    avg_ratios = results.groupby('algorithm')['compression_ratio'].mean()
    avg_ratios.plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_ylabel('Compression Ratio (%)')
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Compression Speed Comparison
    axes[0, 1].set_title('Compression Speed by Algorithm')
    avg_speeds = results.groupby('algorithm')['compression_speed'].mean() / 1024  # KB/s
    avg_speeds.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_ylabel('Speed (KB/s)')
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Compression Ratio by Data Type (if data_type column exists)
    if 'data_type' in results.columns:
        pivot_ratio = results.pivot_table(
            index='algorithm', columns='data_type', values='compression_ratio'
        )
        pivot_ratio.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Compression Ratio by Data Type')
        axes[1, 0].set_ylabel('Compression Ratio (%)')
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].legend(title='Data Type')
        axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Compression vs Decompression Speed
    if 'decompression_speed' in results.columns:
        # Calculate average speeds
        avg_comp = results.groupby('algorithm')['compression_speed'].mean() / 1024  # KB/s
        avg_decomp = results.groupby('algorithm')['decompression_speed'].mean() / 1024  # KB/s
        
        # Combine into DataFrame
        speed_df = pd.DataFrame({
            'Compression': avg_comp,
            'Decompression': avg_decomp
        })
        
        # Plot
        speed_df.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Compression vs Decompression Speed')
        axes[1, 1].set_ylabel('Speed (KB/s)')
        axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
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
    # Define use cases
    use_cases = {
        "Time Series Data Storage": {
            "description": "Storing large volumes of time series data with moderate access frequency",
            "data_characteristics": "Sequential numerical data with trends and patterns",
            "algorithms": ["delta", "delta_of_delta", "for"],
            "reasoning": "Delta-based algorithms excel at compressing time series by storing differences between consecutive values, which are typically small."
        },
        "Text Document Storage": {
            "description": "Archiving text documents with infrequent access",
            "data_characteristics": "Natural language text with word repetition and skewed character distribution",
            "algorithms": ["huffman", "lzw"],
            "reasoning": "Huffman coding leverages character frequency patterns, while LZW captures recurring phrases and patterns in text."
        },
        "Database Column Compression": {
            "description": "Compressing database columns for improved I/O performance",
            "data_characteristics": "Structured data with limited distinct values per column",
            "algorithms": ["dictionary", "for", "rle"],
            "reasoning": "Dictionary encoding is ideal for low-cardinality columns, FOR works well for numerical ranges, and RLE for columns with repeated values."
        },
        "Network Data Transfer": {
            "description": "Compressing data for network transfer in distributed systems",
            "data_characteristics": "Mixed data types with speed requirements",
            "algorithms": ["lzw", "delta"],
            "reasoning": "LZW offers good general-purpose compression with reasonable speed, while delta encoding is very fast for numerical data."
        },
        "Real-time Sensor Data": {
            "description": "Processing and storing real-time sensor readings",
            "data_characteristics": "Continuous numerical data streams with high correlation",
            "algorithms": ["delta", "rle"],
            "reasoning": "Both algorithms offer extremely fast compression and decompression, critical for real-time applications."
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
    # Generate a dataset representing a distributed data processing scenario
    # Mix of numerical, categorical, and text data
    n_records = 5000
    dataset = dg.generate_mixed_dataset(n_records)
    
    # Define compression configurations to test
    compression_configs = [
        {"name": "No Compression", "algorithms": {}},
        {"name": "Type-specific Compression", "algorithms": {
            "numerical": "delta",
            "categorical": "dictionary",
            "text": "huffman"
        }},
        {"name": "Fast Compression", "algorithms": {
            "numerical": "delta",
            "categorical": "rle",
            "text": "lzw"
        }},
        {"name": "High Ratio Compression", "algorithms": {
            "numerical": "for",
            "categorical": "dictionary",
            "text": "huffman"
        }}
    ]
    
    # Simulate network bandwidths (bytes/second)
    network_bandwidths = {
        "Local Network (1 GB/s)": 1_000_000_000,
        "WAN (100 MB/s)": 100_000_000,
        "Internet (10 MB/s)": 10_000_000,
        "Mobile (1 MB/s)": 1_000_000
    }
    
    # Store results
    results = {
        "data_size": {},
        "compression_time": {},
        "decompression_time": {},
        "transfer_time": {},
        "total_time": {}
    }
    
    # Calculate original data size
    original_size = 0
    for column, data in dataset.items():
        if isinstance(data, list):
            # Estimate size
            if all(isinstance(x, (int, float)) for x in data[:100]):
                # Numerical data
                original_size += len(data) * 8  # 8 bytes per float
            elif all(isinstance(x, str) for x in data[:100]):
                # Text/categorical data
                original_size += sum(len(str(x)) for x in data)
    
    results["data_size"]["No Compression"] = original_size
    
    # Simulate compression and transfer for each configuration
    for config in compression_configs:
        config_name = config["name"]
        algorithms = config["algorithms"]
        
        if config_name == "No Compression":
            # No compression benchmark
            compressed_size = original_size
            compression_time = 0
            decompression_time = 0
        else:
            # Apply compression to each column based on data type
            compressed_size = 0
            compression_time = 0
            decompression_time = 0
            
            for column, data in dataset.items():
                # Detect data type
                data_type = utils.detect_data_type(data)
                
                # Select algorithm (default to None)
                algorithm = algorithms.get(data_type)
                
                if algorithm:
                    # Benchmark compression
                    result = ca.benchmark_compression(algorithm, data)
                    
                    # Accumulate metrics
                    compressed_size += result.get("compressed_size", 0) or 0
                    compression_time += result.get("compression_time", 0) or 0
                    decompression_time += result.get("decompression_time", 0) or 0
                else:
                    # No compression for this column
                    if isinstance(data, list):
                        col_size = sum(len(str(x)) for x in data) if all(isinstance(x, str) for x in data[:10]) else len(data) * 8
                    else:
                        col_size = len(str(data))
                    
                    compressed_size += col_size
        
        # Store compression results
        results["data_size"][config_name] = compressed_size
        results["compression_time"][config_name] = compression_time
        results["decompression_time"][config_name] = decompression_time
        
        # Calculate transfer times for different network speeds
        transfer_times = {}
        for network, bandwidth in network_bandwidths.items():
            transfer_time = compressed_size / bandwidth
            transfer_times[network] = transfer_time
        
        results["transfer_time"][config_name] = transfer_times
        
        # Calculate total processing times (compression + transfer + decompression)
        total_times = {}
        for network, transfer_time in transfer_times.items():
            total_time = compression_time + transfer_time + decompression_time
            total_times[network] = total_time
        
        results["total_time"][config_name] = total_times
    
    return results

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
    # Define recommendation weights
    weights = {
        # For different data types (compression_ratio, speed)
        'data_type': {
            'Text': {
                'huffman': (0.9, 0.7),
                'lzw': (0.8, 0.7),
                'rle': (0.3, 0.9),
                'dictionary': (0.7, 0.8),
                'delta': (0.1, 0.9),
                'delta_of_delta': (0.1, 0.8),
                'for': (0.1, 0.9)
            },
            'Time Series': {
                'huffman': (0.4, 0.7),
                'lzw': (0.5, 0.7),
                'rle': (0.3, 0.9),
                'dictionary': (0.5, 0.8),
                'delta': (0.9, 0.9),
                'delta_of_delta': (0.95, 0.8),
                'for': (0.8, 0.9)
            },
            'Categorical': {
                'huffman': (0.6, 0.7),
                'lzw': (0.7, 0.7),
                'rle': (0.7, 0.9),
                'dictionary': (0.95, 0.8),
                'delta': (0.2, 0.9),
                'delta_of_delta': (0.1, 0.8),
                'for': (0.3, 0.9)
            },
            'Binary': {
                'huffman': (0.7, 0.7),
                'lzw': (0.8, 0.7),
                'rle': (0.9, 0.9),
                'dictionary': (0.5, 0.8),
                'delta': (0.3, 0.9),
                'delta_of_delta': (0.2, 0.8),
                'for': (0.3, 0.9)
            },
            'Mixed': {
                'huffman': (0.7, 0.7),
                'lzw': (0.8, 0.7),
                'rle': (0.5, 0.9),
                'dictionary': (0.7, 0.8),
                'delta': (0.6, 0.9),
                'delta_of_delta': (0.5, 0.8),
                'for': (0.6, 0.9)
            }
        },
        
        # Multipliers for data size
        'data_size': {
            'Small': {'compression_ratio': 0.7, 'speed': 1.3},
            'Medium': {'compression_ratio': 1.0, 'speed': 1.0},
            'Large': {'compression_ratio': 1.3, 'speed': 0.7}
        },
        
        # Multipliers for access pattern
        'access_pattern': {
            'Frequent': {'compression_ratio': 0.7, 'speed': 1.3},
            'Moderate': {'compression_ratio': 1.0, 'speed': 1.0},
            'Rare': {'compression_ratio': 1.3, 'speed': 0.7}
        },
        
        # Multipliers for network speed
        'network_speed': {
            'Fast': {'compression_ratio': 0.7, 'speed': 1.3},
            'Medium': {'compression_ratio': 1.0, 'speed': 1.0},
            'Slow': {'compression_ratio': 1.3, 'speed': 0.7}
        }
    }
    
    # Calculate scores for each algorithm
    algorithm_scores = {}
    
    # Base weights from data type
    data_type_weights = weights['data_type'].get(data_type, weights['data_type']['Mixed'])
    
    # Multipliers
    size_multiplier = weights['data_size'].get(data_size, weights['data_size']['Medium'])
    access_multiplier = weights['access_pattern'].get(access_pattern, weights['access_pattern']['Moderate'])
    network_multiplier = weights['network_speed'].get(network_speed, weights['network_speed']['Medium'])
    
    # Calculate final scores
    for algorithm, (ratio_weight, speed_weight) in data_type_weights.items():
        # Apply multipliers
        ratio_score = ratio_weight * size_multiplier['compression_ratio'] * access_multiplier['compression_ratio'] * network_multiplier['compression_ratio']
        speed_score = speed_weight * size_multiplier['speed'] * access_multiplier['speed'] * network_multiplier['speed']
        
        # Overall score (equal weight to both factors)
        overall_score = (ratio_score + speed_score) / 2
        
        algorithm_scores[algorithm] = {
            'compression_score': ratio_score,
            'speed_score': speed_score,
            'overall_score': overall_score
        }
    
    # Sort algorithms by overall score
    sorted_algorithms = sorted(
        algorithm_scores.items(),
        key=lambda x: x[1]['overall_score'],
        reverse=True
    )
    
    # Extract recommendations
    recommendations = {
        'top_overall': sorted_algorithms[0][0],
        'top_compression': sorted(
            algorithm_scores.items(),
            key=lambda x: x[1]['compression_score'],
            reverse=True
        )[0][0],
        'top_speed': sorted(
            algorithm_scores.items(),
            key=lambda x: x[1]['speed_score'],
            reverse=True
        )[0][0],
        'all_scores': algorithm_scores,
        'recommendations': [
            {
                'algorithm': algo,
                'overall_score': score['overall_score'],
                'compression_score': score['compression_score'],
                'speed_score': score['speed_score']
            }
            for algo, score in sorted_algorithms[:3]
        ]
    }
    
    return recommendations