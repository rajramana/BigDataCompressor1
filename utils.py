import numpy as np
import math
import re

def estimate_bits_needed(values):
    """
    Estimate the number of bits needed to represent each value in a list
    
    Parameters:
    -----------
    values : list or numpy.ndarray
        List of values to estimate bit requirements for
        
    Returns:
    --------
    list
        List of bit counts required for each value
    """
    result = []
    for val in values:
        # For zero we need just 1 bit
        if val == 0:
            result.append(1)
        # For positive integers
        elif val > 0:
            result.append(math.ceil(math.log2(val + 1)))
        # For negative values we need 1 extra bit for sign
        else:
            result.append(math.ceil(math.log2(abs(val) + 1)) + 1)
    
    return result

def calculate_entropy(data):
    """
    Calculate the Shannon entropy of data
    
    Parameters:
    -----------
    data : str, list, or array-like
        Data to calculate entropy for
        
    Returns:
    --------
    float
        Shannon entropy in bits per symbol
    """
    if isinstance(data, str):
        # For text data, calculate character frequencies
        freq = {}
        for char in data:
            if char in freq:
                freq[char] += 1
            else:
                freq[char] = 1
    else:
        # For other types, convert to list and count frequencies
        data_list = list(data)
        freq = {}
        for val in data_list:
            if val in freq:
                freq[val] += 1
            else:
                freq[val] = 1
    
    # Calculate entropy
    n = sum(freq.values())
    entropy = 0.0
    
    for count in freq.values():
        p = count / n
        entropy -= p * math.log2(p)
    
    return entropy

def detect_data_type(data):
    """
    Attempt to detect the type of data
    
    Parameters:
    -----------
    data : object
        Data to analyze
        
    Returns:
    --------
    str
        Detected data type ('text', 'numerical', 'time_series', 'categorical', 'binary', 'mixed')
    """
    # Handle string data
    if isinstance(data, str):
        # Check if it's a comma-separated list of numbers
        if re.match(r'^[\d,.]+$', data):
            return 'numerical'
        return 'text'
    
    # Handle bytes data
    if isinstance(data, bytes):
        return 'binary'
    
    # Handle list or numpy array
    if isinstance(data, (list, np.ndarray)):
        # Empty data
        if len(data) == 0:
            return 'unknown'
        
        # Check first few elements
        sample = data[:min(100, len(data))]
        
        # Count types
        num_count = 0
        str_count = 0
        other_count = 0
        
        for item in sample:
            if isinstance(item, (int, float, np.number)):
                num_count += 1
            elif isinstance(item, str):
                str_count += 1
            else:
                other_count += 1
        
        # Determine predominant type
        if num_count == len(sample):
            # Check if it's sequential (possible time series)
            if len(sample) > 5:
                diffs = np.diff(sample[:10])
                # If differences are small and consistent, likely time series
                if np.std(diffs) / np.mean(np.abs(diffs)) < 0.5:
                    return 'time_series'
            return 'numerical'
        elif str_count == len(sample):
            # Count unique values to determine if categorical
            unique_ratio = len(set(sample)) / len(sample)
            if unique_ratio < 0.3:  # Low cardinality
                return 'categorical'
            return 'text'
        elif num_count > str_count and num_count > other_count:
            return 'numerical'
        elif str_count > num_count and str_count > other_count:
            return 'text'
        else:
            return 'mixed'
    
    # Default
    return 'unknown'

def analyze_runs(data):
    """
    Analyze the run-length characteristics of data
    
    Parameters:
    -----------
    data : list, str, or array-like
        Data to analyze
        
    Returns:
    --------
    dict
        Dictionary with run analysis results
    """
    if isinstance(data, str):
        data_list = list(data)
    else:
        data_list = list(data)
    
    if not data_list:
        return {"run_count": 0, "run_ratio": 0, "avg_run_length": 0}
    
    # Count runs
    run_count = 1
    current = data_list[0]
    
    for i in range(1, len(data_list)):
        if data_list[i] != current:
            run_count += 1
            current = data_list[i]
    
    # Calculate run ratio (lower means more runs, more compressible by RLE)
    run_ratio = run_count / len(data_list)
    
    # Calculate average run length
    avg_run_length = len(data_list) / run_count
    
    return {
        "run_count": run_count,
        "run_ratio": run_ratio,
        "avg_run_length": avg_run_length
    }

def analyze_range_compression(data):
    """
    Analyze the potential for range-based compression (like delta or FOR)
    
    Parameters:
    -----------
    data : list or array-like of numbers
        Numerical data to analyze
        
    Returns:
    --------
    dict
        Dictionary with range analysis results
    """
    # Convert to numpy array if not already
    try:
        data_arr = np.array(data, dtype=float)
    except (ValueError, TypeError):
        return {"range_compression_potential": 1.0}
    
    # Calculate statistics
    try:
        data_range = np.max(data_arr) - np.min(data_arr)
        data_std = np.std(data_arr)
        unique_count = len(np.unique(data_arr))
        
        # Theoretical bits required for full range vs standard deviation
        range_bits = max(1, np.ceil(np.log2(data_range + 1))) if data_range > 0 else 1
        std_bits = max(1, np.ceil(np.log2(data_std * 6 + 1))) if data_std > 0 else 1
        
        # Potential compression ratio (higher is better)
        # Compare bits needed for full-precision values vs range-based encoding
        full_bits = 64  # Assume 64-bit floating point
        compressed_bits = range_bits
        
        range_compression_potential = full_bits / compressed_bits
        
        # Analyze delta potential
        if len(data_arr) > 1:
            deltas = np.diff(data_arr)
            delta_range = np.max(deltas) - np.min(deltas)
            delta_bits = max(1, np.ceil(np.log2(delta_range + 1))) if delta_range > 0 else 1
            delta_compression_potential = range_bits / delta_bits if delta_bits > 0 else 1.0
        else:
            delta_compression_potential = 1.0
        
        return {
            "min_value": float(np.min(data_arr)),
            "max_value": float(np.max(data_arr)),
            "range": float(data_range),
            "std_dev": float(data_std),
            "unique_count": int(unique_count),
            "unique_ratio": float(unique_count / len(data_arr)),
            "range_compression_potential": float(range_compression_potential),
            "delta_compression_potential": float(delta_compression_potential)
        }
    except:
        return {"range_compression_potential": 1.0}

def calculate_dictionary_potential(data):
    """
    Calculate the potential for dictionary-based compression
    
    Parameters:
    -----------
    data : list, str, or array-like
        Data to analyze
        
    Returns:
    --------
    dict
        Dictionary with dictionary compression analysis results
    """
    if isinstance(data, str):
        data_list = list(data)
    else:
        data_list = list(data)
    
    if not data_list:
        return {"dictionary_potential": 0, "cardinality": 0}
    
    # Count unique values
    unique_values = set(data_list)
    cardinality = len(unique_values)
    
    # Calculate cardinality ratio (lower means better dictionary compression)
    cardinality_ratio = cardinality / len(data_list)
    
    # Calculate potential bits saved
    original_bits = 8 * len(data_list)  # Assuming 8 bits per element
    bits_per_code = max(1, math.ceil(math.log2(cardinality)))
    dictionary_bits = cardinality * 16  # Assuming 16 bits per dictionary entry
    encoded_bits = len(data_list) * bits_per_code
    
    total_bits = dictionary_bits + encoded_bits
    compression_ratio = 1 - (total_bits / original_bits)
    
    # Calculate dictionary potential score (higher is better)
    dictionary_potential = max(0, compression_ratio)
    
    return {
        "dictionary_potential": dictionary_potential,
        "cardinality": cardinality,
        "cardinality_ratio": cardinality_ratio,
        "bits_per_code": bits_per_code
    }

def get_sample_data(data, max_size=1000):
    """
    Get a representative sample of data for analysis
    
    Parameters:
    -----------
    data : object
        Data to sample
    max_size : int
        Maximum sample size
        
    Returns:
    --------
    object
        Sampled data
    """
    if isinstance(data, (str, list, np.ndarray)):
        if len(data) <= max_size:
            return data
        
        # Take samples from beginning, middle, and end
        third = max_size // 3
        sample = []
        
        # Beginning
        sample.extend(data[:third])
        
        # Middle
        mid_start = max(0, (len(data) - third) // 2)
        sample.extend(data[mid_start:mid_start + third])
        
        # End
        sample.extend(data[-third:])
        
        return sample
    
    return data