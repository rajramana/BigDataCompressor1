import numpy as np
import time
import sys
import json
import math
from collections import Counter
import compression_algorithms as ca
import data_generator as dg

def get_data_size(data):
    """
    Calculate the size of data in bytes
    
    Parameters:
    -----------
    data : various
        The data to measure
        
    Returns:
    --------
    int
        Size in bytes
    """
    if isinstance(data, str):
        return len(data.encode('utf-8'))
    elif isinstance(data, bytes) or isinstance(data, bytearray):
        return len(data)
    elif isinstance(data, list) or isinstance(data, tuple):
        if all(isinstance(x, str) for x in data):
            return sum(len(x.encode('utf-8')) for x in data)
        else:
            # Convert to JSON and measure
            try:
                return len(json.dumps(data).encode('utf-8'))
            except:
                # Fallback for non-serializable lists
                return sys.getsizeof(data)
    elif isinstance(data, np.ndarray):
        return data.nbytes
    elif isinstance(data, dict):
        try:
            return len(json.dumps(data).encode('utf-8'))
        except:
            # Fallback
            return sys.getsizeof(data)
    else:
        # Fallback to sys.getsizeof for other types
        return sys.getsizeof(data)

def compress_with_algorithm(algorithm_name, data):
    """
    Compress data using the specified algorithm
    
    Parameters:
    -----------
    algorithm_name : str
        Name of the compression algorithm to use
    data : various
        Data to compress
        
    Returns:
    --------
    various
        Compressed data
    """
    if algorithm_name == 'Huffman':
        if isinstance(data, str):
            # For demonstration purposes, using our simple implementation
            _, _, codes = ca.huffman_coding_demo(data)
            
            # Encode the data using the generated codes
            encoded = ''.join(codes[c] for c in data)
            
            # Pack bits into bytes for accurate size measurement
            # (simplified for demonstration)
            return encoded
        else:
            raise ValueError("Huffman coding currently only supports string data")
    
    elif algorithm_name == 'Delta':
        if isinstance(data, np.ndarray) or isinstance(data, list):
            # Convert to numpy array if it's a list
            arr = np.array(data) if isinstance(data, list) else data
            
            # Check if data is numeric
            if np.issubdtype(arr.dtype, np.number):
                first_value, deltas = ca.delta_encode(arr)
                return (first_value, deltas)
            else:
                raise ValueError("Delta encoding requires numeric data")
        else:
            raise ValueError("Delta encoding requires array-like data")
    
    elif algorithm_name == 'LZW':
        if isinstance(data, str):
            return ca.lzw_compress(data)
        elif isinstance(data, bytes):
            return ca.lzw_compress(data.decode('utf-8', errors='replace'))
        elif isinstance(data, list) and all(isinstance(x, str) for x in data):
            return ca.lzw_compress(''.join(data))
        else:
            # Try to convert to string
            try:
                return ca.lzw_compress(str(data))
            except:
                raise ValueError("LZW compression requires string-like data")
    
    elif algorithm_name == 'RLE':
        if isinstance(data, list) or isinstance(data, np.ndarray):
            return ca.rle_encode(data)
        elif isinstance(data, str):
            return ca.rle_encode(list(data))
        elif isinstance(data, bytes):
            return ca.rle_encode(list(data))
        else:
            raise ValueError("RLE requires sequence data")
    
    elif algorithm_name == 'Dictionary':
        if isinstance(data, list) or isinstance(data, np.ndarray):
            return ca.dictionary_encode(data)
        elif isinstance(data, str):
            return ca.dictionary_encode(list(data))
        else:
            raise ValueError("Dictionary encoding requires sequence data")
    
    elif algorithm_name == 'FOR':
        if isinstance(data, np.ndarray) or isinstance(data, list):
            # Convert to numpy array if it's a list
            arr = np.array(data) if isinstance(data, list) else data
            
            # Check if data is numeric
            if np.issubdtype(arr.dtype, np.number):
                return ca.for_encode(arr)
            else:
                raise ValueError("FOR requires numeric data")
        else:
            raise ValueError("FOR requires array-like data")
    
    else:
        raise ValueError(f"Unknown compression algorithm: {algorithm_name}")

def decompress_with_algorithm(algorithm_name, compressed_data):
    """
    Decompress data using the specified algorithm
    
    Parameters:
    -----------
    algorithm_name : str
        Name of the compression algorithm to use
    compressed_data : various
        Data to decompress
        
    Returns:
    --------
    various
        Decompressed data
    """
    if algorithm_name == 'Huffman':
        # This is a simplified version for demonstration purposes
        # In a real application, we would need the Huffman tree/codes to decompress
        # For now, we'll just return the compressed data
        return compressed_data
    
    elif algorithm_name == 'Delta':
        first_value, deltas = compressed_data
        return ca.delta_decode(first_value, deltas)
    
    elif algorithm_name == 'LZW':
        return ca.lzw_decompress(compressed_data)
    
    elif algorithm_name == 'RLE':
        return ca.rle_decode(compressed_data)
    
    elif algorithm_name == 'Dictionary':
        encoded, value_to_id = compressed_data
        return ca.dictionary_decode(encoded, value_to_id)
    
    elif algorithm_name == 'FOR':
        reference, offsets, bits_per_value = compressed_data
        return ca.for_decode(reference, offsets, bits_per_value)
    
    else:
        raise ValueError(f"Unknown decompression algorithm: {algorithm_name}")

def verify_decompression(original_data, decompressed_data):
    """
    Verify that decompression recovered the original data correctly
    
    Parameters:
    -----------
    original_data : various
        Original data before compression
    decompressed_data : various
        Data after decompression
        
    Returns:
    --------
    bool
        True if decompression was successful, False otherwise
    """
    if isinstance(original_data, np.ndarray) and isinstance(decompressed_data, np.ndarray):
        return np.array_equal(original_data, decompressed_data)
    elif isinstance(original_data, list) and isinstance(decompressed_data, list):
        return original_data == decompressed_data
    elif isinstance(original_data, str) and isinstance(decompressed_data, str):
        return original_data == decompressed_data
    elif isinstance(original_data, bytes) and isinstance(decompressed_data, bytes):
        return original_data == decompressed_data
    else:
        # Try direct comparison, but this might not work for all types
        try:
            return original_data == decompressed_data
        except:
            return False

def estimate_bits_needed(values):
    """
    Estimate the number of bits needed to represent each value in an array
    
    Parameters:
    -----------
    values : array-like
        Array of values to analyze
        
    Returns:
    --------
    list
        List of bit counts needed for each value
    """
    # Convert to numpy array if not already
    arr = np.array(values)
    
    # For each value, calculate the minimum number of bits required
    # including sign bit for negative numbers
    bits_needed = []
    
    for value in arr:
        if value == 0:
            bits_needed.append(1)  # Need at least 1 bit even for 0
        else:
            # For negative numbers, we add 1 bit for the sign
            abs_value = abs(value)
            
            # Calculate bits needed for the absolute value
            if isinstance(abs_value, int) or abs_value.is_integer():
                # For integers, we need log2(abs_value) + 1 bits
                # (+1 for the sign if negative)
                abs_value = int(abs_value)
                if abs_value > 0:
                    bits = math.floor(math.log2(abs_value)) + 1
                    if value < 0:
                        bits += 1  # Add sign bit
                    bits_needed.append(bits)
                else:
                    bits_needed.append(1)  # For value 0
            else:
                # For floats, we'll just use 32 or 64 bits (standard sizes)
                bits_needed.append(32)  # Using 32-bit float as default
    
    return bits_needed

def calculate_delta_compression_ratio(data):
    """
    Calculate the compression ratio for delta encoding
    
    Parameters:
    -----------
    data : array-like
        Data to compress
        
    Returns:
    --------
    float
        Compression ratio as a percentage
    """
    try:
        # Convert to numpy array if not already
        arr = np.array(data)
        
        # Original size (assuming 64-bit floats or ints)
        original_size = arr.size * 64
        
        # Apply delta encoding
        first_value, deltas = ca.delta_encode(arr)
        
        # Estimate size of compressed data
        delta_bits = estimate_bits_needed(deltas)
        compressed_size = 64 + sum(delta_bits)  # first value + deltas
        
        # Calculate compression ratio
        ratio = 100 * (1 - compressed_size / original_size)
        return ratio
    except Exception as e:
        print(f"Error calculating delta compression ratio: {e}")
        return 0

def calculate_lzw_compression_ratio(data):
    """
    Calculate the compression ratio for LZW encoding
    
    Parameters:
    -----------
    data : str or convertible to str
        Data to compress
        
    Returns:
    --------
    float
        Compression ratio as a percentage
    """
    try:
        # Convert to string if not already
        if not isinstance(data, str):
            try:
                data_str = str(data)
            except:
                # Try to convert byte data
                if isinstance(data, bytes) or isinstance(data, bytearray):
                    data_str = data.decode('utf-8', errors='replace')
                else:
                    raise ValueError("Could not convert data to string for LZW")
        else:
            data_str = data
        
        # Original size in bits (assuming 8 bits per character)
        original_size = len(data_str) * 8
        
        # Apply LZW compression
        compressed = ca.lzw_compress(data_str)
        
        # Estimate compressed size (each code typically needs log2(dictionary_size) bits)
        dict_size = 256 + len(compressed)  # Initial 256 ASCII codes + added entries
        bits_per_code = math.ceil(math.log2(dict_size))
        compressed_size = len(compressed) * bits_per_code
        
        # Calculate compression ratio
        ratio = 100 * (1 - compressed_size / original_size)
        return ratio
    except Exception as e:
        print(f"Error calculating LZW compression ratio: {e}")
        return 0

def calculate_rle_compression_ratio(data):
    """
    Calculate the compression ratio for Run-Length Encoding
    
    Parameters:
    -----------
    data : sequence
        Data to compress
        
    Returns:
    --------
    float
        Compression ratio as a percentage
    """
    try:
        # Handle different data types
        if isinstance(data, str):
            sequence = list(data)
            bits_per_value = 8  # 8 bits per character
        elif isinstance(data, bytes) or isinstance(data, bytearray):
            sequence = list(data)
            bits_per_value = 8  # 8 bits per byte
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            sequence = data
            # Estimate bits per value
            if len(sequence) > 0:
                sample = sequence[0]
                if isinstance(sample, int):
                    if sample > 0:
                        bits_per_value = math.ceil(math.log2(max(sequence) + 1))
                    else:
                        bits_per_value = math.ceil(math.log2(abs(min(sequence)) + 1)) + 1  # +1 for sign
                elif isinstance(sample, float):
                    bits_per_value = 32  # Assuming 32-bit float
                else:
                    bits_per_value = 8  # Default assumption
            else:
                bits_per_value = 8  # Default for empty sequence
        else:
            raise ValueError("Unsupported data type for RLE")
        
        # Original size
        original_size = len(sequence) * bits_per_value
        
        # Apply RLE
        encoded = ca.rle_encode(sequence)
        
        # Estimate compressed size
        # Each entry in the RLE is (value, count)
        # We need bits_per_value for the value and log2(count) bits for the count
        compressed_size = 0
        for value, count in encoded:
            compressed_size += bits_per_value  # For the value
            compressed_size += math.ceil(math.log2(count + 1))  # For the count
        
        # Calculate compression ratio
        ratio = 100 * (1 - compressed_size / original_size)
        return ratio
    except Exception as e:
        print(f"Error calculating RLE compression ratio: {e}")
        return 0

def calculate_dict_compression_ratio(data):
    """
    Calculate the compression ratio for Dictionary Encoding
    
    Parameters:
    -----------
    data : sequence
        Data to compress
        
    Returns:
    --------
    float
        Compression ratio as a percentage
    """
    try:
        # Handle different data types
        if isinstance(data, str):
            sequence = list(data)
            bits_per_value = 8  # 8 bits per character
        elif isinstance(data, bytes) or isinstance(data, bytearray):
            sequence = list(data)
            bits_per_value = 8  # 8 bits per byte
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            sequence = data
            # Estimate bits per value based on data type
            if len(sequence) > 0:
                sample = sequence[0]
                if isinstance(sample, int):
                    bits_per_value = max(8, math.ceil(math.log2(max(abs(min(sequence)), max(sequence)) + 1)) + 1)
                elif isinstance(sample, float):
                    bits_per_value = 32  # Assuming 32-bit float
                elif isinstance(sample, str):
                    avg_len = sum(len(s) for s in sequence) / len(sequence)
                    bits_per_value = int(avg_len * 8)  # 8 bits per character
                else:
                    bits_per_value = 8  # Default assumption
            else:
                bits_per_value = 8  # Default for empty sequence
        else:
            raise ValueError("Unsupported data type for Dictionary Encoding")
        
        # Original size
        original_size = len(sequence) * bits_per_value
        
        # Apply dictionary encoding
        encoded, value_to_id = ca.dictionary_encode(sequence)
        
        # Count unique values for dictionary size
        unique_values = len(value_to_id)
        
        # Bits needed to represent each dictionary entry
        id_bits = math.ceil(math.log2(unique_values)) if unique_values > 1 else 1
        
        # Dictionary size (each entry requires storing the value)
        dict_size = sum(get_data_size(value) for value in value_to_id.keys())
        
        # Encoded data size (each value is replaced by an ID)
        encoded_size = len(encoded) * id_bits
        
        # Total compressed size is dictionary size + encoded data
        compressed_size = dict_size * 8 + encoded_size  # Convert dict_size to bits
        
        # Calculate compression ratio
        ratio = 100 * (1 - compressed_size / original_size)
        return ratio
    except Exception as e:
        print(f"Error calculating Dictionary compression ratio: {e}")
        return 0

def measure_compression_time(algorithm, data):
    """
    Measure compression and decompression time for an algorithm
    
    Parameters:
    -----------
    algorithm : str
        Name of the algorithm to benchmark
    data : various
        Data to compress
        
    Returns:
    --------
    tuple
        (compression_time, decompression_time) in seconds
    """
    # Compression time
    start_time = time.time()
    try:
        compressed = compress_with_algorithm(algorithm, data)
        compression_time = time.time() - start_time
    except Exception as e:
        print(f"Error compressing with {algorithm}: {e}")
        return float('inf'), float('inf')
    
    # Decompression time
    start_time = time.time()
    try:
        decompress_with_algorithm(algorithm, compressed)
        decompression_time = time.time() - start_time
    except Exception as e:
        print(f"Error decompressing with {algorithm}: {e}")
        return compression_time, float('inf')
    
    return compression_time, decompression_time

def measure_algorithm_performance(algorithm, data, **params):
    """
    Measure comprehensive performance metrics for an algorithm
    
    Parameters:
    -----------
    algorithm : str
        Name of the algorithm to benchmark
    data : various
        Data to compress
    params : dict
        Additional parameters for the algorithm
        
    Returns:
    --------
    tuple
        (compression_ratio, compression_time, decompression_time, memory_usage)
    """
    original_size = get_data_size(data)
    
    # Compression
    start_time = time.time()
    compressed = compress_with_algorithm(algorithm, data)
    compression_time = time.time() - start_time
    
    compressed_size = get_data_size(compressed)
    compression_ratio = 100 * (1 - compressed_size / original_size) if original_size > 0 else 0
    
    # Decompression
    start_time = time.time()
    decompressed = decompress_with_algorithm(algorithm, compressed)
    decompression_time = time.time() - start_time
    
    # Memory usage (estimated)
    memory_usage = compressed_size / (1024 * 1024)  # Convert to MB
    
    return compression_ratio, compression_time, decompression_time, memory_usage

def generate_efficiency_score(compression_ratio, compression_time, decompression_time, data_size):
    """
    Generate an efficiency score balancing compression ratio and speed
    
    Parameters:
    -----------
    compression_ratio : float
        Compression ratio as a percentage
    compression_time : float
        Time taken to compress (seconds)
    decompression_time : float
        Time taken to decompress (seconds)
    data_size : int
        Size of the original data in bytes
        
    Returns:
    --------
    float
        Efficiency score (higher is better)
    """
    # Normalize data size to MB for consistent scaling
    size_mb = data_size / (1024 * 1024)
    
    # Speed in MB/s
    if compression_time > 0:
        compression_speed = size_mb / compression_time
    else:
        compression_speed = float('inf')
    
    if decompression_time > 0:
        decompression_speed = size_mb / decompression_time
    else:
        decompression_speed = float('inf')
    
    # Calculate score components
    ratio_score = compression_ratio / 100  # Normalize to 0-1
    
    # Speed scores (logarithmic scale to handle wide range of speeds)
    # Normalize to approximately 0-1 for typical speeds
    comp_speed_score = min(1, math.log10(compression_speed + 1) / 3)
    decomp_speed_score = min(1, math.log10(decompression_speed + 1) / 3)
    
    # Weighted combination
    # We give higher weight to compression ratio and decompression speed
    # as these are typically more important in distributed systems
    score = (0.5 * ratio_score + 0.2 * comp_speed_score + 0.3 * decomp_speed_score) * 100
    
    return score
