import numpy as np
import heapq
from collections import Counter, defaultdict
import struct
import time
import math

# Helper function to implement Huffman coding for demonstration
def huffman_coding_demo(text):
    """
    Implements Huffman coding for demonstration purposes.
    Returns the compressed size, compression ratio and the codes.
    """
    # Calculate frequency of each character
    freq = {}
    for c in text:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1
    
    # Edge case: empty text or single character text
    if len(freq) <= 1:
        if len(freq) == 0:
            return 0, 0, {}
        char = list(freq.keys())[0]
        return len(text), 0, {char: '0'}
    
    # Create a priority queue to store nodes of the Huffman tree
    heap = [[weight, [char, ""]] for char, weight in freq.items()]
    heapq.heapify(heap)
    
    # Build the Huffman tree
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Extract the Huffman codes
    huff = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    codes = {char: code for char, code in huff}
    
    # Calculate the compressed size
    compressed_size = 0
    for char in text:
        compressed_size += len(codes[char])
    
    # Calculate the compression ratio (how much it's reduced)
    original_size = len(text) * 8  # 8 bits per character in ASCII
    compression_ratio = 100 * (original_size - compressed_size) / original_size
    
    return compressed_size, compression_ratio, codes

# Delta encoding implementation
def delta_encode(data):
    """
    Encodes a sequence using delta encoding
    """
    # Convert to numpy array if not already
    arr = np.array(data)
    
    # Store first value
    first_value = arr[0]
    
    # Calculate deltas
    deltas = arr[1:] - arr[:-1]
    
    return first_value, deltas

def delta_decode(first_value, deltas):
    """
    Decodes a delta-encoded sequence
    """
    # Initialize result with first value
    result = [first_value]
    
    # Reconstruct sequence
    current = first_value
    for delta in deltas:
        current += delta
        result.append(current)
    
    return np.array(result)

# Delta-of-delta encoding for smoother sequences
def delta_of_delta_encode(data):
    """
    Encodes a sequence using delta-of-delta encoding
    """
    # Calculate first-level deltas
    first_value, deltas = delta_encode(data)
    
    # Calculate second-level deltas
    if len(deltas) > 0:
        first_delta = deltas[0]
        second_deltas = np.diff(deltas)
        return first_value, first_delta, second_deltas
    else:
        return first_value, None, np.array([])

def delta_of_delta_decode(first_value, first_delta, second_deltas):
    """
    Decodes a delta-of-delta-encoded sequence
    """
    # Reconstruct first-level deltas
    if first_delta is not None:
        deltas = [first_delta]
        current_delta = first_delta
        
        for second_delta in second_deltas:
            current_delta += second_delta
            deltas.append(current_delta)
        
        # Reconstruct original sequence
        return delta_decode(first_value, deltas)
    else:
        return np.array([first_value])

# LZW compression implementation
def lzw_compress(text):
    """
    Compress text using LZW algorithm
    """
    # Handle empty text
    if not text:
        return []
    
    # Convert text to a list of characters if it's a string
    if isinstance(text, str):
        text = [c for c in text]
    
    # Initialize dictionary with single characters
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256
    result = []
    current_string = text[0]
    
    # Process each character
    for char in text[1:]:
        combined_string = current_string + char
        if combined_string in dictionary:
            current_string = combined_string
        else:
            result.append(dictionary[current_string])
            dictionary[combined_string] = next_code
            next_code += 1
            current_string = char
    
    # Add the last code
    if current_string:
        result.append(dictionary[current_string])
    
    return result

def lzw_decompress(compressed):
    """
    Decompress data compressed with LZW algorithm
    """
    # Handle empty compressed data
    if not compressed:
        return ""
    
    # Initialize dictionary with single characters
    dictionary = {i: chr(i) for i in range(256)}
    next_code = 256
    
    # Get first code and its character
    current_code = compressed[0]
    result = dictionary[current_code]
    current_string = result
    
    # Process remaining codes
    for code in compressed[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = current_string + current_string[0]
        else:
            raise ValueError("Invalid compressed data")
        
        result += entry
        
        # Add to dictionary
        dictionary[next_code] = current_string + entry[0]
        next_code += 1
        current_string = entry
    
    return result

# Run-length encoding (RLE)
def rle_encode(data):
    """
    Compress data using run-length encoding
    """
    if not data:
        return []
    
    encoded = []
    current = data[0]
    count = 1
    
    for d in data[1:]:
        if d == current:
            count += 1
        else:
            encoded.append((current, count))
            current = d
            count = 1
    
    # Add the last run
    encoded.append((current, count))
    
    return encoded

def rle_decode(encoded_data):
    """
    Decompress data compressed with run-length encoding
    """
    if not encoded_data:
        return []
    
    decoded = []
    for value, count in encoded_data:
        decoded.extend([value] * count)
    
    return decoded

# Dictionary encoding for categorical data
def dictionary_encode(data):
    """
    Compress categorical data using dictionary encoding
    """
    unique_values = sorted(set(data))
    value_to_id = {val: i for i, val in enumerate(unique_values)}
    
    encoded = [value_to_id[val] for val in data]
    return encoded, value_to_id

def dictionary_decode(encoded_data, value_to_id):
    """
    Decompress data compressed with dictionary encoding
    """
    id_to_value = {i: val for val, i in value_to_id.items()}
    return [id_to_value[i] for i in encoded_data]

# Frame of Reference (FOR) encoding for numerical data
def for_encode(data, bits_per_value=None):
    """
    Compress numerical data using Frame of Reference encoding
    """
    if not data:
        return None, [], bits_per_value
    
    # Convert to numpy array if not already
    arr = np.array(data)
    
    # Get the reference (minimum value)
    reference = np.min(arr)
    
    # Calculate offsets from reference
    offsets = arr - reference
    
    # Determine bits needed per value if not specified
    if bits_per_value is None:
        max_offset = np.max(offsets)
        if max_offset == 0:
            bits_per_value = 1
        else:
            bits_per_value = math.ceil(math.log2(max_offset + 1))
    
    return reference, offsets, bits_per_value

def for_decode(reference, offsets, bits_per_value=None):
    """
    Decompress data compressed with Frame of Reference encoding
    """
    if reference is None:
        return []
    
    # Reconstruct original values
    return reference + np.array(offsets)

# Benchmark function to measure compression performance
def benchmark_compression(algorithm, data, iterations=5):
    """
    Benchmark a compression algorithm's performance
    """
    # Measure compression time
    comp_start = time.time()
    for _ in range(iterations):
        compressed = algorithm(data)
    comp_time = (time.time() - comp_start) / iterations
    
    # Return algorithm name, compression time, compressed size
    return {
        "algorithm": algorithm.__name__,
        "compression_time": comp_time,
        "original_size": len(data) if hasattr(data, "__len__") else None,
        "compressed_data": compressed
    }
