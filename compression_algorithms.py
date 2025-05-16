import numpy as np
import heapq
import math
from collections import Counter, defaultdict

def huffman_coding_demo(text):
    """
    Implements Huffman coding for demonstration purposes.
    Returns the compressed size, compression ratio and the codes.
    
    Parameters:
    -----------
    text : str
        The text to compress
        
    Returns:
    --------
    compressed_size : int
        Size of the compressed text in bits
    compression_ratio : float
        Compression ratio as a percentage of size reduction
    codes : dict
        Dictionary mapping characters to their Huffman codes
    """
    # Calculate frequency of each character
    freq = {}
    for char in text:
        if char in freq:
            freq[char] += 1
        else:
            freq[char] = 1
            
    # Edge case: empty string or single character
    if len(freq) <= 1:
        if len(freq) == 0:
            return 0, 0, {}
        char = list(freq.keys())[0]
        return len(text), 0.0, {char: '0'}
    
    # Create a priority queue (min-heap) with tuples of (frequency, character)
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
    
    # Extract the codes
    codes = {char: code for char, code in sorted(heap[0][1:])}
    
    # Calculate compressed size in bits
    compressed_size = sum(freq[char] * len(codes[char]) for char in freq)
    
    # Calculate original size (assuming 8 bits per char in ASCII)
    original_size = len(text) * 8
    
    # Calculate compression ratio
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    return compressed_size, compression_ratio, codes

def delta_encode(data):
    """
    Encodes a sequence using delta encoding
    
    Parameters:
    -----------
    data : array-like
        The sequence to encode
        
    Returns:
    --------
    first_value : number
        The first value of the sequence
    deltas : list
        List of differences between consecutive values
    """
    # Convert to numpy array if not already
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Store the first value
    first_value = data[0]
    
    # Calculate deltas (differences between consecutive values)
    deltas = np.diff(data)
    
    return first_value, deltas.tolist()

def delta_decode(first_value, deltas):
    """
    Decodes a delta-encoded sequence
    
    Parameters:
    -----------
    first_value : number
        The first value of the original sequence
    deltas : list
        List of differences between consecutive values
        
    Returns:
    --------
    data : list
        The reconstructed sequence
    """
    # Start with the first value
    data = [first_value]
    
    # Reconstruct the sequence using the deltas
    current = first_value
    for delta in deltas:
        current += delta
        data.append(current)
    
    return data

def delta_of_delta_encode(data):
    """
    Encodes a sequence using delta-of-delta encoding
    
    Parameters:
    -----------
    data : array-like
        The sequence to encode
        
    Returns:
    --------
    first_value : number
        The first value of the sequence
    first_delta : number
        The first delta (difference between first two values)
    second_deltas : list
        List of differences between consecutive deltas
    """
    # Convert to numpy array if not already
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Need at least 3 values for delta-of-delta
    if len(data) < 3:
        raise ValueError("Delta-of-delta encoding requires at least 3 values")
    
    # Store first value and first delta
    first_value = data[0]
    first_delta = data[1] - data[0]
    
    # Calculate first-order deltas
    deltas = np.diff(data)
    
    # Calculate second-order deltas (differences of differences)
    second_deltas = np.diff(deltas)
    
    return first_value, first_delta, second_deltas.tolist()

def delta_of_delta_decode(first_value, first_delta, second_deltas):
    """
    Decodes a delta-of-delta-encoded sequence
    
    Parameters:
    -----------
    first_value : number
        The first value of the original sequence
    first_delta : number
        The first delta value
    second_deltas : list
        List of second-order deltas
        
    Returns:
    --------
    data : list
        The reconstructed sequence
    """
    # Start with the first value
    data = [first_value]
    
    # Add the second value using first delta
    data.append(first_value + first_delta)
    
    # Reconstruct the deltas
    deltas = [first_delta]
    for second_delta in second_deltas:
        deltas.append(deltas[-1] + second_delta)
    
    # Reconstruct the sequence
    for delta in deltas[1:]:  # Skip the first delta which we've already used
        data.append(data[-1] + delta)
    
    return data

def lzw_compress(text):
    """
    Compress text using LZW algorithm
    
    Parameters:
    -----------
    text : str or list
        The text or sequence to compress
        
    Returns:
    --------
    compressed : list
        List of codes representing the compressed data
    """
    # Convert to string if it's not already
    if not isinstance(text, str):
        # Try to convert list/array to string
        try:
            text = ''.join(map(str, text))
        except:
            # If that fails, just convert each element to string and join with a separator
            text = ','.join(map(str, text))
    
    # Build the dictionary - start with single-character strings
    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
    
    result = []
    w = ""
    for c in text:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
    
    # Output the code for w
    if w:
        result.append(dictionary[w])
    
    return result

def lzw_decompress(compressed):
    """
    Decompress data compressed with LZW algorithm
    
    Parameters:
    -----------
    compressed : list
        List of codes representing the compressed data
        
    Returns:
    --------
    text : str
        The decompressed text
    """
    # Build the dictionary
    dict_size = 256
    dictionary = {i: chr(i) for i in range(dict_size)}
    
    if not compressed:
        return ""
    
    # First code is always a single character
    result = [dictionary[compressed[0]]]
    w = result[0]
    
    for code in compressed[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == dict_size:
            entry = w + w[0]
        else:
            raise ValueError("Invalid compressed data")
        
        result.append(entry)
        
        # Add new sequence to the dictionary
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        
        w = entry
    
    return ''.join(result)

def rle_encode(data):
    """
    Compress data using run-length encoding
    
    Parameters:
    -----------
    data : list, str, or array-like
        The data to compress
        
    Returns:
    --------
    encoded : list
        List of tuples (value, count) representing runs
    """
    if not data:
        return []
    
    encoded = []
    count = 1
    current = data[0]
    
    for i in range(1, len(data)):
        if data[i] == current:
            count += 1
        else:
            encoded.append((current, count))
            current = data[i]
            count = 1
    
    # Don't forget the last run
    encoded.append((current, count))
    
    return encoded

def rle_decode(encoded_data):
    """
    Decompress data compressed with run-length encoding
    
    Parameters:
    -----------
    encoded_data : list
        List of tuples (value, count) representing runs
        
    Returns:
    --------
    decoded : list
        The decompressed data
    """
    decoded = []
    
    for value, count in encoded_data:
        decoded.extend([value] * count)
    
    return decoded

def dictionary_encode(data):
    """
    Compress categorical data using dictionary encoding
    
    Parameters:
    -----------
    data : list, str, or array-like
        The data to compress
        
    Returns:
    --------
    encoded : list
        List of integer IDs representing the original values
    value_to_id : dict
        Mapping from original values to integer IDs
    """
    # Find all unique values and assign IDs
    unique_values = sorted(set(data), key=lambda x: data.index(x))
    value_to_id = {value: i for i, value in enumerate(unique_values)}
    
    # Encode the data
    encoded = [value_to_id[value] for value in data]
    
    return encoded, value_to_id

def dictionary_decode(encoded_data, value_to_id):
    """
    Decompress data compressed with dictionary encoding
    
    Parameters:
    -----------
    encoded_data : list
        List of integer IDs
    value_to_id : dict
        Mapping from original values to integer IDs
        
    Returns:
    --------
    decoded : list
        The decompressed data
    """
    # Create reverse mapping
    id_to_value = {i: value for value, i in value_to_id.items()}
    
    # Decode the data
    decoded = [id_to_value[id_] for id_ in encoded_data]
    
    return decoded

def for_encode(data, bits_per_value=None):
    """
    Compress numerical data using Frame of Reference encoding
    
    Parameters:
    -----------
    data : array-like
        The numerical data to compress
    bits_per_value : int, optional
        Number of bits to use for each value. If None, it's calculated.
        
    Returns:
    --------
    reference : number
        The reference value (minimum value in the data)
    offsets : list
        List of offsets from the reference value
    bits_per_value : int
        Number of bits used to encode each offset
    """
    # Convert to numpy array if not already
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Find the reference value (minimum)
    reference = np.min(data)
    
    # Calculate offsets from the reference
    offsets = (data - reference).astype(int)
    
    # Determine bits needed for the largest offset
    if bits_per_value is None:
        max_offset = np.max(offsets)
        bits_per_value = max(1, math.ceil(math.log2(max_offset + 1)))
    
    return reference, offsets.tolist(), bits_per_value

def for_decode(reference, offsets, bits_per_value=None):
    """
    Decompress data compressed with Frame of Reference encoding
    
    Parameters:
    -----------
    reference : number
        The reference value
    offsets : list
        List of offsets from the reference value
    bits_per_value : int, optional
        Number of bits used for each offset (not needed for decoding)
        
    Returns:
    --------
    decoded : list
        The decompressed data
    """
    # Add the reference value to each offset
    decoded = [reference + offset for offset in offsets]
    
    return decoded

def benchmark_compression(algorithm, data, iterations=5):
    """
    Benchmark a compression algorithm's performance
    
    Parameters:
    -----------
    algorithm : str
        Name of the algorithm to benchmark ('huffman', 'delta', 'lzw', 'rle', 'dictionary', 'for')
    data : object
        The data to compress
    iterations : int
        Number of iterations for timing
        
    Returns:
    --------
    dict
        Dictionary with benchmark results
    """
    import time
    
    results = {
        'algorithm': algorithm,
        'original_size': None,
        'compressed_size': None,
        'compression_ratio': None,
        'compression_time': None,
        'decompression_time': None
    }
    
    # Calculate original size
    if isinstance(data, str):
        results['original_size'] = len(data) * 8  # 8 bits per character
    elif isinstance(data, bytes):
        results['original_size'] = len(data) * 8  # 8 bits per byte
    elif isinstance(data, (list, np.ndarray)):
        if isinstance(data, np.ndarray):
            results['original_size'] = data.nbytes * 8
        else:
            results['original_size'] = len(data) * 8  # Approximation
    
    # Measure compression time
    start_time = time.time()
    for _ in range(iterations):
        if algorithm == 'huffman':
            compressed_size, compression_ratio, codes = huffman_coding_demo(data)
            compressed = None  # We don't store the actual compressed data for Huffman demo
        elif algorithm == 'delta':
            first_value, deltas = delta_encode(data)
            compressed = (first_value, deltas)
        elif algorithm == 'delta_of_delta':
            first_value, first_delta, second_deltas = delta_of_delta_encode(data)
            compressed = (first_value, first_delta, second_deltas)
        elif algorithm == 'lzw':
            compressed = lzw_compress(data)
        elif algorithm == 'rle':
            compressed = rle_encode(data)
        elif algorithm == 'dictionary':
            encoded, value_to_id = dictionary_encode(data)
            compressed = (encoded, value_to_id)
        elif algorithm == 'for':
            reference, offsets, bits_per_value = for_encode(data)
            compressed = (reference, offsets, bits_per_value)
    
    results['compression_time'] = (time.time() - start_time) / iterations
    
    # Calculate compressed size and ratio
    if algorithm == 'huffman':
        results['compressed_size'] = compressed_size
        results['compression_ratio'] = compression_ratio
    elif algorithm == 'delta':
        # Estimate size: first value (64 bits) + deltas (variable bits)
        first_value, deltas = compressed
        delta_bits = sum(max(1, math.ceil(math.log2(abs(d) + 1)) + 1) for d in deltas)
        results['compressed_size'] = 64 + delta_bits
        results['compression_ratio'] = (1 - results['compressed_size'] / results['original_size']) * 100
    elif algorithm == 'delta_of_delta':
        # Estimate size: first value (64 bits) + first delta (64 bits) + second deltas (variable bits)
        first_value, first_delta, second_deltas = compressed
        delta_bits = sum(max(1, math.ceil(math.log2(abs(d) + 1)) + 1) for d in second_deltas)
        results['compressed_size'] = 64 + 64 + delta_bits
        results['compression_ratio'] = (1 - results['compressed_size'] / results['original_size']) * 100
    elif algorithm == 'lzw':
        # Estimate size: each code is log2(dictionary size) bits
        dict_size = 256 + len(compressed)
        bits_per_code = max(8, math.ceil(math.log2(dict_size)))
        results['compressed_size'] = len(compressed) * bits_per_code
        results['compression_ratio'] = (1 - results['compressed_size'] / results['original_size']) * 100
    elif algorithm == 'rle':
        # Estimate size: each run is (value, count)
        results['compressed_size'] = len(compressed) * 16  # Approximation: 8 bits for value, 8 bits for count
        results['compression_ratio'] = (1 - results['compressed_size'] / results['original_size']) * 100
    elif algorithm == 'dictionary':
        # Estimate size: dictionary + encoded values
        encoded, value_to_id = compressed
        dict_size_bits = len(value_to_id) * 16  # Approximation: each entry is 16 bits
        id_bits = max(1, math.ceil(math.log2(len(value_to_id))))
        encoded_bits = len(encoded) * id_bits
        results['compressed_size'] = dict_size_bits + encoded_bits
        results['compression_ratio'] = (1 - results['compressed_size'] / results['original_size']) * 100
    elif algorithm == 'for':
        # Estimate size: reference value + offsets
        reference, offsets, bits_per_value = compressed
        results['compressed_size'] = 64 + len(offsets) * bits_per_value
        results['compression_ratio'] = (1 - results['compressed_size'] / results['original_size']) * 100
    
    # Measure decompression time
    start_time = time.time()
    for _ in range(iterations):
        if algorithm == 'huffman':
            # Skip actual decompression for Huffman demo
            pass
        elif algorithm == 'delta':
            first_value, deltas = compressed
            decompressed = delta_decode(first_value, deltas)
        elif algorithm == 'delta_of_delta':
            first_value, first_delta, second_deltas = compressed
            decompressed = delta_of_delta_decode(first_value, first_delta, second_deltas)
        elif algorithm == 'lzw':
            decompressed = lzw_decompress(compressed)
        elif algorithm == 'rle':
            decompressed = rle_decode(compressed)
        elif algorithm == 'dictionary':
            encoded, value_to_id = compressed
            decompressed = dictionary_decode(encoded, value_to_id)
        elif algorithm == 'for':
            reference, offsets, bits_per_value = compressed
            decompressed = for_decode(reference, offsets, bits_per_value)
    
    results['decompression_time'] = (time.time() - start_time) / iterations
    
    return results