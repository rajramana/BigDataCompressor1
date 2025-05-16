import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import compression_algorithms as ca
import data_generator as dg
import visualization as viz
import comparative_analysis as analysis
import utils

# Page configuration
st.set_page_config(
    page_title="Compression Techniques for Big Data in Distributed Systems",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and introduction
def introduction():
    st.title("Compression Techniques for Big Data in Distributed Systems")
    
    st.markdown("""
    ## Introduction
    
    In the era of big data, the efficient storage and transmission of large datasets is crucial. 
    Compression techniques play a vital role in reducing the storage footprint and network bandwidth 
    requirements in distributed systems. This research presentation explores various compression 
    techniques specifically designed for big data applications in distributed environments.
    
    ### Research Objectives
    
    1. Analyze existing compression techniques for big data
    2. Implement and evaluate selected compression algorithms
    3. Compare performance metrics for different techniques
    4. Provide recommendations for specific use cases
    
    ### Why Compression in Distributed Systems?
    
    - **Storage Efficiency**: Reduces storage costs and requirements
    - **Network Bandwidth**: Decreases data transfer time between nodes
    - **Processing Speed**: Can improve query performance and reduce I/O bottlenecks
    - **Energy Efficiency**: Reduces overall power consumption in data centers
    """)
    
    st.image("https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg", width=50)

# Theoretical background
def theory_background():
    st.header("Theoretical Background")
    
    st.markdown("""
    ### Compression Fundamentals
    
    Data compression works by identifying and eliminating redundancy in data. There are two primary categories:
    
    1. **Lossless Compression**
       - Preserves all original data
       - Suitable for text, databases, and program files
       - Examples: Huffman coding, LZ77, LZ78, Deflate, LZW
    
    2. **Lossy Compression**
       - Discards some data deemed less important
       - Higher compression ratios
       - Used for multimedia data (images, audio, video)
       - Examples: JPEG, MP3, MPEG
    
    ### Compression in Distributed Systems
    
    In distributed environments, compression strategies must consider:
    
    - **Network Topology**: Affects when and where to compress data
    - **Processing Power Distribution**: Determines computational budget for compression
    - **Workload Characteristics**: Helps choose optimal compression techniques
    - **Data Access Patterns**: Influences compression format and chunking strategies
    
    ### Common Big Data Compression Techniques
    """)
    
    # Create tabs for different compression techniques
    techniques = st.tabs(["Dictionary-based", "Entropy-based", "Delta Encoding", "Columnar Compression", "Frame of Reference"])
    
    with techniques[0]:
        st.markdown("""
        #### Dictionary-based Compression
        
        These algorithms replace recurring patterns with references to a dictionary.
        
        - **LZ77**: Uses a sliding window to find repeated sequences
        - **LZ78**: Builds a dictionary explicitly during compression
        - **LZW**: Adaptive dictionary technique used in GIF and TIFF formats
        - **Snappy**: Google's compression algorithm optimized for speed
        - **LZ4**: Extremely fast compression/decompression
        
        **Best for**: Log files, text data, JSON/XML data
        """)
    
    with techniques[1]:
        st.markdown("""
        #### Entropy-based Compression
        
        These methods encode data based on symbol frequency.
        
        - **Huffman Coding**: Assigns variable-length codes based on frequency
        - **Arithmetic Coding**: Encodes entire messages into a single number
        - **Range Coding**: Variation of arithmetic coding with better implementation properties
        
        **Best for**: Text with skewed frequency distributions, preprocessing for other algorithms
        """)
    
    with techniques[2]:
        st.markdown("""
        #### Delta Encoding
        
        Stores differences between sequential data values rather than the values themselves.
        
        - **Simple Delta**: Stores differences between consecutive values
        - **Delta-of-Delta**: Computes differences between deltas for smoother series
        - **XOR-based Delta**: Uses bitwise XOR for efficient differences
        
        **Best for**: Time series data, sensor data, financial data, sorted database columns
        """)
    
    with techniques[3]:
        st.markdown("""
        #### Columnar Compression
        
        Specifically designed for column-oriented data stores.
        
        - **Run-Length Encoding (RLE)**: Compresses runs of repeated values
        - **Dictionary Encoding**: Maps values to integer IDs for columns with limited cardinality
        - **Bitmap Encoding**: Uses bit vectors to represent data
        
        **Best for**: Data warehouses, analytical databases, tables with many columns
        """)
    
    with techniques[4]:
        st.markdown("""
        #### Frame of Reference (FOR)
        
        Stores differences from a reference value using minimal bits.
        
        - **FOR**: Uses minimum value as reference
        - **PFOR (Patched FOR)**: Handles outliers separately
        - **PFOR-DELTA**: Combines delta encoding with PFOR
        
        **Best for**: Numerical data with narrow ranges, sorted data
        """)

# Algorithm implementation section
def algorithm_implementation():
    st.header("Algorithm Implementation")
    
    st.markdown("""
    This section demonstrates the implementation and functionality of key compression algorithms.
    We'll focus on three primary algorithms that are particularly relevant for big data in distributed systems:
    
    1. **Huffman Coding**: An entropy-based algorithm optimal for compressing data with skewed frequency distributions
    2. **Delta Encoding**: Excellent for time-series and sequential numerical data
    3. **Dictionary-based LZW**: Effective for text and repeated pattern compression
    
    Select an algorithm to see its implementation details, pseudocode, and a live demonstration:
    """)
    
    selected_algorithm = st.selectbox(
        "Choose a compression algorithm",
        ["Huffman Coding", "Delta Encoding", "Dictionary-based LZW"]
    )
    
    if selected_algorithm == "Huffman Coding":
        st.subheader("Huffman Coding Implementation")
        
        st.markdown("""
        ### Algorithm Overview
        
        Huffman coding is an entropy encoding algorithm that assigns variable-length codes to input characters based on their frequencies. The most frequent characters get the shortest codes.
        
        ### Pseudocode
        ```
        function HuffmanCoding(text):
            1. Calculate frequency of each character in text
            2. Build a min-heap (priority queue) of nodes, each containing:
               - character
               - frequency
               - left child (initially None)
               - right child (initially None)
            3. While size of heap > 1:
               a. Extract two nodes with lowest frequency (min1, min2)
               b. Create a new internal node with:
                  - character: None
                  - frequency: min1.freq + min2.freq
                  - left child: min1
                  - right child: min2
               c. Insert this node back into the heap
            4. The remaining node is the root of the Huffman tree
            5. Traverse the tree to generate codes for each character
            6. Encode the text using these codes
            
        function decode(encoded_text, huffman_tree):
            1. Start at the root of the Huffman tree
            2. For each bit in encoded_text:
               a. If bit is 0, go to left child
               b. If bit is 1, go to right child
               c. If at a leaf node (with a character):
                  i. Append character to decoded text
                  ii. Return to the root
            3. Return decoded text
        ```
        
        ### Implementation Highlights
        
        The Python implementation uses heapq for the priority queue, binary tree nodes for the Huffman tree, and bitarray for efficient bit manipulation.
        """)
        
        st.code("""
# A node in the Huffman Tree
class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq      # frequency of symbol
        self.symbol = symbol  # symbol name (character)
        self.left = left      # left node
        self.right = right    # right node
        self.huff = ''        # tree direction (0/1)
        
    def __lt__(self, other):
        return self.freq < other.freq
        """, language="python")
        
        # Show a live demo
        st.subheader("Live Demonstration")
        
        input_text = st.text_area("Enter text to compress with Huffman coding:", 
                                 "This is a sample text to demonstrate Huffman coding compression. "
                                 "Repeat patterns and frequently occurring characters will be compressed efficiently.")
        
        if st.button("Compress using Huffman Coding"):
            if input_text:
                with st.spinner("Compressing..."):
                    # Get frequency of characters
                    freq = {}
                    for c in input_text:
                        if c in freq:
                            freq[c] += 1
                        else:
                            freq[c] = 1
                    
                    # Display character frequencies
                    freq_df = pd.DataFrame(list(freq.items()), columns=['Character', 'Frequency'])
                    freq_df = freq_df.sort_values(by='Frequency', ascending=False)
                    
                    st.write("Character Frequencies:")
                    st.dataframe(freq_df)
                    
                    # Perform compression
                    original_size = len(input_text) * 8  # 8 bits per character
                    compressed_size, compression_ratio, codes = ca.huffman_coding_demo(input_text)
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original Size (bits)", original_size)
                    col2.metric("Compressed Size (bits)", compressed_size)
                    col3.metric("Compression Ratio", f"{compression_ratio:.2f}%")
                    
                    # Display Huffman codes for each character
                    codes_df = pd.DataFrame(list(codes.items()), columns=['Character', 'Code'])
                    codes_df = codes_df.sort_values(by='Code')
                    
                    st.write("Huffman Codes:")
                    st.dataframe(codes_df)
                    
                    # Visualize the compression
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.bar(['Original', 'Compressed'], [original_size, compressed_size])
                    ax.set_ylabel('Size (bits)')
                    ax.set_title('Compression Results')
                    st.pyplot(fig)
            else:
                st.warning("Please enter some text to compress.")
    
    elif selected_algorithm == "Delta Encoding":
        st.subheader("Delta Encoding Implementation")
        
        st.markdown("""
        ### Algorithm Overview
        
        Delta encoding stores differences between consecutive data values rather than the original values. This is highly efficient for time series data or any data where consecutive values are often similar.
        
        ### Pseudocode
        ```
        function DeltaEncode(sequence):
            1. Store the first value as is
            2. For each subsequent value:
               a. Calculate delta = current_value - previous_value
               b. Store delta
            3. Return first_value and list of deltas
            
        function DeltaDecode(first_value, deltas):
            1. Initialize result array with first_value
            2. Initialize current = first_value
            3. For each delta in deltas:
               a. current = current + delta
               b. Append current to result array
            4. Return result array
        ```
        
        ### Implementation Highlights
        
        The Python implementation uses NumPy for efficient array operations and supports both simple delta encoding and delta-of-delta encoding for improved compression of smooth sequences.
        """)
        
        st.code("""
def delta_encode(data):
    '''
    Encode a sequence using delta encoding
    '''
    # Convert to numpy array if not already
    arr = np.array(data)
    
    # Store first value
    first_value = arr[0]
    
    # Calculate deltas
    deltas = arr[1:] - arr[:-1]
    
    return first_value, deltas

def delta_decode(first_value, deltas):
    '''
    Decode a delta-encoded sequence
    '''
    # Initialize result with first value
    result = [first_value]
    
    # Reconstruct sequence
    current = first_value
    for delta in deltas:
        current += delta
        result.append(current)
    
    return np.array(result)
        """, language="python")
        
        # Show a live demo
        st.subheader("Live Demonstration")
        
        data_type = st.selectbox(
            "Choose a data type for demonstration:",
            ["Time Series", "Sensor Readings", "Stock Prices"]
        )
        
        # Generate data based on selection
        if data_type == "Time Series":
            data = dg.generate_time_series(100)
            title = "Time Series Data"
            y_label = "Value"
        elif data_type == "Sensor Readings":
            data = dg.generate_sensor_data(100)
            title = "Sensor Readings"
            y_label = "Temperature (°C)"
        else:  # Stock Prices
            data = dg.generate_stock_prices(100)
            title = "Stock Prices"
            y_label = "Price ($)"
        
        if st.button("Apply Delta Encoding"):
            with st.spinner("Processing..."):
                # Encode the data
                first_value, deltas = ca.delta_encode(data)
                
                # Calculate storage requirements
                original_size = len(data) * 64  # 64-bit floats
                
                # Estimate compressed size (first value at full precision + deltas)
                # In practice, deltas often need fewer bits due to smaller range
                delta_bits = utils.estimate_bits_needed(deltas)
                compressed_size = 64 + sum(delta_bits)  # first value + all deltas
                
                compression_ratio = 100 * (1 - compressed_size / original_size)
                
                # Display the original and encoded data
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Size (bits)", original_size)
                col2.metric("Compressed Size (bits)", int(compressed_size))
                col3.metric("Compression Ratio", f"{compression_ratio:.2f}%")
                
                # Visualize the original data and deltas
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                
                # Plot original data
                ax1.plot(data)
                ax1.set_title(f"Original {title}")
                ax1.set_ylabel(y_label)
                
                # Plot deltas
                ax2.plot(deltas, color='orange')
                ax2.set_title("Delta Values")
                ax2.set_xlabel("Index")
                ax2.set_ylabel("Delta")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display bit requirements
                st.subheader("Bit Requirements for Storage")
                st.write("Original data requires 64 bits per value (using standard double precision).")
                
                bit_counts = {}
                for bits_needed in delta_bits:
                    if bits_needed in bit_counts:
                        bit_counts[bits_needed] += 1
                    else:
                        bit_counts[bits_needed] = 1
                
                bit_df = pd.DataFrame({
                    'Bits Required': list(bit_counts.keys()),
                    'Count': list(bit_counts.values())
                })
                bit_df = bit_df.sort_values('Bits Required')
                
                st.write("Number of delta values requiring each bit length:")
                st.dataframe(bit_df)
    
    elif selected_algorithm == "Dictionary-based LZW":
        st.subheader("LZW Compression Implementation")
        
        st.markdown("""
        ### Algorithm Overview
        
        LZW (Lempel-Ziv-Welch) is a dictionary-based compression algorithm that builds a dictionary of strings dynamically as it encodes the data. It is especially effective for text with recurring patterns.
        
        ### Pseudocode
        ```
        function LZW_Compress(text):
            1. Initialize dictionary with all possible single characters
            2. Initialize current_string to first character of text
            3. For each character c in the rest of the text:
               a. If current_string + c is in dictionary:
                  i. current_string = current_string + c
               b. Else:
                  i. Output the code for current_string
                  ii. Add current_string + c to the dictionary
                  iii. current_string = c
            4. Output the code for current_string
            
        function LZW_Decompress(codes):
            1. Initialize dictionary with all possible single characters
            2. Let first_code be the first code in codes
            3. Output the character for first_code
            4. Initialize current_string to character for first_code
            5. For each remaining code in codes:
               a. If code is in dictionary:
                  i. entry = dictionary[code]
               b. Else:
                  i. entry = current_string + current_string[0]
               c. Output entry
               d. Add current_string + entry[0] to dictionary
               e. current_string = entry
        ```
        
        ### Implementation Highlights
        
        Our Python implementation uses dictionaries for mapping between strings and codes. It handles different data types and provides efficient compression for text with repetitive patterns.
        """)
        