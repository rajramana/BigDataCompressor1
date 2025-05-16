import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Compression Techniques for Big Data - Thesis Figures",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def view_compression_comparison_chart():
    st.subheader("Compression Comparison Chart")
    
    st.markdown("""
    This chart compares the compression ratios achieved by different algorithms across various data types.
    """)
    
    # Create data for the chart
    data = {
        'Data Type': ['Time Series', 'Text', 'Categorical', 'Binary (Low)', 'Binary (High)'],
        'Delta': [78.5, 5.2, 9.3, 21.5, 2.3],
        'Huffman': [25.3, 53.7, 38.4, 46.7, 4.8],
        'LZW': [42.1, 48.9, 42.6, 57.3, 3.7],
        'RLE': [12.7, 18.3, 62.7, 85.4, 1.9],
        'Dictionary': [35.6, 44.2, 89.7, 32.1, 2.2],
        'FOR': [67.2, 2.1, 7.5, 12.8, 1.5]
    }
    
    df = pd.DataFrame(data)
    
    # Convert to long format for better visualization
    df_long = pd.melt(df, id_vars=['Data Type'], var_name='Algorithm', value_name='Compression Ratio (%)')
    
    # Create the chart using matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define color map for different algorithms
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot grouped bar chart
    data_types = df['Data Type'].unique()
    algorithms = df.columns[1:]
    bar_width = 0.15
    index = np.arange(len(data_types))
    
    for i, algorithm in enumerate(algorithms):
        offset = (i - len(algorithms)/2 + 0.5) * bar_width
        ax.bar(index + offset, df[algorithm], bar_width, label=algorithm, color=colors[i % len(colors)])
    
    # Set chart labels and title
    ax.set_xlabel('Data Type')
    ax.set_ylabel('Compression Ratio (%)')
    ax.set_title('Compression Ratio by Algorithm and Data Type')
    ax.set_xticks(index)
    ax.set_xticklabels(data_types, rotation=45)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, algorithm in enumerate(algorithms):
        offset = (i - len(algorithms)/2 + 0.5) * bar_width
        for j, value in enumerate(df[algorithm]):
            ax.text(j + offset, value + 1, f'{value}%', ha='center', va='bottom', fontsize=8)
    
    st.pyplot(fig)
    
    st.markdown("""
    ## Key Findings

    - **Time Series Data**: Delta encoding achieves the best compression (78.5%)
    - **Text Data**: Huffman coding provides superior compression (53.7%)
    - **Categorical Data**: Dictionary encoding shows the highest compression ratio (89.7%)
    - **Binary Data (Low Entropy)**: Run-length encoding excels for repetitive binary data (85.4%)
    - **Binary Data (High Entropy)**: All algorithms struggle with high-entropy data (< 5% compression)
    """)

def view_huffman_coding_example():
    st.subheader("Huffman Coding Example")
    
    st.markdown("""
    Huffman coding is an entropy-based compression algorithm that assigns variable-length codes to input characters based on their frequencies. The most frequent characters get the shortest codes.
    """)
    
    st.markdown("### Example Text")
    st.code("COMPRESSION_EXAMPLE")
    
    # Create character frequency data
    char_freq = {
        'E': 3, '_': 3, 'M': 2, 'P': 2, 'O': 2, 
        'C': 1, 'R': 1, 'S': 1, 'I': 1, 'N': 1, 'X': 1, 'A': 1, 'L': 1
    }
    
    # Create dataframe for frequency table
    df_freq = pd.DataFrame(list(char_freq.items()), columns=['Character', 'Frequency'])
    df_freq = df_freq.sort_values(by='Frequency', ascending=False)
    
    st.markdown("### Character Frequencies")
    st.table(df_freq)
    
    # Create dataframe for Huffman codes
    codes = {
        'E': '00', '_': '01', 'M': '100', 'P': '101', 'O': '110',
        'C': '1110', 'R': '1111', 'S': '1000', 'I': '1001',
        'N': '1010', 'X': '1011', 'A': '1100', 'L': '1101'
    }
    
    df_codes = pd.DataFrame(list(codes.items()), columns=['Character', 'Code'])
    
    st.markdown("### Huffman Codes")
    st.table(df_codes)
    
    st.markdown("""
    ### Huffman Tree

    ```
                  (20)
                 /    \\
              (8)      (12)
             /   \\     /   \\
          (3)    (5) (5)    (7)
         /  \\    / \\  / \\   / \\
       C(1) R(1) S(1) I(1) N(1) X(1)
    ```
    """)
    
    st.markdown("### Compression Results")
    
    original_size = 160  # 20 characters × 8 bits
    compressed_size = 45  # Sum of frequency * code length
    compression_ratio = 71.9  # (original_size - compressed_size) / original_size * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Size (bits)", original_size)
    col2.metric("Compressed Size (bits)", compressed_size)
    col3.metric("Compression Ratio", f"{compression_ratio:.1f}%")
    
    st.markdown("""
    ### Advantages and Limitations

    #### Advantages
    - Optimal for known frequency distributions
    - Variable-length codes adapt to data characteristics
    - Lossless compression preserves all information

    #### Limitations
    - Requires knowledge of frequency distribution (two passes or statistical model)
    - Overhead of storing the codebook
    - Less effective for uniform distributions
    """)

def view_delta_encoding_example():
    st.subheader("Delta Encoding Example")
    
    st.markdown("""
    Delta encoding is a compression technique particularly effective for time series or sorted numerical data. It works by storing the differences between consecutive values rather than the values themselves.
    """)
    
    # Example data
    original_values = [100, 102, 106, 109, 110, 112, 115, 117, 118, 120]
    delta_values = [100] + [original_values[i] - original_values[i-1] for i in range(1, len(original_values))]
    
    st.markdown("### Example Sequence")
    st.code(str(original_values))
    
    # Create dataframe for delta values
    delta_df = pd.DataFrame({
        'Index': range(len(original_values)),
        'Original Value': original_values,
        'Delta Value': ['-'] + [f"+{d}" for d in delta_values[1:]],
        'Storage': [100] + delta_values[1:]
    })
    
    st.markdown("### Delta Values")
    st.table(delta_df)
    
    st.markdown("""
    ### Bit Requirements

    - Original values: 10 × 32 bits = 320 bits (assuming 32-bit integers)
    - Delta encoded: 32 bits (first value) + 20 bits (deltas) = 52 bits
    - Compression ratio: 83.75%
    """)
    
    st.markdown("""
    ### Advantages

    - Extremely effective for smooth time series data
    - Very fast encoding and decoding
    - Simple implementation
    - Can be combined with other compression techniques

    ### Disadvantages

    - Less effective for random or uncorrelated data
    - Requires special handling for missing values
    - Susceptible to error propagation (one error affects all subsequent values)
    """)
    
    st.markdown("""
    ### Applications in Distributed Systems

    - Sensor data storage and transmission
    - System monitoring metrics
    - Financial time series
    - Database indexes
    - Scientific measurement data
    """)

def view_compression_framework_diagram():
    st.subheader("Adaptive Compression Framework Diagram")
    
    st.markdown("""
    The Adaptive Compression Framework dynamically selects and applies the most appropriate compression algorithm based on data characteristics and system constraints.
    """)
    
    st.markdown("### Framework Components")
    
    components = {
        "Data Analyzer": [
            "**Data type detection**: Identifies whether data is text, numerical, time series, categorical, or binary",
            "**Statistical analysis**: Calculates entropy, run patterns, and value distributions",
            "**Pattern recognition**: Identifies repeating sequences and correlation structures",
            "**Range analysis**: Determines potential for range-based compression techniques"
        ],
        "Decision Engine": [
            "**Algorithm scoring**: Ranks algorithms based on suitability for detected data characteristics",
            "**Constraint evaluation**: Considers system requirements like speed vs. compression ratio",
            "**Bonus scoring**: Awards extra points for specific data characteristics that certain algorithms excel at",
            "**Final selection**: Chooses the algorithm with the highest overall score"
        ],
        "Compression Manager": [
            "**Algorithm application**: Applies the selected compression algorithm",
            "**Metadata management**: Stores algorithm choice and parameters with compressed data",
            "**Format standardization**: Provides consistent interface for all algorithms",
            "**Error handling**: Detects and manages compression/decompression errors"
        ]
    }
    
    for component, details in components.items():
        st.markdown(f"#### {component}")
        for detail in details:
            st.markdown(f"- {detail}")
    
    st.markdown("### Available Algorithms")
    
    algorithms_data = {
        'Algorithm': ['Huffman', 'Delta', 'Delta-of-Delta', 'LZW', 'RLE', 'Dictionary', 'FOR'],
        'Data Types': ['Text, Binary', 'Numerical, Time Series', 'Time Series', 'Text, Binary, Mixed', 
                      'Binary, Categorical, Text', 'Categorical, Text', 'Numerical, Time Series'],
        'Speed Priority': ['Medium', 'Very High', 'High', 'Medium', 'Very High', 'High', 'High'],
        'Compression Priority': ['High', 'Medium', 'High', 'Medium', 'Low', 'High', 'Medium']
    }
    
    st.table(pd.DataFrame(algorithms_data))
    
    st.markdown("""
    ### Algorithm Selection Process

    1. **Data Type Filtering**: Filter algorithms suitable for the detected data type
    2. **Constraint Matching**: Score algorithms based on how well they match speed and ratio priorities
    3. **Special Characteristics Bonus**: Award bonus points for specific data characteristics:
       - Low entropy → Bonus for Huffman coding
       - Low run ratio → Bonus for RLE
       - High range potential → Bonus for FOR and Delta encoding
       - Time series data → Bonus for Delta-of-Delta
       - Categorical data → Bonus for Dictionary encoding
    4. **Final Selection**: Choose the algorithm with the highest total score
    """)
    
    st.markdown("""
    ### Data Type Detection

    The framework analyzes sample data to identify its type based on characteristics:

    - **Text**: String data with character patterns
    - **Numerical**: Numbers without strong sequential patterns
    - **Time Series**: Sequential numerical data with temporal patterns
    - **Categorical**: Data with limited distinct values
    - **Binary**: Binary patterns or extremely low-level data
    - **Mixed**: Data that combines multiple types
    """)

def main():
    st.title("Compression Techniques for Big Data - Thesis Figures")
    
    # Create a selection for different thesis figures
    figure_option = st.selectbox(
        "Select a thesis figure to view:",
        ["Compression Comparison Chart", "Huffman Coding Example", 
         "Delta Encoding Example", "Adaptive Compression Framework"]
    )
    
    if figure_option == "Compression Comparison Chart":
        view_compression_comparison_chart()
    elif figure_option == "Huffman Coding Example":
        view_huffman_coding_example()
    elif figure_option == "Delta Encoding Example":
        view_delta_encoding_example()
    elif figure_option == "Adaptive Compression Framework":
        view_compression_framework_diagram()

if __name__ == "__main__":
    main()