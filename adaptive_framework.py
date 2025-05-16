import numpy as np
import math
from collections import Counter
import compression_algorithms as ca
import utils

class AdaptiveCompressionFramework:
    """
    Framework for adaptively selecting and applying compression algorithms
    based on data characteristics and system constraints
    """
    def __init__(self):
        self.algorithms = {
            'huffman': {
                'compress': ca.huffman_coding_demo,
                'decompress': None,  # Simplified for demonstration
                'data_types': ['text', 'categorical'],
                'speed_priority': 'medium',
                'ratio_priority': 'high'
            },
            'delta': {
                'compress': ca.delta_encode,
                'decompress': ca.delta_decode,
                'data_types': ['numerical', 'time_series'],
                'speed_priority': 'high',
                'ratio_priority': 'medium'
            },
            'delta_of_delta': {
                'compress': ca.delta_of_delta_encode,
                'decompress': ca.delta_of_delta_decode,
                'data_types': ['time_series'],
                'speed_priority': 'medium',
                'ratio_priority': 'high'
            },
            'lzw': {
                'compress': ca.lzw_compress,
                'decompress': ca.lzw_decompress,
                'data_types': ['text', 'mixed'],
                'speed_priority': 'medium',
                'ratio_priority': 'high'
            },
            'rle': {
                'compress': ca.rle_encode,
                'decompress': ca.rle_decode,
                'data_types': ['binary', 'categorical'],
                'speed_priority': 'very_high',
                'ratio_priority': 'low'
            },
            'dictionary': {
                'compress': ca.dictionary_encode,
                'decompress': ca.dictionary_decode,
                'data_types': ['categorical'],
                'speed_priority': 'high',
                'ratio_priority': 'medium'
            },
            'for': {
                'compress': ca.for_encode,
                'decompress': ca.for_decode,
                'data_types': ['numerical'],
                'speed_priority': 'high',
                'ratio_priority': 'medium'
            }
        }
    
    def analyze_data(self, data):
        """
        Analyze data characteristics to determine suitable compression algorithms
        """
        analysis = {}
        
        # Determine data type
        if isinstance(data, str):
            analysis['data_type'] = 'text'
        elif isinstance(data, (list, np.ndarray)):
            if len(data) == 0:
                analysis['data_type'] = 'unknown'
            elif isinstance(data[0], (int, float, np.number)):
                # Check for time series characteristics
                if len(data) > 10:
                    # Calculate autocorrelation to detect time series
                    autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
                    if autocorr > 0.7:
                        analysis['data_type'] = 'time_series'
                    else:
                        analysis['data_type'] = 'numerical'
                else:
                    analysis['data_type'] = 'numerical'
            elif isinstance(data[0], str):
                # Check cardinality ratio to detect categorical data
                unique_ratio = len(set(data)) / len(data)
                if unique_ratio < 0.1:
                    analysis['data_type'] = 'categorical'
                else:
                    analysis['data_type'] = 'text'
            else:
                analysis['data_type'] = 'mixed'
        elif isinstance(data, bytes):
            analysis['data_type'] = 'binary'
        else:
            analysis['data_type'] = 'unknown'
        
        # Calculate entropy for text or binary data
        if analysis['data_type'] in ['text', 'binary', 'categorical']:
            try:
                counter = Counter(data)
                total = sum(counter.values())
                probabilities = [count / total for count in counter.values()]
                analysis['entropy'] = -sum(p * math.log2(p) for p in probabilities)
            except:
                analysis['entropy'] = 8.0  # Default value if calculation fails
        
        # Analyze run lengths for RLE potential
        if analysis['data_type'] in ['binary', 'categorical', 'numerical']:
            try:
                runs = 1
                for i in range(1, len(data)):
                    if data[i] != data[i-1]:
                        runs += 1
                analysis['run_ratio'] = runs / len(data)
            except:
                analysis['run_ratio'] = 1.0  # Default value if calculation fails
        
        # Analyze value range for FOR potential
        if analysis['data_type'] in ['numerical', 'time_series']:
            try:
                data_arr = np.array(data)
                min_val = np.min(data_arr)
                max_val = np.max(data_arr)
                range_val = max_val - min_val
                
                # Estimate bits needed for full values vs. offsets
                if max_val > 0:
                    full_bits = math.ceil(math.log2(max_val + 1))
                else:
                    full_bits = 1
                
                if range_val > 0:
                    offset_bits = math.ceil(math.log2(range_val + 1))
                else:
                    offset_bits = 1
                
                analysis['range_compression_potential'] = full_bits / offset_bits
            except:
                analysis['range_compression_potential'] = 1.0  # Default value if calculation fails
        
        return analysis
    
    def select_algorithm(self, analysis, constraints=None):
        """
        Select the most appropriate compression algorithm based on data analysis
        and system constraints
        """
        if constraints is None:
            constraints = {'speed_priority': 'medium', 'ratio_priority': 'medium'}
        
        # Filter algorithms suitable for the data type
        data_type = analysis.get('data_type', 'unknown')
        candidates = []
        
        for algo_name, algo_info in self.algorithms.items():
            if data_type in algo_info['data_types'] or 'all' in algo_info['data_types']:
                candidates.append(algo_name)
        
        if not candidates:
            # Fallback to general-purpose algorithms
            candidates = ['lzw', 'huffman']
        
        # Score candidates based on analysis and constraints
        scores = {}
        for algo in candidates:
            score = 0
            
            # Score based on speed priority
            speed_map = {'very_high': 3, 'high': 2, 'medium': 1, 'low': 0}
            algo_speed = speed_map.get(self.algorithms[algo]['speed_priority'], 1)
            req_speed = speed_map.get(constraints['speed_priority'], 1)
            
            if algo_speed >= req_speed:
                score += algo_speed
            else:
                score -= (req_speed - algo_speed) * 2  # Penalty for not meeting speed requirement
            
            # Score based on compression ratio priority
            ratio_map = {'very_high': 3, 'high': 2, 'medium': 1, 'low': 0}
            algo_ratio = ratio_map.get(self.algorithms[algo]['ratio_priority'], 1)
            req_ratio = ratio_map.get(constraints['ratio_priority'], 1)
            
            if algo_ratio >= req_ratio:
                score += algo_ratio
            else:
                score -= (req_ratio - algo_ratio) * 2  # Penalty for not meeting ratio requirement
            
            # Additional scores based on data characteristics
            if algo == 'rle' and analysis.get('run_ratio', 1.0) < 0.1:
                score += 5  # Bonus for RLE if there are few runs
            
            if algo in ['for', 'delta'] and analysis.get('range_compression_potential', 1.0) > 4:
                score += 3  # Bonus for FOR/delta if range compression is promising
            
            if algo == 'huffman' and analysis.get('entropy', 8.0) < 3.0:
                score += 4  # Bonus for Huffman if entropy is low
            
            if algo == 'delta_of_delta' and data_type == 'time_series':
                score += 2  # Bonus for delta-of-delta on time series
            
            if algo == 'dictionary' and data_type == 'categorical':
                score += 3  # Bonus for dictionary encoding on categorical
            
            scores[algo] = score
        
        # Select the highest scoring algorithm
        if not scores:
            return 'lzw'  # Default fallback
        
        selected = max(scores.items(), key=lambda x: x[1])[0]
        return selected
    
    def compress(self, data, constraints=None):
        """
        Compress data using adaptively selected algorithm
        """
        analysis = self.analyze_data(data)
        selected_algo = self.select_algorithm(analysis, constraints)
        
        # Apply selected algorithm
        compress_func = self.algorithms[selected_algo]['compress']
        compressed_data = compress_func(data)
        
        # Calculate compression statistics
        original_size = utils.get_data_size(data)
        if isinstance(compressed_data, tuple):
            # Some algorithms return multiple values
            compressed_size = sum(utils.get_data_size(item) for item in compressed_data)
        else:
            compressed_size = utils.get_data_size(compressed_data)
        
        compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        # Return compressed data along with algorithm info for decompression
        return {
            'algorithm': selected_algo,
            'compressed_data': compressed_data,
            'analysis': analysis,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio
        }
    
    def decompress(self, compression_result):
        """
        Decompress data using the algorithm specified in compression_result
        """
        algorithm = compression_result['algorithm']
        compressed_data = compression_result['compressed_data']
        
        decompress_func = self.algorithms[algorithm]['decompress']
        if decompress_func is None:
            return compressed_data  # For algorithms without explicit decompression
        
        return decompress_func(*compressed_data if isinstance(compressed_data, tuple) else compressed_data)

    def get_algorithm_details(self):
        """
        Get details about all available algorithms
        """
        return {
            algo_name: {
                'data_types': info['data_types'],
                'speed_priority': info['speed_priority'],
                'ratio_priority': info['ratio_priority']
            }
            for algo_name, info in self.algorithms.items()
        }

    def get_data_type_recommendations(self, data_type):
        """
        Get algorithm recommendations for a specific data type
        """
        recommendations = []
        
        for algo_name, info in self.algorithms.items():
            if data_type in info['data_types']:
                recommendations.append({
                    'algorithm': algo_name,
                    'speed_priority': info['speed_priority'],
                    'ratio_priority': info['ratio_priority']
                })
        
        # Sort by ratio priority (high to low)
        ratio_map = {'very_high': 4, 'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: ratio_map.get(x['ratio_priority'], 0), reverse=True)
        
        return recommendations