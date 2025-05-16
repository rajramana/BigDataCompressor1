import numpy as np
import math
import time

import compression_algorithms as ca
import utils

class AdaptiveCompressionFramework:
    """
    Framework for adaptively selecting and applying compression algorithms
    based on data characteristics and system constraints
    """
    def __init__(self):
        # Define available algorithms with their properties
        self.algorithms = {
            "huffman": {
                "data_types": ["text", "binary"],
                "speed_priority": "medium",
                "ratio_priority": "high"
            },
            "delta": {
                "data_types": ["numerical", "time_series"],
                "speed_priority": "very_high",
                "ratio_priority": "medium"
            },
            "delta_of_delta": {
                "data_types": ["time_series"],
                "speed_priority": "high",
                "ratio_priority": "high"
            },
            "lzw": {
                "data_types": ["text", "binary", "mixed"],
                "speed_priority": "medium",
                "ratio_priority": "medium"
            },
            "rle": {
                "data_types": ["binary", "categorical", "text"],
                "speed_priority": "very_high",
                "ratio_priority": "low"
            },
            "dictionary": {
                "data_types": ["categorical", "text"],
                "speed_priority": "high",
                "ratio_priority": "high"
            },
            "for": {
                "data_types": ["numerical", "time_series"],
                "speed_priority": "high",
                "ratio_priority": "medium"
            }
        }
        
        # Priority level mapping (for scoring)
        self.priority_scores = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "very_high": 4
        }
    
    def analyze_data(self, data):
        """
        Analyze data characteristics to determine suitable compression algorithms
        
        Parameters:
        -----------
        data : object
            The data to analyze
            
        Returns:
        --------
        dict
            Dictionary containing analysis results
        """
        # Get representative sample of data for analysis
        sample_data = utils.get_sample_data(data)
        
        # Detect data type
        data_type = utils.detect_data_type(sample_data)
        
        # Initialize analysis results
        analysis = {
            "data_type": data_type
        }
        
        # Calculate entropy (for all data types)
        try:
            entropy = utils.calculate_entropy(sample_data)
            analysis["entropy"] = entropy
        except:
            # Skip entropy calculation if not applicable
            pass
        
        # Analyze run-length characteristics (useful for RLE)
        try:
            run_analysis = utils.analyze_runs(sample_data)
            analysis.update(run_analysis)
        except:
            # Skip run analysis if not applicable
            pass
        
        # Analyze numerical data compression potential
        if data_type in ["numerical", "time_series"]:
            try:
                range_analysis = utils.analyze_range_compression(sample_data)
                analysis.update(range_analysis)
            except:
                # Skip range analysis if not applicable
                pass
        
        # Analyze dictionary compression potential
        if data_type in ["categorical", "text", "mixed"]:
            try:
                dict_analysis = utils.calculate_dictionary_potential(sample_data)
                analysis.update(dict_analysis)
            except:
                # Skip dictionary analysis if not applicable
                pass
        
        return analysis
    
    def select_algorithm(self, analysis, constraints=None):
        """
        Select the most appropriate compression algorithm based on data analysis
        and system constraints
        
        Parameters:
        -----------
        analysis : dict
            Dictionary with data analysis results
        constraints : dict, optional
            Dictionary with system constraints (e.g., speed_priority, ratio_priority)
            
        Returns:
        --------
        str
            Name of the selected algorithm
        """
        # Default constraints if none provided
        if constraints is None:
            constraints = {
                "speed_priority": "medium",
                "ratio_priority": "medium"
            }
        
        # Normalize constraints
        speed_priority = constraints.get("speed_priority", "medium")
        ratio_priority = constraints.get("ratio_priority", "medium")
        
        # Get data type
        data_type = analysis.get("data_type", "unknown")
        
        # Filter algorithms suitable for this data type
        suitable_algorithms = []
        for algo_name, algo_props in self.algorithms.items():
            if data_type in algo_props["data_types"] or "mixed" in algo_props["data_types"]:
                suitable_algorithms.append(algo_name)
        
        # If no suitable algorithms found, fallback to general-purpose ones
        if not suitable_algorithms:
            suitable_algorithms = ["lzw", "huffman"]
        
        # Score algorithms based on constraints and data characteristics
        algorithm_scores = {}
        
        for algo_name in suitable_algorithms:
            # Base score from constraint matching
            algo_props = self.algorithms[algo_name]
            speed_score = self._match_priority_score(algo_props["speed_priority"], speed_priority)
            ratio_score = self._match_priority_score(algo_props["ratio_priority"], ratio_priority)
            
            # Base score is weighted sum of speed and ratio scores
            base_score = speed_score + ratio_score
            
            # Additional scoring based on data characteristics
            bonus = 0
            
            # Low entropy favors Huffman coding
            if "entropy" in analysis and algo_name == "huffman":
                # Lower entropy (closer to 0) means better Huffman performance
                entropy_bonus = max(0, 5 - analysis["entropy"])
                bonus += entropy_bonus
            
            # Low run ratio favors RLE
            if "run_ratio" in analysis and algo_name == "rle":
                # Lower run ratio means better RLE performance
                run_bonus = max(0, 3 * (1 - analysis["run_ratio"]))
                bonus += run_bonus
            
            # High range compression potential favors FOR and Delta encoding
            if "range_compression_potential" in analysis:
                if algo_name in ["for", "delta"]:
                    range_bonus = min(3, analysis["range_compression_potential"] / 10)
                    bonus += range_bonus
            
            # High dictionary potential favors Dictionary encoding
            if "dictionary_potential" in analysis and algo_name == "dictionary":
                dict_bonus = min(3, analysis["dictionary_potential"] * 5)
                bonus += dict_bonus
            
            # Time series data favors delta-of-delta for smooth series
            if data_type == "time_series" and algo_name == "delta_of_delta":
                # Bonus for delta-of-delta on time series
                bonus += 1
            
            # Final score
            algorithm_scores[algo_name] = base_score + bonus
        
        # Select algorithm with highest score
        selected_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])[0]
        
        return selected_algorithm
    
    def compress(self, data, constraints=None):
        """
        Compress data using adaptively selected algorithm
        
        Parameters:
        -----------
        data : object
            The data to compress
        constraints : dict, optional
            Dictionary with system constraints
            
        Returns:
        --------
        dict
            Dictionary with compression results
        """
        # Analyze data
        analysis = self.analyze_data(data)
        
        # Select algorithm
        algorithm = self.select_algorithm(analysis, constraints)
        
        # Apply selected algorithm
        start_time = time.time()
        
        # Different handling for different algorithms
        if algorithm == "huffman":
            compressed_size, compression_ratio, codes = ca.huffman_coding_demo(data)
            compressed_data = None  # Huffman demo doesn't return actual compressed data
        elif algorithm == "delta":
            first_value, deltas = ca.delta_encode(data)
            compressed_data = (first_value, deltas)
            # Calculate approximate compression metrics
            original_size = len(data) * 8 if isinstance(data, str) else len(data) * 64
            delta_bits = sum(utils.estimate_bits_needed(deltas))
            compressed_size = 64 + delta_bits  # First value + deltas
            compression_ratio = (1 - compressed_size / original_size) * 100
        elif algorithm == "delta_of_delta":
            first_value, first_delta, second_deltas = ca.delta_of_delta_encode(data)
            compressed_data = (first_value, first_delta, second_deltas)
            # Calculate approximate compression metrics
            original_size = len(data) * 8 if isinstance(data, str) else len(data) * 64
            delta_bits = sum(utils.estimate_bits_needed(second_deltas))
            compressed_size = 64 + 64 + delta_bits  # First value + first delta + second deltas
            compression_ratio = (1 - compressed_size / original_size) * 100
        elif algorithm == "lzw":
            compressed_data = ca.lzw_compress(data)
            # Calculate approximate compression metrics
            original_size = len(data) * 8 if isinstance(data, str) else len(data) * 8
            dict_size = 256 + len(compressed_data)
            bits_per_code = max(8, math.ceil(math.log2(dict_size)))
            compressed_size = len(compressed_data) * bits_per_code
            compression_ratio = (1 - compressed_size / original_size) * 100
        elif algorithm == "rle":
            compressed_data = ca.rle_encode(data)
            # Calculate approximate compression metrics
            original_size = len(data) * 8 if isinstance(data, str) else len(data) * 8
            compressed_size = len(compressed_data) * 16  # Each run is (value, count)
            compression_ratio = (1 - compressed_size / original_size) * 100
        elif algorithm == "dictionary":
            encoded, value_to_id = ca.dictionary_encode(data)
            compressed_data = (encoded, value_to_id)
            # Calculate approximate compression metrics
            original_size = len(data) * 8 if isinstance(data, str) else len(data) * 8
            dict_size_bits = len(value_to_id) * 16
            id_bits = max(1, math.ceil(math.log2(len(value_to_id))))
            encoded_bits = len(encoded) * id_bits
            compressed_size = dict_size_bits + encoded_bits
            compression_ratio = (1 - compressed_size / original_size) * 100
        elif algorithm == "for":
            reference, offsets, bits_per_value = ca.for_encode(data)
            compressed_data = (reference, offsets, bits_per_value)
            # Calculate approximate compression metrics
            original_size = len(data) * 8 if isinstance(data, str) else len(data) * 64
            compressed_size = 64 + len(offsets) * bits_per_value
            compression_ratio = (1 - compressed_size / original_size) * 100
        
        compression_time = time.time() - start_time
        
        # Convert to bytes for consistent handling
        if isinstance(compressed_size, int):
            compressed_size = compressed_size // 8
        else:
            compressed_size = int(compressed_size // 8)
            
        if isinstance(original_size, int):
            original_size = original_size // 8
        else:
            original_size = int(original_size // 8)
        
        # Return results
        return {
            "algorithm": algorithm,
            "compressed_data": compressed_data,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "compression_time": compression_time,
            "analysis": analysis
        }
    
    def decompress(self, compression_result):
        """
        Decompress data using the algorithm specified in compression_result
        
        Parameters:
        -----------
        compression_result : dict
            Dictionary with compression results from the compress method
            
        Returns:
        --------
        object
            The decompressed data
        """
        algorithm = compression_result["algorithm"]
        compressed_data = compression_result["compressed_data"]
        
        # Apply decompression for the specific algorithm
        if algorithm == "huffman":
            # Huffman demo doesn't return actual compressed data
            return None
        elif algorithm == "delta":
            first_value, deltas = compressed_data
            return ca.delta_decode(first_value, deltas)
        elif algorithm == "delta_of_delta":
            first_value, first_delta, second_deltas = compressed_data
            return ca.delta_of_delta_decode(first_value, first_delta, second_deltas)
        elif algorithm == "lzw":
            return ca.lzw_decompress(compressed_data)
        elif algorithm == "rle":
            return ca.rle_decode(compressed_data)
        elif algorithm == "dictionary":
            encoded, value_to_id = compressed_data
            return ca.dictionary_decode(encoded, value_to_id)
        elif algorithm == "for":
            reference, offsets, bits_per_value = compressed_data
            return ca.for_decode(reference, offsets, bits_per_value)
    
    def get_algorithm_details(self):
        """
        Get details about all available algorithms
        
        Returns:
        --------
        dict
            Dictionary with information about available algorithms
        """
        return self.algorithms
    
    def get_data_type_recommendations(self, data_type):
        """
        Get algorithm recommendations for a specific data type
        
        Parameters:
        -----------
        data_type : str
            Data type to get recommendations for
            
        Returns:
        --------
        list
            List of recommended algorithms with their properties
        """
        # Validate data type
        valid_types = ["text", "numerical", "time_series", "categorical", "binary", "mixed"]
        if data_type not in valid_types:
            data_type = "mixed"  # Default to mixed
        
        # Find suitable algorithms
        recommendations = []
        for algo_name, algo_props in self.algorithms.items():
            if data_type in algo_props["data_types"] or "mixed" in algo_props["data_types"]:
                recommendations.append({
                    "algorithm": algo_name,
                    "speed_priority": algo_props["speed_priority"],
                    "ratio_priority": algo_props["ratio_priority"]
                })
        
        # Sort by ratio priority (higher to lower)
        recommendations.sort(
            key=lambda x: self.priority_scores[x["ratio_priority"]], 
            reverse=True
        )
        
        return recommendations
    
    def _match_priority_score(self, algorithm_priority, user_priority):
        """
        Calculate priority match score between algorithm and user priorities
        
        Parameters:
        -----------
        algorithm_priority : str
            Algorithm's priority level
        user_priority : str
            User's priority level
            
        Returns:
        --------
        float
            Score representing how well the priorities match
        """
        algo_score = self.priority_scores.get(algorithm_priority, 2)
        user_score = self.priority_scores.get(user_priority, 2)
        
        # Higher score if priorities match, lower if they diverge
        if algo_score == user_score:
            return 3.0
        elif abs(algo_score - user_score) == 1:
            return 2.0
        else:
            return 1.0