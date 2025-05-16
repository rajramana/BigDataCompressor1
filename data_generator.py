import numpy as np
import random
import string

def generate_time_series(n=100, trend=True, seasonality=True, noise=True):
    """
    Generate a synthetic time series dataset for compression testing
    
    Parameters:
    -----------
    n : int
        Number of data points to generate
    trend : bool
        Whether to include a trend component
    seasonality : bool
        Whether to include seasonal patterns
    noise : bool
        Whether to add random noise
        
    Returns:
    --------
    numpy.ndarray
        Generated time series data
    """
    # Create time index
    t = np.arange(n)
    
    # Initialize time series
    y = np.zeros(n)
    
    # Add trend component
    if trend:
        trend_coef = np.random.uniform(0.01, 0.1)
        y += trend_coef * t
    
    # Add seasonality component
    if seasonality:
        # Add multiple seasonal patterns with different frequencies
        frequencies = [0.05, 0.1, 0.2]  # Different frequencies
        amplitudes = [10, 5, 2]         # Different amplitudes
        phases = [0, np.pi/4, np.pi/2]  # Different phase shifts
        
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            y += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # Add noise component
    if noise:
        noise_level = np.random.uniform(0.5, 2.0)
        y += np.random.normal(0, noise_level, n)
    
    return y

def generate_sensor_data(n=100, num_sensors=1, correlation=0.7):
    """
    Generate synthetic sensor data for compression testing
    
    Parameters:
    -----------
    n : int
        Number of time points
    num_sensors : int
        Number of sensors to simulate
    correlation : float
        Correlation between sensors (0-1)
        
    Returns:
    --------
    numpy.ndarray
        If num_sensors is 1, returns a 1D array
        If num_sensors > 1, returns a 2D array (time x sensors)
    """
    if num_sensors == 1:
        # Generate a single sensor time series
        return generate_time_series(n)
    
    # Generate correlated sensor data
    # Start with one series as the reference
    reference = generate_time_series(n)
    
    # Create matrix to hold all sensor data
    data = np.zeros((n, num_sensors))
    data[:, 0] = reference
    
    # Generate correlated time series for other sensors
    for i in range(1, num_sensors):
        # New series is partly reference, partly new
        correlated_part = reference * correlation
        independent_part = (1 - correlation) * generate_time_series(n)
        
        # Combine with some scaling to make it look different
        scale_factor = np.random.uniform(0.8, 1.2)
        data[:, i] = (correlated_part + independent_part) * scale_factor
    
    return data

def generate_text_data(n_words=1000, vocabulary_size=1000, zipf_param=1.5):
    """
    Generate synthetic text data with Zipfian distribution of words
    
    Parameters:
    -----------
    n_words : int
        Number of words in the generated text
    vocabulary_size : int
        Size of the vocabulary to use
    zipf_param : float
        Parameter for Zipf distribution (higher = more skewed frequency)
        
    Returns:
    --------
    str
        Generated text
    """
    # Create vocabulary
    # Use common English words and random words
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
    ]
    
    # If vocabulary size is larger than common words, add random words
    vocab = common_words[:min(len(common_words), vocabulary_size)]
    
    if vocabulary_size > len(vocab):
        # Generate random words
        word_length = 5
        remaining = vocabulary_size - len(vocab)
        
        for _ in range(remaining):
            random_word = ''.join(random.choice(string.ascii_lowercase) 
                                for _ in range(random.randint(3, 10)))
            vocab.append(random_word)
    
    # Generate word frequencies following Zipf distribution
    rank = np.arange(1, len(vocab) + 1)
    freq = 1 / np.power(rank, zipf_param)
    freq = freq / np.sum(freq)  # Normalize to sum to 1
    
    # Sample words based on their frequencies
    words = np.random.choice(vocab, size=n_words, p=freq)
    
    # Add capitalization and punctuation
    text = []
    sentence_length = 0
    capitalize_next = True
    
    for word in words:
        if capitalize_next:
            word = word.capitalize()
            capitalize_next = False
        
        text.append(word)
        sentence_length += 1
        
        # End sentence with probability proportional to length
        if sentence_length > 3 and random.random() < 0.2:
            if random.random() < 0.7:
                text[-1] = text[-1] + "."
            elif random.random() < 0.85:
                text[-1] = text[-1] + "!"
            else:
                text[-1] = text[-1] + "?"
            
            capitalize_next = True
            sentence_length = 0
        else:
            # Add comma with small probability
            if sentence_length > 2 and random.random() < 0.1:
                text[-1] = text[-1] + ","
    
    return " ".join(text)

def generate_stock_prices(n=100, volatility=0.01, drift=0.0005):
    """
    Generate synthetic stock price data using geometric Brownian motion
    
    Parameters:
    -----------
    n : int
        Number of data points
    volatility : float
        Volatility parameter (daily)
    drift : float
        Drift parameter (daily)
        
    Returns:
    --------
    numpy.ndarray
        Generated stock prices
    """
    # Initialize prices array
    prices = np.zeros(n)
    prices[0] = 100.0  # Starting price
    
    # Generate log-normal returns
    dt = 1  # Time step (1 day)
    for i in range(1, n):
        # Geometric Brownian Motion formula
        rng = np.random.normal(0, 1)
        prices[i] = prices[i-1] * np.exp((drift - volatility**2/2) * dt + 
                                        volatility * np.sqrt(dt) * rng)
    
    return prices

def generate_categorical_data(n=1000, n_categories=10, distribution='uniform'):
    """
    Generate synthetic categorical data
    
    Parameters:
    -----------
    n : int
        Number of data points
    n_categories : int
        Number of unique categories
    distribution : str
        Distribution of categories ('uniform', 'skewed', or 'temporal')
        
    Returns:
    --------
    list
        Generated categorical data
    """
    # Generate category labels
    categories = [f'C{i}' for i in range(n_categories)]
    
    if distribution == 'uniform':
        # Uniform distribution
        probabilities = np.ones(n_categories) / n_categories
        data = np.random.choice(categories, size=n, p=probabilities)
        
    elif distribution == 'skewed':
        # Skewed (Zipfian) distribution
        rank = np.arange(1, n_categories + 1)
        probabilities = 1 / rank
        probabilities = probabilities / np.sum(probabilities)  # Normalize
        data = np.random.choice(categories, size=n, p=probabilities)
        
    elif distribution == 'temporal':
        # Temporal pattern (categorical data changes over time)
        data = np.empty(n, dtype=object)
        segment_size = n // 3
        
        # First segment: dominated by first categories
        probs1 = np.exp(-np.arange(n_categories))
        probs1 = probs1 / np.sum(probs1)
        data[:segment_size] = np.random.choice(categories, size=segment_size, p=probs1)
        
        # Second segment: more uniform
        probs2 = np.ones(n_categories) / n_categories
        data[segment_size:2*segment_size] = np.random.choice(
            categories, size=segment_size, p=probs2)
        
        # Third segment: dominated by last categories
        probs3 = np.exp(-np.arange(n_categories)[::-1])
        probs3 = probs3 / np.sum(probs3)
        data[2*segment_size:] = np.random.choice(
            categories, size=n-2*segment_size, p=probs3)
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return data.tolist()

def generate_mixed_dataset(n_records=1000):
    """
    Generate a mixed dataset with multiple column types
    
    Parameters:
    -----------
    n_records : int
        Number of records to generate
        
    Returns:
    --------
    dict
        Dictionary with column names as keys and data arrays as values
    """
    # Define column types and generate data
    dataset = {
        'id': list(range(1, n_records + 1)),
        'timestamp': np.linspace(
            np.datetime64('2020-01-01'), 
            np.datetime64('2020-12-31'), 
            n_records
        ).astype(str),
        'category': generate_categorical_data(n_records, 5, 'skewed'),
        'value': generate_time_series(n_records, trend=True, seasonality=True, noise=True),
        'text': [generate_text_data(random.randint(5, 20), 50) for _ in range(n_records)]
    }
    
    return dataset

def generate_binary_data(n_bytes=1000, entropy='medium'):
    """
    Generate synthetic binary data with controlled entropy
    
    Parameters:
    -----------
    n_bytes : int
        Number of bytes to generate
    entropy : str
        Entropy level ('low', 'medium', or 'high')
        
    Returns:
    --------
    bytes
        Generated binary data
    """
    if entropy == 'low':
        # Low entropy: high redundancy, few unique patterns
        # Generate base pattern
        pattern_size = min(32, n_bytes // 10)
        base_pattern = np.random.bytes(pattern_size)
        
        # Repeat pattern with occasional variations
        result = bytearray()
        while len(result) < n_bytes:
            if random.random() < 0.9:  # 90% chance of repeating pattern
                result.extend(base_pattern)
            else:
                # Add some variation
                variant = bytearray(base_pattern)
                # Modify a few bytes
                for _ in range(random.randint(1, max(1, pattern_size // 4))):
                    pos = random.randint(0, pattern_size - 1)
                    variant[pos] = random.randint(0, 255)
                result.extend(variant)
        
        # Trim to exact size
        return bytes(result[:n_bytes])
    
    elif entropy == 'medium':
        # Medium entropy: some structure but more variation
        result = bytearray()
        
        # Generate repeated byte sequences of different lengths
        while len(result) < n_bytes:
            # Choose a byte value to repeat
            byte_val = random.randint(0, 255)
            # Repeat it a random number of times
            repeat_count = random.randint(1, 20)
            result.extend([byte_val] * repeat_count)
            
            # Occasionally add random bytes
            if random.random() < 0.3:
                num_random = random.randint(1, 10)
                result.extend(random.randint(0, 255) for _ in range(num_random))
        
        # Trim to exact size
        return bytes(result[:n_bytes])
    
    elif entropy == 'high':
        # High entropy: close to random data
        return np.random.bytes(n_bytes)
    
    else:
        raise ValueError(f"Unknown entropy level: {entropy}")