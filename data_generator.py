import numpy as np
import math
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
    # Initialize time series
    x = np.arange(n)
    y = np.zeros(n)
    
    # Add trend component
    if trend:
        trend_component = 0.5 * x
        y += trend_component
    
    # Add seasonal component
    if seasonality:
        seasons = 4  # Number of complete seasonal cycles
        frequency = 2 * math.pi * seasons / n
        seasonal_component = 10 * np.sin(frequency * x)
        y += seasonal_component
    
    # Add noise
    if noise:
        noise_level = 2.0
        random_noise = np.random.normal(0, noise_level, n)
        y += random_noise
    
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
    # Base temperature pattern (e.g., daily cycle)
    time = np.linspace(0, 2*np.pi, n)
    base_temp = 20 + 5 * np.sin(time)  # Temperature oscillating around 20°C
    
    if num_sensors == 1:
        # Add some random noise
        noise = np.random.normal(0, 1, n)
        return base_temp + noise
    else:
        # Generate data for multiple sensors with correlation
        sensors_data = np.zeros((n, num_sensors))
        
        for i in range(num_sensors):
            # Shared component (correlation)
            shared = base_temp
            
            # Individual component (1 - correlation)
            individual = np.random.normal(0, 3, n)
            
            # Combine components based on correlation
            sensors_data[:, i] = correlation * shared + (1 - correlation) * individual
            
            # Add sensor-specific offset
            sensors_data[:, i] += np.random.uniform(-3, 3)
        
        return sensors_data

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
    # Generate a vocabulary
    word_length = np.random.randint(3, 10, vocabulary_size)
    vocabulary = []
    
    for length in word_length:
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        vocabulary.append(word)
    
    # Generate probabilities following Zipf distribution
    ranks = np.arange(1, vocabulary_size + 1)
    probs = 1 / np.power(ranks, zipf_param)
    probs = probs / probs.sum()
    
    # Generate text by sampling from vocabulary based on Zipfian probabilities
    words = np.random.choice(vocabulary, size=n_words, p=probs)
    
    # Add some structure (sentences)
    sentence_length = np.random.randint(5, 15)
    structured_text = []
    
    for i, word in enumerate(words):
        if i % sentence_length == 0 and i > 0:
            structured_text.append('.')
            word = word.capitalize()
        structured_text.append(word)
    
    return ' '.join(structured_text)

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
    # Initial price
    price = 100
    prices = [price]
    
    # Generate price series
    for _ in range(n-1):
        # Daily return with drift and volatility
        daily_return = np.random.normal(drift, volatility)
        
        # Update price
        price = price * (1 + daily_return)
        prices.append(price)
    
    return np.array(prices)

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
    categories = [f"CAT_{i}" for i in range(n_categories)]
    
    if distribution == 'uniform':
        # Uniform distribution of categories
        probs = np.ones(n_categories) / n_categories
        return np.random.choice(categories, size=n, p=probs).tolist()
    
    elif distribution == 'skewed':
        # Skewed distribution (some categories are more common)
        probs = np.random.exponential(scale=1.0, size=n_categories)
        probs = probs / np.sum(probs)
        return np.random.choice(categories, size=n, p=probs).tolist()
    
    elif distribution == 'temporal':
        # Temporal pattern (categories appear in sequences)
        data = []
        current_category = np.random.choice(categories)
        
        for _ in range(n):
            data.append(current_category)
            
            # 10% chance to switch to a different category
            if np.random.random() < 0.1:
                current_category = np.random.choice(categories)
        
        return data
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

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
    dataset = {}
    
    # ID column (unique integers)
    dataset['id'] = np.arange(n_records)
    
    # Timestamp column (sequential datetimes)
    base_time = np.datetime64('2023-01-01')
    time_deltas = np.random.exponential(scale=3600, size=n_records)  # Average 1 hour between events
    time_deltas = np.sort(time_deltas)
    time_deltas = np.cumsum(time_deltas).astype(int)
    dataset['timestamp'] = [base_time + np.timedelta64(int(delta), 's') for delta in time_deltas]
    
    # Numerical columns
    dataset['value1'] = np.random.normal(100, 15, n_records)  # Normal distribution
    dataset['value2'] = np.random.exponential(scale=50, size=n_records)  # Exponential distribution
    
    # Categorical columns
    dataset['category1'] = generate_categorical_data(n_records, 5, 'skewed')
    dataset['category2'] = generate_categorical_data(n_records, 20, 'uniform')
    dataset['category3'] = generate_categorical_data(n_records, 3, 'temporal')
    
    # Text column (short descriptions)
    descriptions = []
    templates = [
        "Observation of {} with value {}",
        "Measurement: {} at position {}",
        "Reading {} from sensor {}",
        "Data point {} with confidence {}"
    ]
    
    for i in range(n_records):
        template = np.random.choice(templates)
        val1 = round(dataset['value1'][i], 2)
        val2 = round(dataset['value2'][i], 2)
        descriptions.append(template.format(val1, val2))
    
    dataset['description'] = descriptions
    
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
        # Low entropy: Mostly repetitive patterns
        pattern_length = 16
        pattern = bytes([random.randint(0, 255) for _ in range(pattern_length)])
        
        # Generate data by repeating the pattern with occasional variations
        data = bytearray()
        while len(data) < n_bytes:
            if random.random() < 0.9:  # 90% chance to use the pattern
                data.extend(pattern)
            else:
                # Small variation
                variation = bytearray(pattern)
                for _ in range(random.randint(1, 3)):
                    pos = random.randint(0, pattern_length - 1)
                    variation[pos] = random.randint(0, 255)
                data.extend(variation)
        
        return bytes(data[:n_bytes])
    
    elif entropy == 'medium':
        # Medium entropy: Some structure but not fully repetitive
        # Create data with some regions of repetition and some random regions
        data = bytearray()
        
        while len(data) < n_bytes:
            if random.random() < 0.5:  # 50% chance for repetitive section
                pattern_length = random.randint(4, 16)
                pattern = bytes([random.randint(0, 255) for _ in range(pattern_length)])
                
                repeat_count = random.randint(5, 20)
                for _ in range(repeat_count):
                    if len(data) < n_bytes:
                        data.extend(pattern)
            else:
                # Random section
                random_length = random.randint(20, 100)
                data.extend([random.randint(0, 255) for _ in range(random_length)])
        
        return bytes(data[:n_bytes])
    
    elif entropy == 'high':
        # High entropy: Nearly random data
        return bytes([random.randint(0, 255) for _ in range(n_bytes)])
    
    else:
        raise ValueError(f"Unknown entropy level: {entropy}")
