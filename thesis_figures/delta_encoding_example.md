# Delta Encoding Example

Delta encoding is a compression technique particularly effective for time series or sorted numerical data. It works by storing the differences between consecutive values rather than the values themselves.

## Example Sequence

Original values: `[100, 102, 106, 109, 110, 112, 115, 117, 118, 120]`

## Delta Encoding Process

1. Store the first value as is: `100`
2. For each subsequent value, store the difference from the previous value

## Delta Values

| Index | Original Value | Delta Value | Storage |
|-------|---------------|-------------|---------|
| 0 | 100 | - | 100 (full precision) |
| 1 | 102 | +2 | 2 |
| 2 | 106 | +4 | 4 |
| 3 | 109 | +3 | 3 |
| 4 | 110 | +1 | 1 |
| 5 | 112 | +2 | 2 |
| 6 | 115 | +3 | 3 |
| 7 | 117 | +2 | 2 |
| 8 | 118 | +1 | 1 |
| 9 | 120 | +2 | 2 |

## Bit Requirements

- Original values: 10 × 32 bits = 320 bits (assuming 32-bit integers)
- Delta encoded: 32 bits (first value) + 20 bits (deltas) = 52 bits
- Compression ratio: 83.75%

## Delta Decoding Process

1. Start with the first value: `100`
2. For each delta, add it to the previous value to get the next value
3. `100 + 2 = 102`, `102 + 4 = 106`, `106 + 3 = 109`, etc.

## Advantages

- Extremely effective for smooth time series data
- Very fast encoding and decoding
- Simple implementation
- Can be combined with other compression techniques

## Disadvantages

- Less effective for random or uncorrelated data
- Requires special handling for missing values
- Susceptible to error propagation (one error affects all subsequent values)

## Applications in Distributed Systems

- Sensor data storage and transmission
- System monitoring metrics
- Financial time series
- Database indexes
- Scientific measurement data