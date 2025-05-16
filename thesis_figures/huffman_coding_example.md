# Huffman Coding Example

Huffman coding is an entropy-based compression algorithm that assigns variable-length codes to input characters based on their frequencies. The most frequent characters get the shortest codes.

## Example Text

```
COMPRESSION_EXAMPLE
```

## Character Frequencies

| Character | Frequency |
|-----------|-----------|
| E | 3 |
| _ | 3 |
| M | 2 |
| P | 2 |
| O | 2 |
| C | 1 |
| R | 1 |
| S | 1 |
| I | 1 |
| N | 1 |
| X | 1 |
| A | 1 |
| L | 1 |

## Huffman Tree Construction

1. Create leaf nodes for each character with their frequencies
2. Build a min-heap (priority queue) with these nodes
3. While there is more than one node in the heap:
   - Extract the two nodes with lowest frequency
   - Create a new internal node with these two as children
   - Assign the sum of the two frequencies to this new node
   - Add this node back to the min-heap
4. The remaining node is the root of the Huffman tree

## Huffman Tree

```
              (20)
             /    \
          (8)      (12)
         /   \     /   \
      (3)    (5) (5)    (7)
     /  \    / \  / \   / \
   C(1) R(1) S(1) I(1) N(1) X(1)
```

## Huffman Codes

| Character | Code |
|-----------|------|
| E | 00 |
| _ | 01 |
| M | 100 |
| P | 101 |
| O | 110 |
| C | 1110 |
| R | 1111 |
| S | 1000 |
| I | 1001 |
| N | 1010 |
| X | 1011 |
| A | 1100 |
| L | 1101 |

## Compression Results

- Original size: 160 bits (20 characters × 8 bits)
- Compressed size: 45 bits
- Compression ratio: 71.9%

## Advantages and Limitations

### Advantages
- Optimal for known frequency distributions
- Variable-length codes adapt to data characteristics
- Lossless compression preserves all information

### Limitations
- Requires knowledge of frequency distribution (two passes or statistical model)
- Overhead of storing the codebook
- Less effective for uniform distributions