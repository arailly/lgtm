# LGTM: A Fast and Accurate kNN Search Algorithm in High-dimensional Spaces

This is an implementation of LGTM proposed by [this paper](https://link.springer.com/chapter/10.1007/978-3-030-86475-0_22).

## Requirements
- Linux (confirmed on Ubuntu 18.04)
- g++ >= 7.5.0
- CMake >= 3.19
- OpenMP >= 5.0.1

## How To Build
```
$ cmake .
$ make
```

## Make Approximate K Nearest Neighbors Graph (AKNNG)
### Options
- n: Cardinality of dataset
- d: Dimensionality
- K: Degree of AKNNG

### Argument
1. Path to dataset (fvecs format)

### Usage
```
$ ./make-aknng -n 1000000 -d 128 -K 20 /path/to/dataset.fvecs
```

### Result
- aknng.ivecs: Adjacent nodes of each data

## Make Graph-based Index of LGTM
### Options
- n: Cardinality of dataset
- d: Dimensionality
- D: Degree of each node (NOTE: Finally, each node has 2 * D degree, because our algorithm makes processed AKNNG bidirected)

### Arguments
1. Path to dataset
2. Path to AKNNG

### Usage
```
$ ./make-index -n 1000000 -d 128 -D 15 /path/to/dataset.fvecs ./aknng.ivecs
```

### Result
- lgtm-index.ivecs: Adjacent nodes of each data

## AkNN Search by LGTM Algorithm
### Options
- k: Result size
- N: Number of query
- n: Cardinality of dataset
- d: Dimensionality
- m: Number of hash functions
- w: Hash width
- t: Number of hash tables (= Number of available threads)
- D: Degree of each node of LGTM Index
- c: Size of Candidate set during search
- s: Number of point got from LSH

### Arguments
1. Path to queries
2. Path to dataset
3. Path to LGTM Index

### Usage
```
./aknn-search -k 10 -N 10000 -n 1000000 -d 128 -m 4 -w 200 -t 8 -D 15 -c 10 -s 50 /path/to/queries.fvecs /path/to/dataset.fvecs ./lgtm-index.ivecs
```

### Result
- result.csv: Data ids of result
- log.csv: search time for each querying

## Sample Dataset
- SIFT, GIST (http://corpus-texmex.irisa.fr/)
- Deep (https://research.yandex.com/datasets/biganns)

## Setting Examples
| Dataset | n | d | K | D | k | N | m | w | t | c | s |
| ------- | - | - | - | - | - | - | - | - | - | - | - |
| SIFT    |  1,000,000 | 128 | 20 | 15 | 10 | 10,000 |  4 | 200 | 8 | 10 | 50 |
| GIST    |  1,000,000 | 960 | 40 | 40 | 10 |  1,000 |  5 |   1 | 8 | 10 | 50 |
| Deep    | 10,000,000 |  96 | 30 | 15 | 10 | 10,000 | 10 |   1 | 8 | 10 | 50 |

