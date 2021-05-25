# MMJoin
Fast Set similarity join (SSJ) / Set containment join (SCJ) Project Query Evaluation using Matrix Multiplication

This is an extension of the original MMJoin implementation as described in [1], mostly optimized for the self-join operation. 

### How it works

Input dataset:
```
R0: 2 11
R1: 10 11
R2: 3 4 5 11
R3: 3 5 10 11
R4: 5 6 7 11
```
Index:
```
I(0) = I(1) = I(8) = I(9) = []
I(2) = [R0]
I(3) = [R2, R3]
I(4) = [R2]
I(5) = [R2, R3, R4]
I(6) = [R4]
I(7) = [R4]
I(10) = [R1, R3]
I(11) = [R0, R1, R2, R3, R4]
```

Input sets must be sorted by set size in ascending order.

We want to find:
- the number of pairs of sets where one is contained within the other set (SCJ), or
- the number of pairs of sets where each pair shares c common elements (SSJ)

The main reasoning is to distinguish sets into light and heavy based on the set degree/size. The same applies also for set elements, based on the inverted list degree/size.

Suppose that the cutoff degree for sets and elements is equal to 3. Thus, there are two light sets, i.e. R0 and R1, since their degree is less than 3. The rest, i.e. R2-R4 are are characterized as heavy sets. For the set elements, only 5 and 11 are characterized as heavy elements since the respective inverted lists' sizes are greater than 3.

To process the join:
- We use the inverted index for the light sets
- We use the inverted index for the light elements of the heavy sets
- We use the matrix multiplication for the heavy elements of the heavy sets

### Build
In order to succesfully build the executable, you will need to have installed
- The Eigen library
- The Intel MKL library
- OpenMP

```
mkdir build && cd build
cmake ..
make -j4
```

### Execute

```
./mmjoin [OPTION...]

      --input arg          Input dataset file
      -c arg               Overlap/Common elements (for set similarity join)
      --scj                Set containment join (ignores c)
      --iter arg           Number of iterations for the cost-based optimizer
      --rf arg             The relation factor for the cost-based optimizer
      --eps arg            The epsilon factor for the cost-based optimizer
      --threads arg        Number of parallel threads
      --delta-set arg      Delta Set (Set cutoff degree)
      --delta-element arg  Delta Element (Element cutoff degree)
      --help               Print help
```

#### Example (SCJ)
```
./mmjoin --input ./datasets/simple.txt --scj --delta-set 3 --delta-element 3
```
#### Example (SSJ)
```
./mmjoin --input ./datasets/simple.txt -c 2 
```
### TODO
 - Optimize matrix multiplication to produce only the upper triangle (without the diagonal)
 - Offload matrix multiplication to the GPU
 
### References

[1] Shaleen Deep, Xiao Hu, and Paraschos Koutris. 2020. Fast Join Project Query Evaluation using Matrix Multiplication. In Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data (SIGMOD '20).
