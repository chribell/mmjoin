# MMJoin
Fast Set similarity join (SSJ) / Set containment join (SCJ) Project Query Evaluation using Matrix Multiplication

This is an extension of the original MMJoin implementation as described in [1], mostly optimized for the self-join operation. 

### How it works

### Build
In order to succesfully build the executable, you will need to have to installed
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
