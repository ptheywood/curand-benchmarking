# Benchmarking CuRAND 


The curand docs / website do not offer recent performance comparisons for the different RNG engines available.

This is a very simple app to do that.


## Building


```
mkdir -p build
cd build
cmake ..
make
```

NVTX markers can be enabled via -DUSE_NVTX=ON


## Running

```
cd build/
./curandbench
```