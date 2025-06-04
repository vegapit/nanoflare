```
cmake -B build -D LIBTORCH_DIR=$LIBTORCH
cmake --build build --config Release

cd build

# Run tests
make test

# Run benchmarking
./tests/models_benchmarking
```