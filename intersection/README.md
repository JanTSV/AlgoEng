# Solution

This is the solution for Problem 1.

To get started, simply run
```bash
make
```

This should build an executable (`intersection`) and copy it to your working directory.

The size of A (`n`) is inputted with a command line parameter:
```
Usage: ./intersection <n>
```

# Comparison

`csv_test_whats_faster.csv` and `csv_test_whats_faster.PNG` yield a comparison of the running times of all strategies run on my PC.

To create a `csv_test_whats_faster.csv` yourself, run
```bash
cargo test --release csv_test_whats_faster 
```
