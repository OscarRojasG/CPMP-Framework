# FRG and BS-FRG (C++)

The greedy heuristic FRG and the beam search guided by it (BS-FRG) are the heuristic baselines of the
paper; FRG is also the expert of the imitation stage. The baselines were run with a fork of the
authors' C++ code:

- Original: [rilianx/cpmp](https://github.com/rilianx/cpmp) (commit `a495076`, June 2025).
- Fork used in the paper: [felipe-astudillo-s/cpmp](https://github.com/felipe-astudillo-s/cpmp), tag `paper-v1`.

The fork differs from the original only in `src/main_cpmp.cpp`: `k`, the number of destination stacks
tried for each single move of the beam search, is read from the command line again (it was fixed to
0, which leaves BS-FRG without single moves), and the optional last argument `--moves` prints the
sequence of moves of the solution. `Bsg.*`, `Greedy.*` and `Layout.*` are unchanged.

## Build

From this folder:

```bash
git clone --branch paper-v1 https://github.com/felipe-astudillo-s/cpmp.git
make -C cpmp/src
cp cpmp/src/frg .
```

It builds with the default C++ compiler on Linux and macOS and leaves the program in
`frg_solver/frg`, the path recorded in the result files of `experiments/FRG_BSFRG/`.

The reported runs used a static Linux x86-64 binary built from this tag with GCC 9
(`g++ -O2 -std=c++11 -static main_cpmp.cpp Layout.cpp Greedy.cpp Bsg.cpp -o frg`, in the image
`gcc:9`); built that way, the file has SHA-256
`a7481ba0e169324bd0845c7f55fd38a93cd526efcf8f1cd8f9dac828bed87c65`.

The solver breaks ties with `rand()`, whose sequence depends on the C library. Built on Linux
(glibc), it reproduces the per-instance moves of `experiments/FRG_BSFRG/`. On macOS a few instances
differ by one or two moves (FRG: 24 of the 760, average 31.292 instead of 31.293).

This program is run directly, as in the examples below. It is not a replacement for `../src/frg`, the
binary called by the Python wrappers (`FRGSolver`, `BSGFRGSolver`): that one was compiled from the
same code with `k` fixed to 0, so its command line has no `k`.

## Usage

```
./frg H instance a k max_steps beams <flag> seed [--moves]
```

- `H`: maximum height of the bay (tiers of the CV group plus 2).
- `a`: parameter of the heuristic (1.2 in all runs); `k`: number of destinations considered.
- `max_steps`: step limit. It was never binding in the reported runs (all 760 instances are solved).
- `beams`: 0 for FRG; number of candidates for BS-FRG (1,000 or 10,000 in the paper).
- `<flag>`: `--compound` (FRG with the unblocking assignment of destination stacks),
  `--no-assignment` (FRG without it, the variant that labelled the supervised bank) or
  `--mixed` (BS-FRG).
- `seed`: seed of the random tie-breaking (2 in the reported runs).

The first line of the output is the number of moves (−1 if no solution was found) and the solver
time. With `--moves`, a second line lists the moves as `origin,destination` (0-indexed stacks).

Example with the settings of the paper (`a`, `k`, flag and seed), for the CV group 3-3 (`H = 5`):

```bash
./frg 5 ../instances/benchmarks/cvs/3-3/data3-3-1.dat 1.2 2 100 0 --compound 2        # FRG
./frg 5 ../instances/benchmarks/cvs/3-3/data3-3-1.dat 1.2 3 100 10000 --mixed 2      # BS-FRG, 10,000 candidates
```
