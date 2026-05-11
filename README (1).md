# Parallel Image Processing Pipeline

[![Build & Test](https://github.com/YOUR_USERNAME/parallel-image-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/parallel-image-pipeline/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5-green.svg)](https://www.openmp.org)
[![C Standard](https://img.shields.io/badge/C-C11-orange.svg)]()

**ME 522 – High-Performance Scientific Computing**  
Indian Institute of Technology Mandi

---

## Overview

A fully-parallel, five-stage image processing pipeline implemented in C with
OpenMP, benchmarked on the IIT Mandi HPC cluster.  The pipeline applies:

| Stage | Operation | Parallelism |
|-------|-----------|-------------|
| 1 | Batch image load + float normalisation | `#pragma omp parallel for` (image batch) |
| 2 | Gaussian blur (5×5 separable kernel) | `#pragma omp parallel for` (pixel rows) |
| 3 | Sobel edge detection (3×3) | `#pragma omp parallel for` (pixel rows) |
| 4 | Unsharp mask sharpening | `#pragma omp parallel for` (element-wise) |
| 5 | Histogram equalisation | `#pragma omp parallel for reduction(+:hist[0:256])` |

Performance is measured via strong and weak scaling experiments across 1–32
threads, analysed with Amdahl's law, and visualised with Python/matplotlib.

---

## Repository Structure

```
parallel-image-pipeline/
├── src/
│   ├── main.c          # CLI entry point, timing output, batch orchestration
│   ├── pipeline.c/h    # Five-stage pipeline runner (single + batch)
│   ├── image_io.c/h    # PPM I/O, float normalisation, batch loader
│   ├── filters.c/h     # Gaussian blur, Sobel edge detection, unsharp mask
│   └── histogram.c/h   # Histogram computation (OpenMP reduction) + equalisation
├── python/
│   ├── plot_scaling.py     # Speedup, efficiency, Amdahl, stage breakdown plots
│   ├── plot_images.py      # Side-by-side pipeline stage comparison figure
│   └── requirements.txt
├── scripts/
│   ├── generate_test_images.py   # Synthetic PPM image generator
│   ├── merge_strong_scaling.py   # Merge multi-run CSVs before plotting
│   ├── plot_weak_scaling.py      # Weak scaling efficiency plot
│   └── slurm/
│       ├── strong_scaling.sh     # SLURM job: fixed batch, 1–32 threads
│       └── weak_scaling.sh       # SLURM job: problem size ∝ thread count
├── results/                      # Generated timing CSVs (git-tracked placeholder)
├── Makefile                      # Targets: all, clean, test, bench, plot
├── .github/workflows/ci.yml      # GitHub Actions CI (build + smoke test)
└── README.md
```

---

## Quick Start (Local Machine)

### Prerequisites

| Tool | Minimum Version |
|------|----------------|
| GCC  | 9.x (OpenMP 4.5 support required for array reduction) |
| GNU Make | 4.x |
| Python | 3.10+ |
| numpy | 1.24+ |
| matplotlib | 3.7+ |

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/parallel-image-pipeline.git
cd parallel-image-pipeline

# Install Python dependencies
pip install -r python/requirements.txt

# Build
make all

# Generate 20 synthetic 512×512 test images
make generate_images

# Run pipeline (4 threads, write outputs + timing CSV)
make test
```

Output images land in `output/` and timing data in `results/test_timings.csv`.

---

## Build Targets

| Target | Description |
|--------|-------------|
| `make all` | Compile the pipeline binary with `-O2 -fopenmp` |
| `make clean` | Remove build artifacts, binary, and output PPMs |
| `make generate_images` | Create 20 synthetic 512×512 PPM images in `data/` |
| `make test` | Build + generate images + run pipeline (4 threads) |
| `make bench` | Strong scaling: run at T=1,2,4,8 and write CSVs |
| `make plot` | Generate all scaling figures from CSVs |
| `make hpc_submit` | Submit SLURM jobs (IIT Mandi HPC cluster) |

---

## Command-Line Usage

```
./pipeline -i <input_dir> [-o <output_dir>] [-t <threads>]
           [-s <sharpness>] [-b] [-c <csv_path>]

  -i  Input directory of .ppm images          (required)
  -o  Output directory for processed images   (default: output/)
  -t  OpenMP thread count                     (default: 4)
  -s  Sharpening strength in [0.5, 1.5]       (default: 1.0)
  -b  Benchmark mode: skip writing output PPMs
  -c  Write per-image timing data to CSV
```

**Examples:**

```bash
# Basic run
./pipeline -i data/ -o output/ -t 8

# Benchmark (no output writes, timing to CSV)
OMP_NUM_THREADS=16 ./pipeline -i data/ -t 16 -b -c results/timing_t16.csv

# Strong scaling sweep
for T in 1 2 4 8 16 32; do
    ./pipeline -i data/ -t $T -b -c results/timing_t${T}.csv
done
make plot
```

---

## Parallelisation Details

### OpenMP Directives Used

```c
/* Batch-level parallelism (outer) – image_load_batch, pipeline_run_batch */
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < count; i++) { ... }

/* Row-level parallelism (inner) – Gaussian blur, Sobel, sharpening */
#pragma omp parallel for schedule(static)
for (int y = 0; y < H; y++) { ... }

/* Histogram accumulation with array reduction (OpenMP 4.5+) */
#pragma omp parallel for reduction(+: hist[0:HIST_BINS]) schedule(static)
for (int i = 0; i < npix; i++) { hist[bin]++ ; }
```

### Thread Safety

- **Shared (read-only):** kernel arrays, image dimensions, input pixel data
- **Private per thread:** loop indices, intermediate accumulators
- **Reduction:** histogram bin array (`hist[0:256]`)
- **No locks:** array reduction avoids explicit `omp critical` sections

### Cache Considerations

The Gaussian blur uses a two-pass separable decomposition with separate
horizontal and vertical phases. Each row is processed in a contiguous memory
access pattern; `schedule(static)` gives each thread a contiguous chunk of
rows to minimise false sharing at cache-line boundaries.

---

## Experimental Results (Expected)

| Thread Count | Speedup (total) | Efficiency |
|:---:|:---:|:---:|
| 1  | 1.0×  | 100% |
| 2  | 1.9×  | ~95% |
| 4  | 3.6×  | ~90% |
| 8  | 6.5×  | ~81% |
| 16 | 10.8× | ~68% |
| 32 | 14.2× | ~44% |

> Amdahl serial fraction ≈ 4.5% (dominated by file I/O and histogram CDF
> computation).  Theoretical limit ≈ 22× at infinite threads.

---

## HPC Cluster Usage (IIT Mandi)

```bash
# Load modules (adjust to cluster's module system)
module load gcc/12.2.0 python/3.11

# Build
make all

# Submit strong scaling job (1–32 threads, 5 runs each)
sbatch scripts/slurm/strong_scaling.sh

# Submit weak scaling job
sbatch scripts/slurm/weak_scaling.sh

# Monitor
squeue -u $USER
```

Job logs: `results/slurm_strong_<jobid>.log`

---

## Generating Plots

After running benchmarks (locally or on HPC):

```bash
# If multiple runs per thread count, merge first:
python3 scripts/merge_strong_scaling.py --results results/

# Generate all scaling plots
make plot

# Generate per-stage image comparison
python3 python/plot_images.py \
    --input data/image_0000.ppm \
    --output_dir output/ \
    --out results/plots/stage_comparison.png
```

Plots written to `results/plots/`:
- `speedup.png` – Speedup vs thread count (per stage + total)
- `efficiency.png` – Parallel efficiency vs thread count
- `amdahl_overlay.png` – Amdahl theoretical vs observed
- `stage_breakdown.png` – Stacked bar chart of per-stage time
- `weak_scaling.png` – Weak scaling efficiency
- `stage_comparison.png` – Side-by-side pipeline stage images

---

## References

1. R. Chandra et al., *Parallel Programming in OpenMP*, Academic Press, 2001.
2. M. Quinn, *Parallel Programming in C with MPI and OpenMP*, McGraw-Hill, 2003.
3. R. C. Gonzalez & R. E. Woods, *Digital Image Processing*, 4th ed., Pearson, 2018.
4. L. R. Scott et al., *Scientific Parallel Computing*, Princeton UP, 2005.
5. OpenMP ARB, *OpenMP API Specification v5.2*, 2021. https://www.openmp.org
6. Intel, *OpenMP Best Practices – False Sharing*. https://www.intel.com

---

## License

MIT – see [LICENSE](LICENSE).

---

*ME 522 Course Project · IIT Mandi · Parallel Image Processing Pipeline using OpenMP*
