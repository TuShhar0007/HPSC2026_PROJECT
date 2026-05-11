/*
 * main.c – Parallel Image Processing Pipeline (ME 522, IIT Mandi)
 *
 * Usage:
 *   ./pipeline -i <input_dir> -o <output_dir> [-t <threads>] [-s <sharpness>]
 *              [-b]          (benchmark mode: suppress output writes)
 *              [-c <csv>]    (write per-image timing CSV)
 *
 * Environment:
 *   OMP_NUM_THREADS overrides -t if set externally.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "image_io.h"
#include "pipeline.h"

/* ── Argument defaults ──────────────────────────────────────────────────── */
#define DEFAULT_THREADS   4
#define DEFAULT_SHARPNESS 1.0f
#define DEFAULT_OUT_DIR   "output"

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s -i <input_dir> [-o <output_dir>] [-t <threads>]\n"
        "          [-s <sharpness>] [-b] [-c <csv_path>]\n"
        "\n"
        "  -i  Input directory containing .ppm images   (required)\n"
        "  -o  Output directory for processed images    (default: output/)\n"
        "  -t  Number of OpenMP threads                 (default: %d)\n"
        "  -s  Sharpening strength (0.5–1.5)            (default: %.1f)\n"
        "  -b  Benchmark mode: skip writing output PPMs\n"
        "  -c  Write per-image timing data to CSV file\n",
        prog, DEFAULT_THREADS, DEFAULT_SHARPNESS);
}

/* ── Helpers ────────────────────────────────────────────────────────────── */

static void ensure_dir(const char *path) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", path);
    system(cmd);
}

static void write_timing_csv(const char *path,
                              PipelineTiming *timings, int count) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }
    fprintf(f, "image_idx,t_io_load,t_gaussian,t_sobel,t_sharpen,t_histogram,t_total\n");
    for (int i = 0; i < count; i++) {
        fprintf(f, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                i,
                timings[i].t_io_load,
                timings[i].t_gaussian,
                timings[i].t_sobel,
                timings[i].t_sharpen,
                timings[i].t_histogram,
                timings[i].t_total);
    }
    fclose(f);
    printf("Timing CSV written to: %s\n", path);
}

/* ── Main ───────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *input_dir  = NULL;
    const char *output_dir = DEFAULT_OUT_DIR;
    const char *csv_path   = NULL;
    int   nthreads    = DEFAULT_THREADS;
    float sharpness   = DEFAULT_SHARPNESS;
    int   bench_mode  = 0;

    /* ── Parse arguments ─────────────────────────────────────────────── */
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-i") && i+1 < argc) input_dir  = argv[++i];
        else if (!strcmp(argv[i], "-o") && i+1 < argc) output_dir = argv[++i];
        else if (!strcmp(argv[i], "-t") && i+1 < argc) nthreads   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i+1 < argc) sharpness  = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i+1 < argc) csv_path   = argv[++i];
        else if (!strcmp(argv[i], "-b"))                bench_mode = 1;
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    if (!input_dir) {
        fprintf(stderr, "Error: -i <input_dir> is required.\n");
        usage(argv[0]); return 1;
    }

    printf("═══════════════════════════════════════════════════════════\n");
    printf("  ME 522 – Parallel Image Processing Pipeline (OpenMP)\n");
    printf("  IIT Mandi | OpenMP threads: %d | Sharpness: %.2f\n",
           nthreads, sharpness);
    printf("═══════════════════════════════════════════════════════════\n");

    /* ── Load batch ──────────────────────────────────────────────────── */
    double t_load_start = omp_get_wtime();
    int count = 0;
    Image **batch = image_load_batch(input_dir, &count);
    double t_load_end = omp_get_wtime();

    if (!batch || count == 0) {
        fprintf(stderr, "No .ppm images found in: %s\n", input_dir);
        return 1;
    }
    printf("Loaded %d images in %.3f s\n", count, t_load_end - t_load_start);

    /* ── Allocate result and timing arrays ───────────────────────────── */
    PipelineResult **results  = calloc(count, sizeof(PipelineResult *));
    PipelineTiming  *timings  = calloc(count, sizeof(PipelineTiming));

    /* ── Run pipeline ─────────────────────────────────────────────────── */
    double t_pipe_start = omp_get_wtime();
    pipeline_run_batch(batch, count, sharpness, nthreads, results, timings);
    double t_pipe_end   = omp_get_wtime();

    double wall_total = t_pipe_end - t_pipe_start;
    printf("Pipeline complete: %.3f s wall time (%.1f img/s)\n",
           wall_total, count / wall_total);

    /* ── Aggregate per-stage timings ─────────────────────────────────── */
    double sum_gauss = 0, sum_sobel = 0, sum_sharp = 0, sum_hist = 0;
    for (int i = 0; i < count; i++) {
        sum_gauss += timings[i].t_gaussian;
        sum_sobel += timings[i].t_sobel;
        sum_sharp += timings[i].t_sharpen;
        sum_hist  += timings[i].t_histogram;
    }
    printf("\n── Per-stage aggregate (sum across all images) ──\n");
    printf("  Gaussian blur     : %.3f s\n", sum_gauss);
    printf("  Sobel edges       : %.3f s\n", sum_sobel);
    printf("  Unsharp mask      : %.3f s\n", sum_sharp);
    printf("  Histogram eq.     : %.3f s\n", sum_hist);
    printf("  Total pipeline    : %.3f s\n", wall_total);

    /* ── Optional: write output PPMs ─────────────────────────────────── */
    if (!bench_mode) {
        ensure_dir(output_dir);
        char path[512];
        for (int i = 0; i < count; i++) {
            if (!results[i]) continue;
            snprintf(path, sizeof(path), "%s/blur_%04d.ppm",     output_dir, i);
            image_save_ppm(results[i]->blurred,   path);
            snprintf(path, sizeof(path), "%s/edges_%04d.ppm",    output_dir, i);
            image_save_ppm(results[i]->edges,     path);
            snprintf(path, sizeof(path), "%s/sharp_%04d.ppm",    output_dir, i);
            image_save_ppm(results[i]->sharpened, path);
            snprintf(path, sizeof(path), "%s/eq_%04d.ppm",       output_dir, i);
            image_save_ppm(results[i]->equalized, path);
        }
        printf("\nOutput images written to: %s/\n", output_dir);
    }

    /* ── Optional: write CSV ─────────────────────────────────────────── */
    if (csv_path) write_timing_csv(csv_path, timings, count);

    /* ── Cleanup ──────────────────────────────────────────────────────── */
    for (int i = 0; i < count; i++) {
        if (results[i]) {
            pipeline_result_free(results[i]);
            free(results[i]);
        }
    }
    free(results);
    free(timings);
    image_free_batch(batch, count);

    return 0;
}
