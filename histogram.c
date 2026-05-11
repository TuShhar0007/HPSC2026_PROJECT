#include "histogram.h"
#include "image_io.h"

#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* ══════════════════════════════════════════════════════════════════════════
 * histogram_compute
 * ══════════════════════════════════════════════════════════════════════════
 *
 * Parallelised with the array-reduction extension introduced in OpenMP 4.5.
 * Each thread accumulates into its own private copy of hist[]; the runtime
 * sums the copies at the implicit barrier at the end of the region.
 *
 * The reduction clause  reduction(+: hist[0:HIST_BINS])  is standard
 * OpenMP 4.5+.  If your compiler only supports OpenMP 3.1, replace with
 * a manual critical section or per-thread histograms merged afterwards.
 */
void histogram_compute(const Image *src, long long hist[HIST_BINS]) {
    memset(hist, 0, HIST_BINS * sizeof(long long));

    int npix = src->width * src->height;
    const float *data = src->data;

    #pragma omp parallel for reduction(+: hist[0:HIST_BINS]) schedule(static)
    for (int i = 0; i < npix; i++) {
        int bin = (int)(data[i] * (HIST_BINS - 1) + 0.5f);
        if (bin < 0)          bin = 0;
        if (bin >= HIST_BINS) bin = HIST_BINS - 1;
        hist[bin]++;
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 * histogram_equalize
 * ══════════════════════════════════════════════════════════════════════════
 *
 * 1. Compute histogram  (parallel reduction)
 * 2. Build cumulative distribution function  (serial – O(HIST_BINS))
 * 3. Apply equalization LUT to each pixel  (parallel)
 */
Image *histogram_equalize(const Image *src) {
    if (!src || src->channels != 1) return NULL;   /* grayscale only */

    int npix = src->width * src->height;

    /* Step 1: histogram */
    long long hist[HIST_BINS];
    histogram_compute(src, hist);

    /* Step 2: CDF → equalization LUT (serial, negligible cost) */
    long long cdf[HIST_BINS];
    cdf[0] = hist[0];
    for (int b = 1; b < HIST_BINS; b++)
        cdf[b] = cdf[b - 1] + hist[b];

    long long cdf_min = 0;
    for (int b = 0; b < HIST_BINS; b++) {
        if (cdf[b] > 0) { cdf_min = cdf[b]; break; }
    }

    float lut[HIST_BINS];
    float denom = (float)(npix - cdf_min);
    for (int b = 0; b < HIST_BINS; b++) {
        if (denom <= 0.0f) { lut[b] = 0.0f; continue; }
        lut[b] = (float)(cdf[b] - cdf_min) / denom;
        if (lut[b] < 0.0f) lut[b] = 0.0f;
        if (lut[b] > 1.0f) lut[b] = 1.0f;
    }

    /* Step 3: apply LUT */
    Image *dst = image_alloc(src->width, src->height, 1);
    if (!dst) return NULL;

    const float *in  = src->data;
    float       *out = dst->data;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < npix; i++) {
        int bin = (int)(in[i] * (HIST_BINS - 1) + 0.5f);
        if (bin < 0)          bin = 0;
        if (bin >= HIST_BINS) bin = HIST_BINS - 1;
        out[i] = lut[bin];
    }

    return dst;
}
