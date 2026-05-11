#include "filters.h"
#include "image_io.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* ══════════════════════════════════════════════════════════════════════════
 * Stage 2 – Gaussian Blur (5×5 separable kernel, σ ≈ 1.0)
 * ══════════════════════════════════════════════════════════════════════════
 *
 * Separable: first convolve each row (horizontal pass) into a temp buffer,
 * then convolve each column (vertical pass) into the output.  Each pass is
 * parallelised across rows / columns with OpenMP.
 *
 * Kernel weights (unnormalized): [ 1  4  6  4  1 ] / 16
 */

static const float GAUSS5[5] = { 1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16 };

Image *filter_gaussian_blur(const Image *src) {
    int W = src->width, H = src->height, C = src->channels;
    Image *tmp = image_alloc(W, H, C);   /* intermediate after horiz pass */
    Image *dst = image_alloc(W, H, C);
    if (!tmp || !dst) { image_free(tmp); image_free(dst); return NULL; }

    /* ── Horizontal pass ─────────────────────────────────────────────── */
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C; c++) {
                float acc = 0.0f;
                for (int k = -2; k <= 2; k++) {
                    int xx = x + k;
                    if (xx < 0) xx = 0;
                    if (xx >= W) xx = W - 1;
                    acc += GAUSS5[k + 2] * src->data[(y * W + xx) * C + c];
                }
                tmp->data[(y * W + x) * C + c] = acc;
            }
        }
    }

    /* ── Vertical pass ───────────────────────────────────────────────── */
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C; c++) {
                float acc = 0.0f;
                for (int k = -2; k <= 2; k++) {
                    int yy = y + k;
                    if (yy < 0) yy = 0;
                    if (yy >= H) yy = H - 1;
                    acc += GAUSS5[k + 2] * tmp->data[(yy * W + x) * C + c];
                }
                dst->data[(y * W + x) * C + c] = acc;
            }
        }
    }

    image_free(tmp);
    return dst;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Stage 3 – Sobel Edge Detection
 * ══════════════════════════════════════════════════════════════════════════
 *
 * Converts to grayscale, then applies 3×3 Sobel operators Gx and Gy.
 * Output is a single-channel image with edge magnitude ∈ [0, 1].
 *
 *        Gx = [-1  0  1]        Gy = [-1 -2 -1]
 *             [-2  0  2]             [ 0  0  0]
 *             [-1  0  1]             [ 1  2  1]
 */

Image *filter_sobel(const Image *src) {
    /* Work on a grayscale copy */
    Image *gray = (src->channels == 1) ? image_clone(src) : image_to_gray(src);
    if (!gray) return NULL;

    int W = gray->width, H = gray->height;
    Image *dst = image_alloc(W, H, 1);
    if (!dst) { image_free(gray); return NULL; }

    #pragma omp parallel for schedule(static)
    for (int y = 1; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {

            /* Sample 3×3 neighbourhood */
            float p00 = gray->data[(y-1)*W + (x-1)];
            float p01 = gray->data[(y-1)*W + (x  )];
            float p02 = gray->data[(y-1)*W + (x+1)];
            float p10 = gray->data[(y  )*W + (x-1)];
            /* p11 not needed for Sobel */
            float p12 = gray->data[(y  )*W + (x+1)];
            float p20 = gray->data[(y+1)*W + (x-1)];
            float p21 = gray->data[(y+1)*W + (x  )];
            float p22 = gray->data[(y+1)*W + (x+1)];

            float gx = -p00 + p02 - 2*p10 + 2*p12 - p20 + p22;
            float gy = -p00 - 2*p01 - p02 + p20 + 2*p21 + p22;

            dst->data[y * W + x] = clampf(sqrtf(gx*gx + gy*gy) / 4.0f);
        }
    }

    image_free(gray);
    return dst;
}

/* ══════════════════════════════════════════════════════════════════════════
 * Stage 4 – Unsharp Mask Sharpening
 * ══════════════════════════════════════════════════════════════════════════
 *
 * sharpened = src + strength * (src - blurred)
 *
 * strength ≈ 0.5–1.5 produces natural-looking enhancement.
 */

Image *filter_sharpen(const Image *src, float strength) {
    Image *blurred = filter_gaussian_blur(src);
    if (!blurred) return NULL;

    int W = src->width, H = src->height, C = src->channels;
    Image *dst = image_alloc(W, H, C);
    if (!dst) { image_free(blurred); return NULL; }

    size_t total = (size_t)W * H * C;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; i++) {
        float detail = src->data[i] - blurred->data[i];
        dst->data[i] = clampf(src->data[i] + strength * detail);
    }

    image_free(blurred);
    return dst;
}
