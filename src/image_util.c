/*
  * ESPRESSIF MIT License
  *
  * Copyright (c) 2018 <ESPRESSIF SYSTEMS (SHANGHAI) PTE LTD>
  *
  * Permission is hereby granted for use on ESPRESSIF SYSTEMS products only, in which case,
  * it is free of charge, to any person obtaining a copy of this software and associated
  * documentation files (the "Software"), to deal in the Software without restriction, including
  * without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
  * and/or sell copies of the Software, and to permit persons to whom the Software is furnished
  * to do so, subject to the following conditions:
  *
  * The above copyright notice and this permission notice shall be included in all copies or
  * substantial portions of the Software.
  *
  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
  * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
  * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
  * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  *
  */

#include "image_util.h"

void image_zoom_in_twice(uint8_t *dimage,
                         int dw,
                         int dh,
                         int dc,
                         uint8_t *simage,
                         int sw,
                         int sc)
{
    for (int dyi = 0; dyi < dh; dyi++)
    {
        int _di = dyi * dw;

        int _si0 = dyi * 2 * sw;
        int _si1 = _si0 + sw;

        for (int dxi = 0; dxi < dw; dxi++)
        {
            int di = (_di + dxi) * dc;
            int si0 = (_si0 + dxi * 2) * sc;
            int si1 = (_si1 + dxi * 2) * sc;

            if (1 == dc)
            {
                dimage[di] = (uint8_t)((simage[si0] + simage[si0 + 1] + simage[si1] + simage[si1 + 1]) >> 2);
            }
            else if (3 == dc)
            {
                dimage[di] = (uint8_t)((simage[si0] + simage[si0 + 3] + simage[si1] + simage[si1 + 3]) >> 2);
                dimage[di + 1] = (uint8_t)((simage[si0 + 1] + simage[si0 + 4] + simage[si1 + 1] + simage[si1 + 4]) >> 2);
                dimage[di + 2] = (uint8_t)((simage[si0 + 2] + simage[si0 + 5] + simage[si1 + 2] + simage[si1 + 5]) >> 2);
            }
            else
            {
                for (int dci = 0; dci < dc; dci++)
                {
                    dimage[di + dci] = (uint8_t)((simage[si0 + dci] + simage[si0 + 3 + dci] + simage[si1 + dci] + simage[si1 + 3 + dci] + 2) >> 2);
                }
            }
        }
    }
    return;
}

void image_resize_linear(uint8_t *dst_image, uint8_t *src_image, int dst_w, int dst_h, int dst_c, int src_w, int src_h)
{
    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;

    int dst_stride = dst_c * dst_w;
    int src_stride = dst_c * src_w;

    if (fabs(scale_x - 2) <= 1e-6 && fabs(scale_y - 2) <= 1e-6)
    {
        image_zoom_in_twice(
            dst_image,
            dst_w,
            dst_h,
            dst_c,
            src_image,
            src_w,
            dst_c);
    }
    else
    {
        for (int y = 0; y < dst_h; y++)
        {
            float fy[2];
            fy[0] = (float)((y + 0.5) * scale_y - 0.5); // y
            int src_y = (int)fy[0];                     // y1
            fy[0] -= src_y;                             // y - y1
            fy[1] = 1 - fy[0];                          // y2 - y
            src_y = DL_IMAGE_MAX(0, src_y);
            src_y = DL_IMAGE_MIN(src_y, src_h - 2);

            for (int x = 0; x < dst_w; x++)
            {
                float fx[2];
                fx[0] = (float)((x + 0.5) * scale_x - 0.5); // x
                int src_x = (int)fx[0];                     // x1
                fx[0] -= src_x;                             // x - x1
                if (src_x < 0)
                {
                    fx[0] = 0;
                    src_x = 0;
                }
                if (src_x > src_w - 2)
                {
                    fx[0] = 0;
                    src_x = src_w - 2;
                }
                fx[1] = 1 - fx[0]; // x2 - x

                for (int c = 0; c < dst_c; c++)
                {
                    dst_image[y * dst_stride + x * dst_c + c] = round(
                        src_image[src_y * src_stride + src_x * dst_c + c] * fx[1] * fy[1] 
                        + src_image[src_y * src_stride + (src_x + 1) * dst_c + c] * fx[0] * fy[1] 
                        + src_image[(src_y + 1) * src_stride + src_x * dst_c + c] * fx[1] * fy[0] 
                        + src_image[(src_y + 1) * src_stride + (src_x + 1) * dst_c + c] * fx[0] * fy[0]
                    );
                }
            }
        }
    }
}

void image_cropper(uint8_t *rot_data, uint8_t *src_data, int rot_w, int rot_h, int rot_c, int src_w, int src_h, float rotate_angle, float ratio, float *center)
{
    int rot_stride = rot_w * rot_c;
    float rot_w_start = 0.5f - (float)rot_w / 2;
    float rot_h_start = 0.5f - (float)rot_h / 2;

    //rotate_angle must be radius
    float si = sin(rotate_angle);
    float co = cos(rotate_angle);

    int src_stride = src_w * rot_c;

    for (int y = 0; y < rot_h; y++)
    {
        for (int x = 0; x < rot_w; x++)
        {
            float xs, ys, xr, yr;
            xs = ratio * (rot_w_start + x);
            ys = ratio * (rot_h_start + y);

            xr = xs * co + ys * si;
            yr = -xs * si + ys * co;

            float fy[2];
            fy[0] = center[1] + yr; // y
            int src_y = (int)fy[0]; // y1
            fy[0] -= src_y;         // y - y1
            fy[1] = 1 - fy[0];      // y2 - y
            src_y = DL_IMAGE_MAX(0, src_y);
            src_y = DL_IMAGE_MIN(src_y, src_h - 2);

            float fx[2];
            fx[0] = center[0] + xr; // x
            int src_x = (int)fx[0]; // x1
            fx[0] -= src_x;         // x - x1
            if (src_x < 0)
            {
                fx[0] = 0;
                src_x = 0;
            }
            if (src_x > src_w - 2)
            {
                fx[0] = 0;
                src_x = src_w - 2;
            }
            fx[1] = 1 - fx[0]; // x2 - x

            for (int c = 0; c < rot_c; c++)
            {
                rot_data[y * rot_stride + x * rot_c + c] = round(src_data[src_y * src_stride + src_x * rot_c + c] * fx[1] * fy[1] + src_data[src_y * src_stride + (src_x + 1) * rot_c + c] * fx[0] * fy[1] + src_data[(src_y + 1) * src_stride + src_x * rot_c + c] * fx[1] * fy[0] + src_data[(src_y + 1) * src_stride + (src_x + 1) * rot_c + c] * fx[0] * fy[0]);
            }
        }
    }
}

void image_abs_diff(uint8_t *dst, uint8_t *src1, uint8_t *src2, int count)
{
    while (count > 0)
    {
        *dst = (uint8_t)abs((int)*src1 - (int)*src2);
        dst++;
        src1++;
        src2++;
        count--;
    }
}

void image_threshold(uint8_t *dst, uint8_t *src, int threshold, int value, int count, en_threshold_mode mode)
{
    int l_val = 0;
    int r_val = 0;
    switch (mode)
    {
    case BINARY:
        r_val = value;
        break;
    default:
        break;
    }
    while (count > 0)
    {
        *dst = (*src > threshold) ? r_val : l_val;

        dst++;
        src++;
        count--;
    }
}

void image_kernel_get_min(uint8_t *dst, uint8_t *src, int w, int h, int c, int stride)
{
    uint8_t min1 = 255;
    uint8_t min2 = 255;
    uint8_t min3 = 255;

    if (c == 3)
    {
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                if (src[0] < min1)
                    min1 = src[0];
                if (src[1] < min2)
                    min2 = src[1];
                if (src[2] < min3)
                    min3 = src[2];
                src += 3;
            }
            src += stride - w * 3;
        }
        dst[0] = min1;
        dst[1] = min2;
        dst[2] = min3;
    }
    else if (c == 1)
    {
        for (int j = 0; j < h; j++)
        {
            for (int i = 0; i < w; i++)
            {
                if (src[0] < min1)
                    min1 = src[0];
                src += 1;
            }
            src += stride - w;
        }
        dst[0] = min1;
    }
    else
    {
    }
}

// By default 3x3 Kernel, so padding is 2

void image_erode(uint8_t *dst, uint8_t *src, int src_w, int src_h, int src_c)
{
    int stride = src_w * src_c;

    // 1st row, 1st col
    image_kernel_get_min(dst, src, 2, 2, src_c, stride);
    dst += src_c;

    // 1st row
    for (int i = 1; i < src_w - 1; i++)
    {
        image_kernel_get_min(dst, src, 3, 2, src_c, stride);
        dst += src_c;
        src += src_c;
    }

    // 1st row, last col
    image_kernel_get_min(dst, src, 2, 2, src_c, stride);
    dst += src_c;
    src -= src_c * (src_w - 2);

    for (int j = 1; j < src_h - 1; j++)
    {
        // 1st col
        image_kernel_get_min(dst, src, 2, 3, src_c, stride);
        dst += src_c;

        for (int i = 1; i < src_w - 1; i++)
        {
            image_kernel_get_min(dst, src, 3, 3, src_c, stride);
            dst += src_c;
            src += src_c;
        }

        // last col
        image_kernel_get_min(dst, src, 2, 3, src_c, stride);
        dst += src_c;
        src += src_c * 2;
    }

    // last row
    image_kernel_get_min(dst, src, 2, 2, src_c, stride);
    dst += src_c;

    for (int i = 1; i < src_w - 1; i++)
    {
        image_kernel_get_min(dst, src, 3, 2, src_c, stride);
        dst += src_c;
        src += src_c;
    }

    // last row, last col
    image_kernel_get_min(dst, src, 2, 2, src_c, stride);
}

void matrix_print(Matrix *m)
{
    printf("Matrix: %dx%d\n", m->h, m->w);
    for (int i = 0; i < m->h; i++)
    {
        for (int j = 0; j < m->w; j++)
        {
            printf("%f ", m->array[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

Matrix *malloc_rand_matrix(int h, int w, int thresh)
{
    Matrix *m = matrix_alloc(h, w);
    unsigned int seed = esp_timer_get_time();
    srand(seed);
    for (int i = 0; i < m->h; i++)
    {
        for (int j = 0; j < m->w; j++)
        {
            m->array[i][j] = rand() % thresh;
        }
    }
    return m;
}

Matrix *get_affine_transform(float *srcx, float *srcy, float *dstx, float *dsty)
{
    Matrix *m = matrix_alloc(2, 3);
    float A[3][2] = {0};
    float Ainv[3][3] = {0};
    for (int i = 0; i < 3; i++)
    {
        A[i][0] = srcx[i];
        A[i][1] = srcy[i];
    }
    float Adet = (A[0][0] * A[1][1] + A[0][1] * A[2][0] + A[1][0] * A[2][1]) - (A[2][0] * A[1][1] + A[1][0] * A[0][1] + A[0][0] * A[2][1]);
    if (Adet == 0)
    {
        printf("the src is linearly dependent\n");
        return NULL;
    }
    Ainv[0][0] = (A[1][1] - A[2][1]) / Adet;
    Ainv[0][1] = (A[2][1] - A[0][1]) / Adet;
    Ainv[0][2] = (A[0][1] - A[1][1]) / Adet;
    Ainv[1][0] = (A[2][0] - A[1][0]) / Adet;
    Ainv[1][1] = (A[0][0] - A[2][0]) / Adet;
    Ainv[1][2] = (A[1][0] - A[0][0]) / Adet;
    Ainv[2][0] = (A[1][0] * A[2][1] - A[2][0] * A[1][1]) / Adet;
    Ainv[2][1] = (A[2][0] * A[0][1] - A[0][0] * A[2][1]) / Adet;
    Ainv[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / Adet;

    for (int i = 0; i < 3; i++)
    {
        m->array[0][i] = Ainv[i][0] * dstx[0] + Ainv[i][1] * dstx[1] + Ainv[i][2] * dstx[2];
        m->array[1][i] = Ainv[i][0] * dsty[0] + Ainv[i][1] * dsty[1] + Ainv[i][2] * dsty[2];
    }
    return m;
}

Matrix *get_inv_affine_matrix(Matrix *m)
{
    Matrix *minv = matrix_alloc(2, 3);
    float mdet = (m->array[0][0]) * (m->array[1][1]) - (m->array[1][0]) * (m->array[0][1]);
    if (mdet == 0)
    {
        printf("the matrix m is wrong !\n");
        return NULL;
    }

    minv->array[0][0] = m->array[1][1] / mdet;
    minv->array[0][1] = -(m->array[0][1] / mdet);
    minv->array[0][2] = ((m->array[0][1]) * (m->array[1][2]) - (m->array[0][2]) * (m->array[1][1])) / mdet;
    minv->array[1][0] = -(m->array[1][0]) / mdet;
    minv->array[1][1] = (m->array[0][0]) / mdet;
    minv->array[1][2] = ((m->array[0][2]) * (m->array[1][0]) - (m->array[0][0]) * (m->array[1][2])) / mdet;
    return minv;
}

Matrix *get_inverse_matrix(Matrix *m)
{
    if (m->w != m->h)
    {
        printf("the input is not a square matrix !\n");
        return NULL;
    }

    Matrix *matw = matrix_alloc(m->h, 2 * (m->w));
    Matrix *inv = matrix_alloc(m->h, m->w);
    matrixType **w = matw->array;
    float eps = 1e-6;

    for (int i = 0; i < matw->h; i++)
    {
        for (int j = 0; j < m->w; j++)
        {
            w[i][j] = m->array[i][j];
        }
        w[i][(m->w) + i] = 1;
    }

    for (int i = 0; i < matw->h; i++)
    {
        if (fabs(w[i][i]) < eps)
        {
            int j;
            for (j = i + 1; j < matw->h; j++)
            {
                if (fabs(w[j][i]) > eps)
                    break;
            }
            if (j == matw->h)
            {
                printf("This matrix is irreversible!\n");
                return NULL;
            }
            for (int k = i; k < matw->w; k++)
            {
                w[i][k] += w[j][k];
            }
        }
        float factor = w[i][i];
        for (int k = i; k < matw->w; k++)
        {
            w[i][k] /= factor;
        }
        for (int k = i + 1; k < matw->h; k++)
        {
            factor = -w[k][i];
            for (int l = i; l < matw->w; l++)
            {
                w[k][l] += (factor * w[i][l]);
            }
        }
    }
    for (int i = (matw->h) - 1; i > 0; i--)
    {
        for (int j = i - 1; j >= 0; j--)
        {
            float factor = -w[j][i];
            for (int k = i; k < matw->w; k++)
            {
                w[j][k] += (factor * w[i][k]);
            }
        }
    }
    for (int i = 0; i < m->h; i++)
    {
        for (int j = 0; j < m->w; j++)
        {
            inv->array[i][j] = w[i][(m->h) + j];
        }
    }
    return inv;
}

Matrix *get_perspective_transform(float *srcx, float *srcy, float *dstx, float *dsty)
{
    Matrix *m = matrix_alloc(3, 3);
    Matrix *A = matrix_alloc(8, 8);

    for (int i = 0; i < 4; i++)
    {
        A->array[i][0] = srcx[i];
        A->array[i][1] = srcy[i];
        A->array[i][2] = 1;
        A->array[i][3] = 0;
        A->array[i][4] = 0;
        A->array[i][5] = 0;
        A->array[i][6] = -dstx[i] * srcx[i];
        A->array[i][7] = -dstx[i] * srcy[i];
    }
    for (int i = 4; i < 8; i++)
    {
        A->array[i][0] = 0;
        A->array[i][1] = 0;
        A->array[i][2] = 0;
        A->array[i][3] = srcx[i - 4];
        A->array[i][4] = srcy[i - 4];
        A->array[i][5] = 1;
        A->array[i][6] = -dsty[i - 4] * srcx[i - 4];
        A->array[i][7] = -dsty[i - 4] * srcy[i - 4];
    }
    Matrix *Ainv = get_inverse_matrix(A);
    for (int i = 0; i < 8; i++)
    {
        m->array[i / 3][i % 3] = (((Ainv->array[i][0]) * dstx[0]) + ((Ainv->array[i][1]) * dstx[1]) + ((Ainv->array[i][2]) * dstx[2]) + ((Ainv->array[i][3]) * dstx[3]) +
                                  ((Ainv->array[i][4]) * dsty[0]) + ((Ainv->array[i][5]) * dsty[1]) + ((Ainv->array[i][6]) * dsty[2]) + ((Ainv->array[i][7]) * dsty[3]));
    }
    m->array[2][2] = 1;
    return m;
}