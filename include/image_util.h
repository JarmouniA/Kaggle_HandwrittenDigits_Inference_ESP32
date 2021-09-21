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
#ifndef _IMAGE_UTIL_H_
#define _IMAGE_UTIL_H_


#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#include "esp_timer.h"

#define LANDMARKS_NUM (10)

#define MAX_VALID_COUNT_PER_IMAGE (30)

#define DL_IMAGE_MIN(A, B) ((A) < (B) ? (A) : (B))
#define DL_IMAGE_MAX(A, B) ((A) < (B) ? (B) : (A))

#define RGB565_MASK_RED 0xF800
#define RGB565_MASK_GREEN 0x07E0
#define RGB565_MASK_BLUE 0x001F

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum
    {
        BINARY, /*!< binary */
    } en_threshold_mode;
    
    /**
     * @brief Resize an image to half size 
     * 
     * @param dimage      The output image
     * @param dw          Width of the output image
     * @param dh          Height of the output image
     * @param dc          Channel of the output image
     * @param simage      Source image
     * @param sw          Width of the source image
     * @param sc          Channel of the source image
     */
    void image_zoom_in_twice(uint8_t *dimage,
                             int dw,
                             int dh,
                             int dc,
                             uint8_t *simage,
                             int sw,
                             int sc);

    /**
     * @brief Resize the image in RGB888 format via bilinear interpolation
     * 
     * @param dst_image    The output image
     * @param src_image    Source image
     * @param dst_w        Width of the output image
     * @param dst_h        Height of the output image
     * @param dst_c        Channel of the output image
     * @param src_w        Width of the source image
     * @param src_h        Height of the source image
     */
    void image_resize_linear(uint8_t *dst_image, uint8_t *src_image, int dst_w, int dst_h, int dst_c, int src_w, int src_h);

    /**
     * @brief Crop， rotate and zoom the image in RGB888 format, 
     * 
     * @param corp_image       The output image
     * @param src_image        Source image
     * @param rotate_angle     Rotate angle
     * @param ratio            scaling ratio
     * @param center           Center of rotation
     */
    void image_cropper(uint8_t *corp_image, uint8_t *src_image, int dst_w, int dst_h, int dst_c, int src_w, int src_h, float rotate_angle, float ratio, float *center);

    /**
     * @brief Get the pixel difference of two images
     * 
     * @param dst       The output pixel difference
     * @param src1      Input image 1
     * @param src2      Input image 2
     * @param count     Total pixels of the input image
     */
    void image_abs_diff(uint8_t *dst, uint8_t *src1, uint8_t *src2, int count);

    /**
     * @brief Binarize an image to 0 and value. 
     * 
     * @param dst           The output image
     * @param src           Source image
     * @param threshold     Threshold of binarization
     * @param value         The value of binarization
     * @param count         Total pixels of the input image
     * @param mode          Threshold mode
     */
    void image_threshold(uint8_t *dst, uint8_t *src, int threshold, int value, int count, en_threshold_mode mode);

    /**
     * @brief Erode the image
     * 
     * @param dst          The output image
     * @param src          Source image
     * @param src_w        Width of the source image
     * @param src_h        Height of the source image
     * @param src_c        Channel of the source image
     */
    void image_erode(uint8_t *dst, uint8_t *src, int src_w, int src_h, int src_c);

    typedef float matrixType;
    typedef struct
    {
        int w;              /*!< width */
        int h;              /*!< height */
        matrixType **array; /*!< array */
    } Matrix;

    /**
     * @brief Allocate a 2d matrix
     * 
     * @param h                Height of matrix
     * @param w                Width of matrix
     * @return Matrix*         2d matrix
     */
    Matrix *matrix_alloc(int h, int w);

    /**
     * @brief Free a 2d matrix
     * 
     * @param m    2d matrix 
     */
    void matrix_free(Matrix *m);

    /**
     * @brief Get the affine transformation matrix
     * 
     * @param srcx          Source x coordinates
     * @param srcy          Source y coordinates
     * @param dstx          Destination x coordinates
     * @param dsty          Destination y coordinates
     * @return Matrix*      The resulting transformation matrix
     */
    Matrix *get_affine_transform(float *srcx, float *srcy, float *dstx, float *dsty);

    void image_kernel_get_min(uint8_t *dst, uint8_t *src, int w, int h, int c, int stride);

    void image_threshold(uint8_t *dst, uint8_t *src, int threshold, int value, int count, en_threshold_mode mode);

    Matrix *malloc_rand_matrix(int h, int w, int thresh);

    void matrix_print(Matrix *m);

    Matrix *get_inv_affine_matrix(Matrix *m);

    Matrix *get_inverse_matrix(Matrix *m);

    Matrix *get_perspective_transform(float *srcx, float *srcy, float *dstx, float *dsty);

#ifdef __cplusplus
}
#endif

#endif // _H_IMAGE_UTIL_H_