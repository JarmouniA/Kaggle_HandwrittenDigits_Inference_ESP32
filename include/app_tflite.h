/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef _APP_TFLITE_H_
#define _APP_TFLITE_H_

#include "esp_log.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t app_tflite_init(void);
void tf_start_inference(void);
void tf_stop_inference();

#ifdef __cplusplus
}
#endif

#endif  // _APP_TFLITE_H_
