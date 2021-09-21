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
/*
#include <cstdint>

#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

//#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "app_tflite.h"

#include "image_provider.h"
#include "model_settings.h"
#include "model.h"

namespace {
  // setting up logging
  tflite::ErrorReporter* error_reporter = nullptr;

  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;

  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 12 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

  
static bool isRunning = false;
//static QueueHandle_t  predictionQueue =NULL;

static const char *TAG = "App_TFLite";

esp_err_t app_tflite_init(void)
{
  tflite::InitializeTarget();

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);

  // Check the model to ensure its schema version is compatible with 
  // the version we are using
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return ESP_FAIL;
  }

  // The AllOpsResolver loads all of the operations available in 
  // TensorFlow Lite for Microcontrollers.
  // Used by the interpreter to access the operations 
  // that are used by the model.
  //
  //static tflite::AllOpsResolver op_resolver;
  
  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroMutableOpResolver<6> op_resolver;
  op_resolver.AddConv2D();
  op_resolver.AddRelu();
  op_resolver.AddMaxPool2D();
  op_resolver.AddReshape();
  op_resolver.AddFullyConnected();
  op_resolver.AddSoftmax();

  // Instantiate an interpreter to run the model with.
  ESP_LOGI(TAG, "Instantiating an interpreter");
  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize, error_reporter
  );
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  ESP_LOGI(TAG, "Allocating Tensors memory");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return ESP_FAIL;
  }
  ESP_LOGI(TAG, "Tensors memory allocated successfully");

  // Obtain pointers to the model's input tensors.
  // 0 represents the first (and only) input tensor.
  input = interpreter->input(0);

  isRunning = false;

  // This queue will hold the index of Max prediction
  //predictionQueue = xQueueCreate(5, sizeof(const char*));
  return ESP_OK;
}

void tf_start_inference(void) 
{
  ESP_LOGI(TAG, "Starting inference");

  if(isRunning)
    goto exit;
  else
    isRunning = true;

  // integers buffer for image capture by camera
  temp_buffer = (uint8_t*) malloc((fb_height * fb_width) * sizeof(uint8_t));
  if (temp_buffer == NULL) {
        ESP_LOGE(TAG, "Temp_buffer was not allocated.");
  }

  while(isRunning)
  {
    // Get normalized image array from image_provider.
    float* tmp_buffer = (float*) malloc(kMaxImageSize * sizeof(float));
    if (tmp_buffer == NULL) {
        ESP_LOGE(TAG, "Tmp_buffer not allocated.");
        continue;
    }
    ESP_LOGI(TAG, "Loading normalized image array from image_provider.");
    esp_err_t status = GetImage(kNumCols, kNumRows, kNumChannels, tmp_buffer);
    if ( status != ESP_OK) 
    {
      ESP_LOGE(TAG, "Image loading failed.");
      continue;
    }

    for (uint i = 0; i < kMaxImageSize; i++)
    {
      // Quantize input : from floating point values to signed 8bit integers
      input->data.int8[i] = (tmp_buffer[i] / input->params.scale ) + input->params.zero_point;
    }
    free(tmp_buffer);

    // Run the model on this input and make sure it succeeds.
    ESP_LOGI(TAG, "Invoking interpreter.");
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Interpreter invoke failed.");
      continue;
    }

    // Obtain pointers to the model's output tensors.
    output = interpreter->output(0);

    float y_pred[kCategoryCount];
    // Dequantize the output from uint8 to floating-point
    for (uint i = 0; i < kCategoryCount; i++)
    {
      y_pred[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    }

    uint8_t max_porb_index = 0;
    
    ESP_LOGI(TAG, "Showing results");
    for (uint i = 0; i < kCategoryCount; i++)
    {
      ESP_LOGI(TAG, "Label = %s, Prob = %f", kCategoryLabels[i], y_pred[i]);
      if (y_pred[i] > y_pred[max_porb_index])
      {
        max_porb_index = i;
      }
    }
    ESP_LOGI(TAG, "Predicted label is : %s", kCategoryLabels[max_porb_index]);
    ESP_LOGI(TAG, "-------------------------\n--------------------------");
    
    vTaskDelay(4000 / portTICK_RATE_MS);
  }

  free(temp_buffer);

  exit:    
    vTaskDelete(NULL);
}

void tf_stop_inference()
{
  isRunning = false;
}
*/