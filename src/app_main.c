#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_spi_flash.h"

#include "app_tflite.h"
#include "app_wifi.h"
#include "app_httpClient.h"

static const char *TAG = "App_Main";

void app_main()
{
  ESP_LOGI(TAG, "Starting main application");

  app_wifi_main();
  vTaskDelay(50 / portTICK_PERIOD_MS);
  app_httpClient_main();
  vTaskDelay(50 / portTICK_PERIOD_MS);
  TF_init_status = app_tflite_init();
  
  if (TF_init_status == ESP_OK){
      //Start prediction task
      xTaskCreate(&tf_start_inference, "tf_start_inference", 1024*20, NULL, tskIDLE_PRIORITY, NULL);
  } else ESP_LOGI(TAG, "TfLite was not initialized");

  while(true)
  {
    vTaskDelay(10 / portTICK_PERIOD_MS);
  }
  ESP_LOGI(TAG, "Main application execution completed.");
}