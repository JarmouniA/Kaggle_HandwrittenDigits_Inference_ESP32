// Copyright 2015-2021 Espressif Systems (Shanghai) PTE LTD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h> 
#include <stdlib.h>
#include <string.h>

#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_http_client.h"
#include "esp_timer.h"
#include "esp_log.h"

#include "sdkconfig.h"

static const char *TAG = "App_HTTPClient";

#define SERVER_IP           "192.168.8.101"
#define SERVER_HTTP_PORT    8000

#define MAX_HTTP_RECV_BUFFER 512
#define MAX_HTTP_OUTPUT_BUFFER 2048

static EventGroupHandle_t server_event_group;

#define SERVER_CONNECTED_BIT     BIT0
#define SERVER_DISCONNECTED_BIT  BIT1

esp_err_t _http_event_handle(esp_http_client_event_t *evt)
{
    static char *output_buffer;  // Buffer to store response of http request from event handler
    static int output_len;       // Stores number of bytes read
    switch(evt->event_id) 
    {
        // This event occurs when there are any errors during execution
        case HTTP_EVENT_ERROR:
            ESP_LOGI(TAG, "HTTP_EVENT_ERROR");
            break;

        // Once the HTTP has been connected to the server, no data exchange has been performed
        case HTTP_EVENT_ON_CONNECTED:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_CONNECTED");
            break;

        // After sending all the headers to the server
        case HTTP_EVENT_HEADERS_SENT:
            ESP_LOGI(TAG, "HTTP_EVENT_HEADERS_SENT");
            break;

        // Occurs when receiving each header sent from the server
        case HTTP_EVENT_ON_HEADER:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_HEADER, key = %s, value = %s", 
            evt->header_key, evt->header_value);
            break;

        // Occurs when receiving data from the server, possibly multiple portions of the packet
        case HTTP_EVENT_ON_DATA:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_DATA, len = %d", evt->data_len);
            if (!esp_http_client_is_chunked_response(evt->client)) 
            {
                // If user_data buffer is configured,
                // copy the response into the buffer.
                if (evt->user_data) {
                    memcpy(evt->user_data + output_len, evt->data, evt->data_len);
                } else {
                    if (output_buffer == NULL)
                    {
                        output_buffer = (char *) malloc(esp_http_client_get_content_length(evt->client));
                        output_len = 0;
                        if (output_buffer == NULL) 
                        {
                            ESP_LOGE(TAG, "Failed to allocate memory for output buffer");
                            return ESP_FAIL;
                        }
                        memcpy(output_buffer + output_len, evt->data, evt->data_len);
                    }
                    memcpy(output_buffer + output_len, evt->data, evt->data_len);
                }
            }
            break;

        // Occurs when finish a HTTP session
        case HTTP_EVENT_ON_FINISH:
            ESP_LOGI(TAG, "HTTP_EVENT_ON_FINISH");
            if (output_buffer != NULL) 
            {
                // Response is accumulated in output_buffer.
                // Uncomment the below line to print the accumulated response.
                ESP_LOG_BUFFER_HEX(TAG, output_buffer, output_len);
                free(output_buffer);
                output_buffer = NULL; // to avoid having a dangling pointer
            }
            output_len = 0;
            break;

        // The connection has been disconnected
        case HTTP_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "HTTP_EVENT_DISCONNECTED");
            int mbedtls_err = 0;
            esp_err_t err = esp_tls_get_and_clear_last_error(evt->data, 
                                                            &mbedtls_err, 
                                                            NULL);
            if (err != 0) 
            {
                if (output_buffer != NULL) 
                {
                    free(output_buffer);
                    output_buffer = NULL; // to avoid having a dangling pointer
                }
                output_len = 0;
                ESP_LOGI(TAG, "Last esp error code: 0x%x", err);
                ESP_LOGI(TAG, "Last mbedtls failure: 0x%x", mbedtls_err);
            }
            break;
    }
    return ESP_OK;
}

esp_err_t http_rest_GET_request(const char* resourcePath, const char* parameters)
{
    ESP_LOGI(TAG, "Sending a GET request...");

    // GET Request
    esp_http_client_config_t config = {
        .host = SERVER_IP,
        .port = SERVER_HTTP_PORT,
        .path = "/",
        //.path = "/image",
        //.query = "outputFormat=png&ImageID=500",
        .method = HTTP_METHOD_GET,
        .event_handler = _http_event_handle,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);

    //vTaskDelay(500 / portTICK_PERIOD_MS);
    /*
    char* path = (char*) malloc( (sizeof(resourcePath) + sizeof(parameters) + 2) * sizeof(char) );
    if (path != NULL)
    {
        strcpy(path, resourcePath);
        if (parameters != NULL && strlen(parameters) > 0)
        {
            strcat(path, "?");
            strcat(path, parameters);
            ESP_LOGI(TAG, "GET request path : %s", path);
        }
        //ESP_ERROR_CHECK(esp_http_client_set_url(client, "/image?outputFormat=png&ImageID=500"));//(const char*)path));
        ESP_LOGI(TAG, "URL is set.");
        free(path);
        path = NULL;
        ESP_LOGI(TAG, "Path freed.");
    } else {
        ESP_LOGE(TAG, "path variable hasn't been allocated.");
        ESP_ERROR_CHECK(esp_http_client_cleanup(client));
        return ESP_FAIL;
    }
    */
    
    ESP_ERROR_CHECK(esp_http_client_set_url(client, "/image?outputFormat=png&ImageID=500"));
    //vTaskDelay(500 / portTICK_PERIOD_MS);

    esp_err_t err = ESP_FAIL;
    ESP_LOGI(TAG, "esp_http_client_perform.");
    ESP_ERROR_CHECK( (err = esp_http_client_perform(client)) );
    ESP_LOGI(TAG, "esp_http_client_perform done.");
    if (err == ESP_OK)
    {
        ESP_LOGI(TAG, "Status = %d, content_length = %d",
            esp_http_client_get_status_code(client),
            esp_http_client_get_content_length(client)
            );
    } else {
        ESP_LOGE(TAG, "Failed to send GET request: %s", esp_err_to_name(err));
        ESP_ERROR_CHECK(esp_http_client_cleanup(client));
        return ESP_FAIL;
    }
    
    vTaskDelay(100 / portTICK_PERIOD_MS);

    //ESP_ERROR_CHECK(esp_http_client_close(client));
    ESP_ERROR_CHECK(esp_http_client_cleanup(client));

    ESP_LOGI(TAG, "GET request has been sent.");
    return ESP_OK;
}

esp_err_t http_rest_POST_request(const char* requestPath, char* post_data)
{
    ESP_LOGI(TAG, "Sending a POST request...");

    // POST Request
    esp_http_client_config_t config = {
        .host = SERVER_IP,
        .port = SERVER_HTTP_PORT,
        .path = "/",
        .method = HTTP_METHOD_POST,
        .event_handler = _http_event_handle,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);

    char output_buffer[MAX_HTTP_OUTPUT_BUFFER] = {0};   // Buffer to store response of http request
    int content_length = 0;
    
    esp_http_client_set_url(client, requestPath);
    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_err_t err;
    ESP_ERROR_CHECK( (err = esp_http_client_open(client, strlen(post_data))) );
    if (err != ESP_OK) 
    {
        ESP_LOGE(TAG, "Failed to open HTTP connection: %s", esp_err_to_name(err));
        ESP_ERROR_CHECK(esp_http_client_cleanup(client));
        return ESP_FAIL;
    } else {
        int wlen = esp_http_client_write(client, post_data, strlen(post_data));
        if (wlen < 0) 
        {
            ESP_LOGE(TAG, "Write failed");
        }
        int data_len = esp_http_client_read_response(client, output_buffer, MAX_HTTP_OUTPUT_BUFFER);
        if (data_len >= 0) 
        {
            ESP_LOGI(TAG, "HTTP POST Status = %d, content_length = %d",
                esp_http_client_get_status_code(client),
                esp_http_client_get_content_length(client));
            ESP_LOG_BUFFER_HEX(TAG, output_buffer, strlen(output_buffer));
        } else {
            ESP_LOGE(TAG, "Failed to read response");
        }
    }

    //ESP_ERROR_CHECK(esp_http_client_close(client));
    ESP_ERROR_CHECK(esp_http_client_cleanup(client));

    ESP_LOGI(TAG, "POST request has been sent.");
    return ESP_OK;
}

static void test_server_connectivity(void * pvParameters)
{
    for( ;; )
    {
        esp_http_client_config_t config = {
            .host = SERVER_IP,
            .port = SERVER_HTTP_PORT,
            .path = "/index.html",
            .method = HTTP_METHOD_GET,
            .event_handler = _http_event_handle,
        };
        esp_http_client_handle_t client = esp_http_client_init(&config);

        ESP_LOGI(TAG, "Connecting to server...");
        esp_err_t err = ESP_FAIL;
        int response_code = 0;
        do
        {
            ESP_ERROR_CHECK( (err = esp_http_client_perform(client)) );
            if (err == ESP_OK) 
            {
                response_code = esp_http_client_get_status_code(client);
                ESP_LOGI(TAG, "Status = %d, content_length = %d",
                    response_code,
                    esp_http_client_get_content_length(client));
            }
            vTaskDelay(1000 / portTICK_PERIOD_MS);
        } while (err != ESP_OK || response_code != 200);
        ESP_LOGI(TAG, "Connection to server is good.");

        //ESP_ERROR_CHECK(esp_http_client_close(client));
        ESP_ERROR_CHECK(esp_http_client_cleanup(client));

        xEventGroupSetBits(server_event_group, SERVER_CONNECTED_BIT);

        EventBits_t bits = xEventGroupWaitBits(server_event_group,
                    SERVER_DISCONNECTED_BIT,
                    pdTRUE, // BIT_x should be cleared before returning.
                    pdFALSE,
                    portMAX_DELAY);
    }
}

void app_httpClient_main(void)
{
    ESP_LOGI(TAG, "Starting httpClient application.");
    
    server_event_group = xEventGroupCreate();
    EventBits_t bits;

    bits = xEventGroupClearBits(
                server_event_group,    // The event group being updated.
                SERVER_CONNECTED_BIT | SERVER_DISCONNECTED_BIT // The bits being cleared.
                );
    
    TaskHandle_t xHandle = NULL;
    xTaskCreate(&test_server_connectivity, "connectivity_test_task", 
                    1024*10, NULL, tskIDLE_PRIORITY, &xHandle);
    
    bits = xEventGroupWaitBits(server_event_group,
                SERVER_CONNECTED_BIT,
                pdTRUE,
                pdFALSE,
                portMAX_DELAY);

    esp_err_t err;
    ESP_LOGI(TAG, "Getting 1st image");
    const char* image_param = "outputFormat=png&ImageID=500";
    do {
        err = http_rest_GET_request("/image", image_param);
        if (err != ESP_OK)
        {
            xEventGroupSetBits(server_event_group, SERVER_DISCONNECTED_BIT);
            bits = xEventGroupWaitBits(server_event_group,
                SERVER_CONNECTED_BIT,
                pdTRUE,
                pdFALSE,
                portMAX_DELAY);
        }
        vTaskDelay(50 / portTICK_PERIOD_MS);
    } while (err != ESP_OK);
    
    /*
    ESP_LOGI(TAG, "Posting 1st result");
    char* result_JSON = "{\"ImageID\": 2, \"result\": 7}";
    do
    {
        err = http_rest_POST_request("/result", result_JSON);
        if (err != ESP_OK)
        {
            xEventGroupSetBits(server_event_group, SERVER_DISCONNECTED_BIT);
            bits = xEventGroupWaitBits(server_event_group,
                SERVER_CONNECTED_BIT,
                pdTRUE,
                pdFALSE,
                portMAX_DELAY);
        }
        vTaskDelay(50 / portTICK_PERIOD_MS);
    } while (err != ESP_OK);

    ESP_LOGI(TAG, "Posting 2nd result");
    result_JSON = "{\"ImageID\": 5, \"result\": 3}";
    do
    {
        err = http_rest_POST_request("/result", result_JSON);
        if (err != ESP_OK)
        {
            xEventGroupSetBits(server_event_group, SERVER_DISCONNECTED_BIT);
            bits = xEventGroupWaitBits(server_event_group,
                SERVER_CONNECTED_BIT,
                pdTRUE,
                pdFALSE,
                portMAX_DELAY);    
        }
        vTaskDelay(50 / portTICK_PERIOD_MS);
    } while (err != ESP_OK);
    
    ESP_LOGI(TAG, "Getting 2nd image");
    image_param = "outputFormat=png&ImageID=700";
    do
    {
        err = http_rest_GET_request("/image", image_param);
        if (err != ESP_OK)
        {
            xEventGroupSetBits(server_event_group, SERVER_DISCONNECTED_BIT);
            bits = xEventGroupWaitBits(server_event_group,
                SERVER_CONNECTED_BIT,
                pdTRUE,
                pdFALSE,
                portMAX_DELAY);
        }
        vTaskDelay(50 / portTICK_PERIOD_MS);
    } while (err != ESP_OK);
    
    ESP_LOGI(TAG, "Posting an export request");
    do
    {
        err = http_rest_POST_request("/export", NULL);
        if (err != ESP_OK)
        {
            xEventGroupSetBits(server_event_group, SERVER_DISCONNECTED_BIT);
            bits = xEventGroupWaitBits(server_event_group,
                SERVER_CONNECTED_BIT,
                pdTRUE,
                pdFALSE,
                portMAX_DELAY);
        }
        vTaskDelay(50 / portTICK_PERIOD_MS);
    } while (err != ESP_OK);

    if(xHandle != NULL)
    {
        vTaskDelete(xHandle);
    }
    vEventGroupDelete(server_event_group);
    */

    ESP_LOGI(TAG, "httpClient application completed.");
}