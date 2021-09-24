#ifndef _APP_HTTP_CLIENT_H_
#define _APP_HTTP_CLIENT_H_

#include "freertos/event_groups.h"

static EventGroupHandle_t server_event_group;

#define SERVER_CONNECTED_BIT     BIT0
#define SERVER_DISCONNECTED_BIT  BIT1

#ifdef __cplusplus
extern "C" {
#endif

void httpClient_getImage(uint8_t*);
void app_httpClient_main(void);

#ifdef __cplusplus
}
#endif

#endif // _APP_HTTP_CLIENT_H_