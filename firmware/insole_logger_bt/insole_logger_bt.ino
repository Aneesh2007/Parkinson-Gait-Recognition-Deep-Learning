// Wireless logger v2: FreeRTOS sampler task at 100 Hz + queue, output over Bluetooth (device name "insole").
// Board: DOIT ESP32 DEVKIT V1 (classic Bluetooth; not the S3 or C3 chips).
#include "BluetoothSerial.h"
#if !defined(CONFIG_BT_ENABLED) || !defined(CONFIG_BLUEDROID_ENABLED)
#error Bluetooth is not enabled for this board. Choose DOIT ESP32 DEVKIT V1.
#endif

BluetoothSerial SerialBT;

const int PINS[6] = {32, 33, 34, 39, 36, 35};   // L_toe, L_mid, L_heel, R_toe, R_mid, R_heel (verified by pressing each pad)
const int N = 6;
const uint32_t PERIOD_MS = 10;   // 100 Hz

struct Sample { uint32_t n; uint32_t t; uint16_t v[6]; };
QueueHandle_t q;

void samplerTask(void*) {
  TickType_t last = xTaskGetTickCount();
  uint32_t idx = 0;
  for (;;) {
    vTaskDelayUntil(&last, pdMS_TO_TICKS(PERIOD_MS));
    Sample s;
    s.n = idx++;
    s.t = millis();
    for (int i = 0; i < N; i++) s.v[i] = analogRead(PINS[i]);
    xQueueSend(q, &s, 0);
  }
}

void setup() {
  Serial.begin(115200);
  SerialBT.begin("insole");
  analogReadResolution(12);
  q = xQueueCreate(500, sizeof(Sample));
  xTaskCreatePinnedToCore(samplerTask, "sampler", 4096, NULL, 3, NULL, 1);
  delay(500);
  Serial.println("n,t_ms,L_toe,L_mid,L_heel,R_toe,R_mid,R_heel");
}

void loop() {
  if (uxQueueMessagesWaiting(q) < 5) { delay(5); return; }
  static char buf[1024];
  int len = 0;
  Sample s;
  while (len < 900 && xQueueReceive(q, &s, 0) == pdTRUE) {
    len += snprintf(buf + len, sizeof(buf) - len, "%lu,%lu,%u,%u,%u,%u,%u,%u\n",
                    (unsigned long)s.n, (unsigned long)s.t,
                    s.v[0], s.v[1], s.v[2], s.v[3], s.v[4], s.v[5]);
  }
  if (len > 0) {
    if (SerialBT.hasClient()) SerialBT.write((const uint8_t*)buf, len);
    else Serial.write((const uint8_t*)buf, len);
  }
}
