// Smart insole logger: 6 FSR channels at 100 Hz, CSV over USB serial.
// Board: DOIT ESP32 DEVKIT V1, Serial Monitor / recorder baud: 115200.
// Output line: n,t_ms,L_toe,L_mid,L_heel,R_toe,R_mid,R_heel
//   n    = sample counter (lets the recorder detect dropped lines)
//   t_ms = millis() when the sample was taken
// Raw ADC values: about 4095 unpressed, dropping towards 0 under load.

const int PINS[6] = {32, 33, 34, 39, 36, 35};   // L_toe, L_mid, L_heel, R_toe, R_mid, R_heel (verified by pressing each pad)
const int N = 6;
const uint32_t PERIOD_US = 10000;  // 10 ms = 100 Hz

uint32_t nextTick;
uint32_t sampleIdx = 0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  delay(1000);
  Serial.println("n,t_ms,L_toe,L_mid,L_heel,R_toe,R_mid,R_heel");
  nextTick = micros() + PERIOD_US;
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - nextTick) < 0) return;   // not time yet

  nextTick += PERIOD_US;
  if ((int32_t)(now - nextTick) > (int32_t)(5 * PERIOD_US)) {
    nextTick = now + PERIOD_US;                // fell far behind: resync
  }

  int v[N];
  for (int i = 0; i < N; i++) v[i] = analogRead(PINS[i]);

  char buf[72];
  int len = snprintf(buf, sizeof(buf), "%lu,%lu,%d,%d,%d,%d,%d,%d\n",
                     (unsigned long)sampleIdx, (unsigned long)millis(),
                     v[0], v[1], v[2], v[3], v[4], v[5]);
  Serial.write((const uint8_t*)buf, len);
  sampleIdx++;
}
