// Live test of all 6 FSR channels. No Python needed.
// Board: DOIT ESP32 DEVKIT V1. Serial Monitor baud: 115200.
//
// 1. Upload with the insole lying flat and NOTHING pressed.
// 2. Wait 2 seconds (it measures the unpressed level of each sensor).
// 3. Press each sensor with a finger. Its column shows PRESSED when it works.
//
// Set PLOTTER to true to use Tools > Serial Plotter instead (six coloured lines).
// Pins match insole_logger.ino: L_toe 32, L_mid 33, L_heel 34, R_toe 35, R_mid 36, R_heel 39.

const bool PLOTTER = false;
const int PINS[6] = {32, 33, 34, 39, 36, 35};   // L_toe, L_mid, L_heel, R_toe, R_mid, R_heel (verified by pressing each pad)
const char* NAMES[6] = {"L_toe", "L_mid", "L_heel", "R_toe", "R_mid", "R_heel"};
const int PRESS_DROP = 400;   // reading must fall this far below the unpressed level

int baseline[6];

int readAvg(int pin) {
  long s = 0;
  for (int k = 0; k < 8; k++) s += analogRead(pin);
  return s / 8;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  delay(1500);
  Serial.println();
  Serial.println("Measuring unpressed level. Hands off the insole...");
  delay(500);
  for (int i = 0; i < 6; i++) baseline[i] = 0;
  for (int n = 0; n < 20; n++) {
    for (int i = 0; i < 6; i++) baseline[i] += readAvg(PINS[i]);
    delay(50);
  }
  for (int i = 0; i < 6; i++) baseline[i] /= 20;

  Serial.println("Unpressed levels (about 4000 to 4095 is normal):");
  for (int i = 0; i < 6; i++) {
    Serial.print("  "); Serial.print(NAMES[i]); Serial.print(" = "); Serial.print(baseline[i]);
    if (baseline[i] < 3000) Serial.print("   <-- low: wire loose, shorted, or wrong pin");
    Serial.println();
  }
  Serial.println("Now press each sensor.\n");
}

void loop() {
  int v[6];
  for (int i = 0; i < 6; i++) v[i] = readAvg(PINS[i]);

  if (PLOTTER) {
    for (int i = 0; i < 6; i++) {
      Serial.print(NAMES[i]); Serial.print(":"); Serial.print(v[i]);
      Serial.print(i < 5 ? "," : "\n");
    }
  } else {
    for (int i = 0; i < 6; i++) {
      bool pressed = (baseline[i] - v[i]) > PRESS_DROP;
      Serial.print(NAMES[i]); Serial.print("=");
      Serial.print(v[i]);
      Serial.print(pressed ? " PRESSED" : "        ");
      Serial.print(i == 2 ? "  |  " : "  ");
    }
    Serial.println();
  }
  delay(150);
}
