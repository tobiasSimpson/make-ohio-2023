#include <math.h>
#include <Servo.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 20, 4); 

#define EARTH_TILT 0.40840704496667307

Servo servo1;
Servo servo2;

float lat =  40.001074 * 2 * M_PI / 360.0;
float lng = -83.015873 * 2 * M_PI / 360.0;
float time = 0;

const int UPDATES_PER_MINUTE = 5;
const int DELAY = 60 * 1000 / UPDATES_PER_MINUTE;

// String messages[] = { "Loading...", "Processing...", "Calculating...", "Initializing..."};

void setup() {
  servo1.attach(5);
  servo2.attach(3);

  lcd.init();
  lcd.backlight();
  // for (int m = 0; true; m++, m %= 4) {
  //   int length = 0;
  //   int si = 0;
  //   while (messages[m][si] != '\0') { si++; length++; }
  //   lcd.setCursor((16 - length) / 2, 0);
  //   lcd.print(messages[m]);
  //   for (int i = 0; i < 16; i++) {
  //     delay(100);
  //     lcd.setCursor(i, 1);
  //     lcd.print("=");
  //   }
  //   lcd.setCursor(0, 1);
  //   lcd.print("                ");
  //   lcd.setCursor(0, 0);
  //   lcd.print("                ");
  // }

  Serial.begin(9600);
}

void loop() {
  // time += DELAY;
  // delay(DELAY);
  // writeAngle(angleFromLatLngTime(lat, lng, time));

  // for (int i = 90; i <= 180; i += 1) {
  //   writeAngle(i);
  //   delay(25);
  // }
  // for (int i = 180; i >= 0; i -= 1) {
  //   writeAngle(i);
  //   delay(25);
  // }
  int SAMPLES_PER_SECOND = 50;

  lcd.setCursor(0, 0);
  float samples[SAMPLES_PER_SECOND];

  for (int i = 0; i < SAMPLES_PER_SECOND; i++) samples[i] = 2000000000;
  for (int i = 0; i < SAMPLES_PER_SECOND; i++) {
    float sample = analogRead(A0) / 1023.0 * 5.0 * 3.33;
    samples[i] = sample;
    for (int j = 0; j < SAMPLES_PER_SECOND - 1; j++) {
      if (samples[j] > sample) {
        for (int k = SAMPLES_PER_SECOND - 2; k >= j; k--) {
          samples[k + 1] = samples[k];
        }
        samples[j] = sample;
        break;
      }
    }
    delay(1000 / SAMPLES_PER_SECOND);
  }

  for (int i = 0; i < SAMPLES_PER_SECOND; i++) {
    Serial.print(samples[i]);
    Serial.print(" ");
  }
  Serial.println();

  lcd.print(samples[SAMPLES_PER_SECOND / 2]);
  lcd.print(" V");
  lcd.print("                ");
}

int angleFromLatLngTimeRev(float lat, float lng, float time, float rev) {
  lng += (time - 12) * 2 * M_PI / 24.0 + rev * 2 * M_PI;

  float x = cos(lng) * cos(lat);
  float y = sin(lng) * cos(lat);
  float z = sin(lat);
  
  float newY = cos(EARTH_TILT) * y + -sin(EARTH_TILT) * z;
  float newZ = sin(EARTH_TILT) * y + cos(EARTH_TILT) * z;

  float dot = x * cos(rev) + newY * sin(rev);

  return 90 - (acos(dot) * 360.0 / (2.0 * M_PI));
}

void writeAngle(int angle) {
  servo1.write(angle);
  servo2.write(180 - angle);
}









