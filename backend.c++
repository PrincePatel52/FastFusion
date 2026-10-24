#include <TinyGPSPlus.h>
#include "DHT.h"

#define GPS_RX_PIN 16   // ESP32 RX2  <- GPS TX
#define GPS_TX_PIN 17   // ESP32 TX2  -> GPS RX
#define DHTPIN     4
#define DHTTYPE    DHT22

TinyGPSPlus gps;
HardwareSerial GPSSerial(2);
DHT dht(DHTPIN, DHTTYPE);

unsigned long lastPrint = 0;
float lastTemp = NAN;
float lastHum  = NAN;

void setup() {
  Serial.begin(115200);
  GPSSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  dht.begin();
  delay(1000);
  Serial.println("SPORTS_TRACKER_READY");
}

void loop() {
  // GPS ka data lagatar padho
  while (GPSSerial.available() > 0) {
    gps.encode(GPSSerial.read());
  }

  if (millis() - lastPrint >= 2000) {
    lastPrint = millis();

    // Fix tabhi valid jab location valid ho aur 3 sec se purani na ho
    bool fix = gps.location.isValid() && gps.location.age() < 3000;

    float lat   = fix ? gps.location.lat() : 0.0;
    float lon   = fix ? gps.location.lng() : 0.0;
    float speed = (fix && gps.speed.isValid()) ? gps.speed.kmph() : 0.0;
    if (speed < 1.0) speed = 0.0;   // khade hone par GPS drift hatao
    int   sats  = gps.satellites.isValid() ? gps.satellites.value() : 0;
    float alt   = (fix && gps.altitude.isValid()) ? gps.altitude.meters() : 0.0;
    float hdop  = gps.hdop.isValid() ? gps.hdop.hdop() : 99.0;

    // DHT22 - fail ho to pichli sahi value rakho
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t)) lastTemp = t;
    if (!isnan(h)) lastHum  = h;

    Serial.print("DATA,");
    Serial.print(fix ? 1 : 0);      Serial.print(",");
    Serial.print(lat, 6);           Serial.print(",");
    Serial.print(lon, 6);           Serial.print(",");
    Serial.print(speed, 2);         Serial.print(",");
    Serial.print(sats);             Serial.print(",");
    if (isnan(lastTemp)) Serial.print("nan"); else Serial.print(lastTemp, 1);
    Serial.print(",");
    if (isnan(lastHum))  Serial.print("nan"); else Serial.print(lastHum, 1);
    Serial.print(",");
    Serial.print(alt, 1);           Serial.print(",");
    Serial.println(hdop, 1);
  }
}