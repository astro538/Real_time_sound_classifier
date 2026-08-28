#include <WiFi.h>
#include <FastLED.h>

#define LED_PIN 5
#define NUM_LEDS 12

CRGB leds[NUM_LEDS];

const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";

WiFiServer server(80);

void setColor(CRGB c) {
  fill_solid(leds, NUM_LEDS, c);
  FastLED.show();
}

void setup() {

  Serial.begin(115200);

  FastLED.addLeds<WS2812B, LED_PIN, GRB>(
    leds, NUM_LEDS
  );

  FastLED.setBrightness(20);
  setColor(CRGB::Black);

  Serial.println("Connecting to WiFi...");

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  server.begin();

  Serial.println("HTTP server started!");
}

void loop() {

  WiFiClient client = server.available();

  if (!client)
    return;

  String request = "";
  int contentLength = 0;

  // Read headers line by line
  while (client.connected()) {
    if (client.available()) {
      String line = client.readStringUntil('\n');

      if (line.startsWith("Content-Length:")) {
        contentLength = line.substring(16).toInt();
      }

      if (line == "\r") {  // blank line = end of headers
        break;
      }
    }
  }

  // Now read the body (this is where "speech", "bark" etc. actually live)
  String body = "";
  while (body.length() < contentLength && client.connected()) {
    if (client.available()) {
      char c = client.read();
      body += c;
    }
  }

  body.toLowerCase();

  if (body.indexOf("speech") >= 0)
    setColor(CRGB::Blue);

  else if (body.indexOf("bark") >= 0)
    setColor(CRGB::Orange);

  else if (body.indexOf("knock") >= 0)
    setColor(CRGB::Purple);

  else if (body.indexOf("music") >= 0)
    setColor(CRGB::Green);

  else if (body.indexOf("clapping") >= 0)
    setColor(CRGB::Yellow);

  else if (body.indexOf("doorbell") >= 0)
    setColor(CRGB::Magenta);

  else if (body.indexOf("alarm") >= 0)
    setColor(CRGB::Red);

  else if (body.indexOf("noise") >= 0)
    setColor(CRGB::Black);

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/plain");
  client.println("Connection: close");
  client.println();
  client.println("OK");

  client.stop();
}