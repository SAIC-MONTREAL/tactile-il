// A basic everyday NeoPixel strip test program.

// NEOPIXEL BEST PRACTICES for most reliable operation:
// - Add 1000 uF CAPACITOR between NeoPixel strip's + and - connections.
// - MINIMIZE WIRING LENGTH between microcontroller board and first pixel.
// - NeoPixel strip's DATA-IN should pass through a 300-500 OHM RESISTOR.
// - AVOID connecting NeoPixels on a LIVE CIRCUIT. If you must, ALWAYS
//   connect GROUND (-) first, then +, then data.
// - When using a 3.3V microcontroller with a 5V-powered NeoPixel strip,
//   a LOGIC-LEVEL CONVERTER on the data line is STRONGLY RECOMMENDED.
// (Skipping these may work OK on your workbench but can fail in the field)

#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
#include <avr/power.h> // Required for 16 MHz Adafruit Trinket
#endif

// Which pin on the Arduino is connected to the NeoPixels?
// On a Trinket or Gemma we suggest changing this to 1:
#define LED_PIN    6

// How many NeoPixels are attached to the Arduino?
#define DEFAULT_LED_COUNT 4

// Declare our NeoPixel strip object:
Adafruit_NeoPixel strip(DEFAULT_LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
// Argument 1 = Number of pixels in NeoPixel strip
// Argument 2 = Arduino pin number (most are valid)
// Argument 3 = Pixel type flags, add together as needed:
//   NEO_KHZ800  800 KHz bitstream (most NeoPixel products w/WS2812 LEDs)
//   NEO_KHZ400  400 KHz (classic 'v1' (not v2) FLORA pixels, WS2811 drivers)
//   NEO_GRB     Pixels are wired for GRB bitstream (most NeoPixel products)
//   NEO_RGB     Pixels are wired for RGB bitstream (v1 FLORA pixels, not v2)
//   NEO_RGBW    Pixels are wired for RGBW bitstream (NeoPixel RGBW products)

String line;
int n_leds = DEFAULT_LED_COUNT;
bool input = false;


void setColor(uint32_t color) {
  for(int i=0; i<strip.numPixels(); i++) { // For each pixel in strip...
    strip.setPixelColor(i, color);         //  Set pixel's color (in RAM)
  }
    strip.show();                          //  Update strip to match
}

void processLine(String s)
{
  int i, n, r, g, b;
  String t;

  t = s.substring(0, 3);
  n = t.toInt();

  if(n != n_leds) {
    for(i=0;i<n_leds;i++)
      strip.setPixelColor(i, strip.Color(0, 0, 0));
    strip.show();
    strip.updateLength(n);
    n_leds = n;
  }

  for(i=0;i<n_leds;i++) {
    t = s.substring(3 + 0 + i * 9, 3 + 3 + i * 9);
    r = t.toInt();
    t = s.substring(3 + 3 + i * 9, 3 + 6 + i * 9);
    g = t.toInt();
    t = s.substring(3 + 6 + i * 9, 3 + 9 + i * 9);
    b = t.toInt();

    strip.setPixelColor(i, strip.Color(r, g, b));
  }
  strip.show();

}

// input format is a count followed by count triples of rgb. each number is exactly three characters long. So a strip of length 1 all white is
// 001255255255

void setup() {

  // END of Trinket-specific code.

  strip.begin();             // INITIALIZE NeoPixel strip object (REQUIRED)
  strip.show();              // Turn OFF all pixels ASAP
  strip.setBrightness(100);  // Set BRIGHTNESS to about 1/5 (max = 255)
  Serial.begin(9600);
  processLine("016255255255000000255000255000255000000255255255000000255000255000255000000255255255000000255000255000255000000255255255000000255000255000255000000");
  strip.show();
  input = false;
  line = "";
  while(!Serial)             // needed on Micro
    ;;
}

void loop() {
  if(Serial.available()) {
    char c = Serial.read();
    Serial.print(c);
    if(c == '\n'){
      input = true;
    } else {
      line = line + c;
    }
  }

  if(input) {
    processLine(line);
    line = "";
    input = false;
  }
}

