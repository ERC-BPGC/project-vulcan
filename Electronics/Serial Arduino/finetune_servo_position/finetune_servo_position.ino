#include <Servo.h>

Servo myservo;

int pos = 0;

void setup() {
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
  delay(100);
  myservo.write(40);
  delay(100);
}

void loop() {
  delay(1000);
}
