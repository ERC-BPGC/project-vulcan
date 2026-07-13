#include <Servo.h>

//CW -> towards zero (looking at servo)


Servo eye_left_right;
Servo eye_up_down;

Servo eyelid_up;
Servo myservo3;
Servo myservo4;
// Servo myservo5;
// Servo myservo6;
// Servo myservo7;
Servo eye_other;

int eye_left_right_angle = 90;
int eye_up_down_angle = 90;

int eyelid_up_angle = 30;
int eye_other_angle = 50;

int limit = 20, increment = 5;

int eye_change = 15;




unsigned long previousMillis = 0;     // Store the last time the servo moved
const unsigned long interval = 5000;  // 5 seconds in milliseconds
const unsigned long interval_blink = 100;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);

  eye_left_right.attach(2);
  eye_up_down.attach(3);

  eyelid_up.attach(4);
  //myservo4.attach(5);
  // myservo5.attach(6);
  // myservo6.attach(7);
  // myservo7.attach(8);
  // myservo8.attach(9);

  eye_left_right.write(90);
  eye_up_down.write(90);
  eyelid_up.write(eyelid_up_angle);

  eye_other.attach(5);
  eye_other.write(eye_other_angle);
}

void loop() {

  unsigned long currentMillis = millis();  // Get the current time

  if (Serial.available() > 0) {


    // Serial.println(Serial.parseInt());
    int receivedValue = 5;
    receivedValue = (int)Serial.read();
    Serial.println(receivedValue);


    if (receivedValue == 7) {
      //go left
      eye_left_right_angle += increment;

    } else if (receivedValue == 2) {
      //go right
      eye_left_right_angle -= increment;

    } else if (receivedValue == 1) {
      //go up
      eye_up_down_angle += increment;
    } else if (receivedValue == 3) {
      //go down
      eye_up_down_angle -= increment;
    } else if (receivedValue == 5) {
      //center
    }

    if (eye_left_right_angle >= 90 + limit)
      eye_left_right_angle -= increment;
    if (eye_left_right_angle <= 90 - limit)
      eye_left_right_angle += increment;

    if (eye_up_down_angle >= 90 + limit)
      eye_up_down_angle -= increment;
    if (eye_up_down_angle <= 90 - limit)
      eye_up_down_angle += increment;

    // Serial.println(eye_left_right_angle);
    eye_left_right.write(eye_left_right_angle);
    eye_up_down.write(eye_up_down_angle);
  }



  if (currentMillis - previousMillis >= interval) {
    // It's time to move the servo
    previousMillis = currentMillis;  // Save the last time the servo moved


    eyelid_up.write(eyelid_up_angle - eye_change);
    eye_other.write(eye_other_angle + eye_change);
    // eye_change = -eye_change;
    // delay(interval_blink);
    eyelid_up.write(eyelid_up_angle + eye_change);
     eye_other.write(eye_other_angle - eye_change);

    delay(interval_blink);
    eyelid_up.write(eyelid_up_angle - eye_change);
    eye_other.write(eye_other_angle + eye_change);


  }
}