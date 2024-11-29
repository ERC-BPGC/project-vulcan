#include <ESP32Servo.h>

// Define the servo object
Servo Mouth_linear;
Servo Mouth_Rotational;
//Servo myServo3;
//Servo myServo4;
Servo L_lid_D;
Servo L_lid_U;
Servo L_eye_LR;
Servo L_eye_UD;
Servo R_eye_UD;
Servo R_eye_LR;
Servo R_lid_U;
Servo R_lid_D;
//Servo myServo13;
//Servo myServo14;
//Servo myServo15;
//Servo myServo16;

bool blinking = 1;
bool Talk = 0;

hw_timer_t *timer1 = NULL;
hw_timer_t *timer2 = NULL;

int L_eye_H = 87;
int L_eye_V = 97;
int R_eye_H = 125;
int R_eye_V = 80;

int increment = 4;

volatile int angle_2 = 0;
volatile bool increasing2 = true;

volatile int angle3 = 0;
volatile bool increasing3 = true;

int blink_speed = 2;
int talking_speed = 2;
int bl_counter = 0;
int tl_counter = 0;

int randomNumber = random(5, 10);

void IRAM_ATTR onTimer1() 
{
  tl_counter++;
  if(blinking==1)
  {
    if(bl_counter%(randomNumber*1000) == 0)
    {
      L_lid_U.write(70);
      R_lid_U.write(40);
    }
    if(bl_counter%(randomNumber*1000) == 250)
    {
      L_lid_U.write(0);
      R_lid_U.write(135);
      randomNumber = random(3, 7);
      blinking = 0;
    }
    bl_counter++;
  }
  if(Talk==1)
  {
    if(tl_counter% 800 == 0)
    {
      Mouth_linear.write(125);
    }
    if(tl_counter%800 == 400)
    {
      Mouth_linear.write(95);
    }
  }
}


void setup() {
  Serial.begin(115200);

  Mouth_linear.attach(15);
  Mouth_Rotational.attach(2);
  //  myServo3.attach(4);
  //  myServo4.attach(16);
  L_lid_D.attach(17);
  L_lid_U.attach(5);
  L_eye_LR.attach(18);
  L_eye_UD.attach(19);
  R_eye_UD.attach(32);
  R_eye_LR.attach(33);
  R_lid_U.attach(25);
  R_lid_D.attach(26);
  //  myServo13.attach(27);
  //  myServo14.attach(14);
  //  myServo15.attach(12);
  //  myServo16.attach(13);
  
  L_lid_D.write(85);
  L_lid_U.write(0);
  L_eye_LR.write(87);
  L_eye_UD.write(97);
  R_eye_UD.write(80);
  R_eye_LR.write(125);
  R_lid_U.write(135);
  R_lid_D.write(117);
  Mouth_linear.write(125);
  Mouth_Rotational.write(85);

    // Create a hardware timer
  timer1 = timerBegin(1000000); // Timer 1, prescaler of 80 (1 tick = 1 µs)
  timerAttachInterrupt(timer1, &onTimer1); // Attach interrupt
  timerAlarm(timer1, 1000, true, 0); // Set alarm to 5 seconds (5,000,000 µs)

//  timer2 = timerBegin(1000000); // Timer 2
//  timerAttachInterrupt(timer2, &onTimer2);
//  timerAlarm(timer2, 1000, true, 0); // 300 miliseconds for Servo 2
}

void check_cond()
{
   if (L_eye_V >= 120)
      L_eye_V = 120;
    if (L_eye_V <= 75)
      L_eye_V = 75;
      
    if (L_eye_H >= 105)
      L_eye_H = 105;
    if (L_eye_H <= 60)
      L_eye_H = 60;

    if (R_eye_V >= 100)
      R_eye_V = 100;
    if (R_eye_V <= 55)
      R_eye_V = 55;
      
    if (R_eye_H >= 145)
      R_eye_H = 145;
    if (R_eye_H <= 95)
      R_eye_H = 95;
}
void loop() {
  unsigned long currentMillis = millis();  // Get the current time

  if (Serial.available() > 0) 
  {
    int receivedValue = 5;
    receivedValue = Serial.parseInt();
    Serial.println(receivedValue);
   
    if (receivedValue == 1)             //Eye left
    {
      L_eye_H += increment;
      R_eye_H += increment;
    } 
    else if (receivedValue == 2)        //Eye right
    {
      L_eye_H -= increment;
      R_eye_H -= increment;
    } 
    else if (receivedValue == 3)        //Eye up 
    {
      L_eye_V += increment;
      R_eye_V -= increment;
    } 
    else if (receivedValue == 4)        //Eye down 
    {
      
      L_eye_V -= increment;
      R_eye_V += increment;
    } 
    else if (receivedValue == 5)        //Centre 
    {
      L_eye_H = 87;
      L_eye_V = 97;
      R_eye_H = 125;
      R_eye_V = 80;
    } 
    else if (receivedValue == 6)        //Left wink
    {  
       blinking = 0;
       for (int angle = 0; angle <= 70; angle++) 
       {
          L_lid_U.write(angle);
          delay(10);
       }
       delay(500);
       L_lid_U.write(0);
       blinking = 1;
    } 
    else if (receivedValue == 7)        //Right wink
    {
      blinking = 0;
      for (int angle = 135; angle >= 40; angle--) 
      {
        R_lid_U.write(angle);
        delay(10);
      }
      delay(500);
      R_lid_U.write(135);
      blinking = 1;
    } 
    else if (receivedValue == 8)      //Talk
    {
      Talk = 1;
    } 
    else if (receivedValue == 9)      //Stop talk
    {
      Talk = 0;
    } 
    else if (receivedValue == 10)     //Yawn
    {
      if(Talk == 0)
      {
        for (int angle = 95; angle <= 125; angle++) 
        {
            Mouth_linear.write(angle);
            Mouth_Rotational.write(angle-10);
            delay(30);
        }
        delay(800);
        for (int angle = 125; angle >= 95; angle--) 
        {
            Mouth_linear.write(angle);
            Mouth_Rotational.write(angle-10);
            delay(5);
        } 

      }
    } 
    else if (receivedValue == 11)     //Look around
    {
      for (int angle = 130; angle <= 140; angle++)
      {
        R_eye_LR.write(angle);
        L_eye_LR.write(3*angle-325);
        delay(30);
      }
      for (int angle = 140; angle >= 85; angle--) 
      {
        R_eye_LR.write(angle);
        L_eye_LR.write(3*angle-325);
        delay(30);
      }

      for (int angle = 130; angle <= 150; angle++)
      {
        R_eye_LR.write(angle);
        L_eye_LR.write(3*angle-325);
        delay(15);
      }
      for (int angle = 140; angle >= 85; angle--) 
      {
        R_eye_LR.write(angle);
        L_eye_LR.write(3*angle-325);
        delay(15);
      }
      
    } 
    else if (receivedValue == 12) 
    {
      //Eye roll
      /*for (int i = 0; i < steps; i++) 
      {
        // Calculate the angle in radians
        angle = (2 * PI / steps) * i;

        // Calculate x and y using parametric equations
        int x = xc + radius * cos(angle);
        int y = yc + radius * sin(angle);

        // Ensure x and y are within the allowed range
        if (x >= 70 && x <= 105 && y >= 86 && y <= 108) {
            printf("(%d, %d)\n", x, y);
        }
      }*/
    } else if (receivedValue == 13) {
      //Sleep
    } else if (receivedValue == 14) {
      //Squint
    } else if (receivedValue == 15) {
      //Creep eyes
    }
  }
    check_cond();
   
    if(blinking == 0)
    {
      L_lid_D.write(85);
      L_lid_U.write(0);
      R_lid_U.write(135);
      R_lid_D.write(117);
      blinking = 1;
    }
    L_eye_LR.write(L_eye_H);
    L_eye_UD.write(L_eye_V);
    R_eye_UD.write(R_eye_V);
    R_eye_LR.write(R_eye_H);
    
    if(Talk == 0)
    {
      Mouth_linear.write(125);
      Mouth_Rotational.write(85);
    }

}
