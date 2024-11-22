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


void setup() {
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
Mouth_Rotational.write(80);
}

void L_LID_D_cycle()
{
  for (int angle = 85; angle <= 100; angle++) 
  {
    L_lid_D.write(angle);
    delay(15);
  }
  for (int angle = 100; angle >= 85; angle--) 
  {
    L_lid_D.write(angle);
    delay(15);
  }
}

void L_LID_U_cycle()
{
  for (int angle = 0; angle <= 70; angle++) 
  {
    L_lid_U.write(angle);
    delay(15);
  }
  for (int angle = 70; angle >= 0; angle--) 
  {
    L_lid_U.write(angle);
    delay(15);
  }
}
void L_ball_UD_cycle()
{
  for (int angle = 97; angle <= 120; angle++) //75 - 97 - 120
  {
    L_eye_UD.write(angle);
    delay(15);
  }
  for (int angle = 120; angle >= 75; angle--) 
  {
    L_eye_UD.write(angle);
    delay(15);
  }
   for (int angle = 75; angle <= 95; angle++) //75 - 97 - 120
  {
    L_eye_UD.write(angle);
    delay(15);
  }
}
void L_ball_LR_cycle()
{
  for (int angle = 50; angle <= 125; angle++) //50 - 87 - 125
  {
    L_eye_LR.write(angle);
    delay(15);
  }
  for (int angle = 125; angle >= 50; angle--) 
  {
    L_eye_LR.write(angle);
    delay(15);
  }
  L_eye_LR.write(87);
  delay(15);
}
void R_ball_UD_cycle()
{
  for (int angle = 80; angle <= 105; angle++) //55 - 80 - 105
  {
    R_eye_UD.write(angle);
    delay(15);
  }
  for (int angle = 105; angle >= 55; angle--) 
  {
    R_eye_UD.write(angle);
    delay(15);
  }
   for (int angle = 55; angle <= 80; angle++) 
  {
    R_eye_UD.write(angle);
    delay(15);
  }
  delay(1000);
}
void R_ball_LR_cycle()
{
  for (int angle = 125; angle <= 150; angle++) //85 - 125 - 150
  {
    R_eye_LR.write(angle);
    delay(15);
  }
  for (int angle = 150; angle >= 85; angle--) 
  {
    R_eye_LR.write(angle);
    delay(15);
  }
  for (int angle = 85; angle <= 125; angle++) 
  {
    R_eye_LR.write(angle);
    delay(15);
  }
  delay(1000);
}
void R_LID_U_cycle()
{
  for (int angle = 135; angle >= 40; angle--) 
  {
    R_lid_U.write(angle);
    delay(15);
  }
  for (int angle = 40; angle <= 135; angle++) 
  {
    R_lid_U.write(angle);
    delay(15);
  }
}
void R_LID_D_cycle()
{
  for (int angle = 103; angle <= 117; angle++) 
  {
    R_lid_D.write(angle);
    delay(15);
  }
  for (int angle = 117; angle >= 103; angle--) 
  {
    R_lid_D.write(angle);
    delay(15);
  }
}


void mouth_linear_cycle()
{
  for (int angle = 95; angle <= 125; angle++) 
  {
    Mouth_linear.write(angle);
    delay(15);
  }
  for (int angle = 125; angle >= 95; angle--) 
  {
    Mouth_linear.write(angle);
    delay(15);
  }
}
void mouth_rot_cycle()
{
  for (int angle = 80; angle <= 115; angle++) 
  {
    Mouth_Rotational.write(angle);
    delay(15);
  }
  for (int angle = 115; angle >= 80; angle--) 
  {
    Mouth_Rotational.write(angle);
    delay(15);
  }
  
}

void loop() {

//  L_LID_D_cycle();      //  85-100
//  L_LID_U_cycle();      //  0-70
//  L_ball_LR_cycle();    //  50-125
//  L_ball_UD_cycle();    //  97-120
//  R_ball_UD_cycle();    //  80-105
//  R_ball_LR_cycle();    //  125-150
//  R_LID_U_cycle();      //  40-135
//  R_LID_D_cycle();      //  103-117
//  mouth_rot_cycle();    //  80-115
//  mouth_linear_cycle(); //  95-125
  
  L_lid_D.write(85);
  L_lid_U.write(0);
  L_eye_LR.write(87);
  L_eye_UD.write(97);
  R_eye_UD.write(80);
  R_eye_LR.write(125);
  R_lid_U.write(135); 
  R_lid_D.write(117);
  Mouth_linear.write(125);
  Mouth_Rotational.write(80);
}
