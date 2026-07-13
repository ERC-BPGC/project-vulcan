#include <TimerOne.h>
#include <Servo.h>


Servo Lverti;
Servo Lhori;
Servo Lupper;
Servo Llower;

Servo Rverti;
Servo Rhori;
Servo Rupper;
Servo Rlower;

Servo Mouth1;
Servo Mouth2; 



void setup(void)
{
  Timer1.initialize(40000000);  // 40 sec
  Timer1.attachInterrupt(timer_call); // function called
  noInterrupts(); // none while initialising

  // All motors of eye subsystem
  {
    Lverti.attach(2);
    Lhori.attach(3);
    Lupper.attach(4);
    Llower.attach(5);

    Rverti.attach(6);
    Rhori.attach(7);
    Rupper.attach(8);
    Rlower.attach(9);
  }

  //All motors of mouth subsystem
  {
    Mouth1.attach(10);
    Mouth2.attach(11);
  }
  interrupts();   //  here we go
}

void timer_call(void)
{ 
  if(random(2) == 1)  //blink once
  {
    for(int i = 0; i<60; i++)
    {
      Lupper.write(i);
      Rupper.write(i);
      delay(1);
    }
    for(int i = 60; i>0; i--)
    {
      Lupper.write(i);
      Rupper.write(i);
      delay(1);
    }
  }
  else                //blink twice
  {
    for(int i = 0; i<60; i++)
    {
      Lupper.write(i);
      Rupper.write(i);
      delay(1);
    }
    for(int i = 60; i>0; i--)
    {
      Lupper.write(i);
      Rupper.write(i);
      delay(1);
    }
    for(int i = 0; i<60; i++)
    {
      Lupper.write(i);
      Rupper.write(i);
      delay(1);
    }
    for(int i = 60; i>0; i--)
    {
      Lupper.write(i);
      Rupper.write(i);
      delay(1);
    }
  }

  // digitalWrite(led, ledState);
  Timer1.initialize(10000000 * random(4,12));
}


void loop(void)
{

  delay(10000);
}
