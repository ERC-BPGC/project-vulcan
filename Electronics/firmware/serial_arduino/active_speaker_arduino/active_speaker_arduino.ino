int incoming[2];

void setup(){
    Serial.begin(115200);
    Serial.setTimeout(1);
}

void loop(){
  while(Serial.available() >= 2){
    
    for (int i = 0; i < 2; i++){
      incoming[i] = Serial.read();
    }
  }
}