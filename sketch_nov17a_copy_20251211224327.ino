#include <Servo.h>

Servo servoX;  
Servo servoY;
const int LASER_PIN = 7;

int posX = 1500;
int posY = 1500;
const int step = 20;

void setup() {
  servoX.attach(9);
  
  servoY.attach(10);
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);
  
  Serial.begin(115200);
  homePosition();
  Serial.println("Arduino + Laser ready");
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim();

    if (data.startsWith("X") && data.indexOf("Y") > 0) {
      int xVal = getValue(data, 'X');
      int yVal = getValue(data, 'Y');
      int laserState = getValue(data, 'L');

      xVal = constrain(xVal, 900, 2100);
      yVal = constrain(yVal, 900, 2100);

      digitalWrite(LASER_PIN, laserState ? HIGH : LOW);
      smoothMove(xVal, yVal);
    }
  }
}

int getValue(String data, char letter) {
  int index = data.indexOf(letter);
  if (index == -1) return -1;
  int spaceIndex = data.indexOf(' ', index);
  if (spaceIndex == -1) spaceIndex = data.length();
  return data.substring(index + 1, spaceIndex).toInt();
}

void smoothMove(int targetX, int targetY) {
  if (abs(posX - targetX) < 4 && abs(posY - targetY) < 4) return;

  while (abs(posX - targetX) > step || abs(posY - targetY) > step) {
    if (abs(posX - targetX) > step) posX += (posX < targetX) ? step : -step;
    if (abs(posY - targetY) > step) posY += (posY < targetY) ? step : -step;
    
    servoX.writeMicroseconds(posX);
    servoY.writeMicroseconds(posY);

    delay(30);
  }

  servoX.writeMicroseconds(targetX);
  servoY.writeMicroseconds(targetY);
  posX = targetX;
  posY = targetY;
}

void homePosition() {
  servoX.writeMicroseconds(1500);
  servoY.writeMicroseconds(1500);
  posX = 1500;
  posY = 1500;
  digitalWrite(LASER_PIN, LOW);
  delay(600);
}
