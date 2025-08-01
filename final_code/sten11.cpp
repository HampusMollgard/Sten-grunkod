//SETTING SPEED
float currentSpeedL = 0;        //CM/S
float currentSpeedRear = 0;
float currentSpeedR = 0;     //CM/S
const int fanPin = 8;

//SPECIFICATIONS CAR
#define wheelBaseWidth 16.3
#define wheelBaseLength 10.3
#define radiusWheelsF 3.36
#define radiusWheelsR 3.36
#define pi 3.14
#define speed 13

#include <Dynamixel2Arduino.h>
#define DXL_SERIAL   Serial1
#define DEBUG_SERIAL Serial
const int DXL_DIR_PIN = 28;
const uint8_t MotorL_ID = 1;
const uint8_t MotorR_ID = 2;
const uint8_t MotorT_ID = 3;
const uint8_t MotorRear_ID = 4;
const uint8_t MotorArm_ID = 5;
const uint8_t MotorWrist_ID = 6;
const float DXL_PROTOCOL_VERSION = 2.0;
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;

unsigned long int  odoMillis = millis();
float odoMeter = 0;

#include "FastIMU.h"
#include <Wire.h>
#define IMU_ADDRESS 0x69     //Change to the address of the IMU
#define PERFORM_CALIBRATION  //Comment to disable startup calibration
BMI160 IMU;                  //Change to the name of any supported IMU!
calData calib = { 0 };       //Calibration data
AccelData IMUAccel;          //Sensor data
GyroData IMUGyro;
MagData IMUMag;
float angleZ = 0;
float angleX = 0;

void setup() {
  // put your setup code here, to run once:
  DEBUG_SERIAL.begin(9600);
  dxl.begin(1000000);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  dxl.ping(MotorL_ID);
  dxl.torqueOff(MotorL_ID);
  dxl.setOperatingMode(MotorL_ID, OP_VELOCITY);
  dxl.torqueOn(MotorL_ID);

  dxl.ping(MotorR_ID);
  dxl.torqueOff(MotorR_ID);
  dxl.setOperatingMode(MotorR_ID, OP_VELOCITY);
  dxl.torqueOn(MotorR_ID);
  dxl.setGoalCurrent(MotorR_ID, 1023);

  dxl.ping(MotorT_ID);
  dxl.torqueOff(MotorT_ID);  // Always turn torque off before changing modes
  dxl.setOperatingMode(MotorT_ID, OP_POSITION);
  dxl.torqueOn(MotorT_ID);
  dxl.setGoalPosition(MotorT_ID, 512);

  dxl.ping(MotorRear_ID);
  dxl.torqueOff(MotorRear_ID);
  dxl.setOperatingMode(MotorRear_ID, OP_VELOCITY);
  dxl.torqueOn(MotorRear_ID);

  dxl.ping(MotorArm_ID);
  dxl.torqueOff(MotorArm_ID);  // Always turn torque off before changing modes
  dxl.setOperatingMode(MotorArm_ID, OP_POSITION);
  dxl.torqueOn(MotorArm_ID);
  dxl.setGoalVelocity(MotorArm_ID, 512);
  dxl.setGoalPosition(MotorArm_ID, 850);

  dxl.ping(MotorWrist_ID);
  dxl.torqueOff(MotorWrist_ID);  // Always turn torque off before changing modes
  dxl.setOperatingMode(MotorWrist_ID, OP_POSITION);
  dxl.torqueOn(MotorWrist_ID);
  dxl.setGoalVelocity(MotorWrist_ID, 300);
  dxl.setGoalPosition(MotorWrist_ID, 400);

  Serial2.begin(115200); // RX2 (UART2)
  
  pinMode(fanPin, OUTPUT);  // Set pin 9 as an output
  digitalWrite(fanPin, LOW);  // Turn pin 9 HIGH
  pinMode(10, INPUT);
  pinMode(11, INPUT);


  setSpeedMotors(100);


  int err = IMU.init(calib, IMU_ADDRESS);
  if (err != 0) {
    Serial.print("Error initializing IMU: ");
    Serial.println(err);
    while (true) {
      ;
    }
  }

  if (err != 0) {
    Serial.print("Error Setting range: ");
    Serial.println(err);
    while (true) {
      ;
    }
  }

#ifdef PERFORM_CALIBRATION
  Serial.println("FastIMU Calibrated Quaternion example");
  Serial.println("Keep IMU level.");
  delay(100);
  IMU.calibrateAccelGyro(&calib);
  Serial.println("Calibration done!");
  Serial.println("Accel biases X/Y/Z: ");
  Serial.print(calib.accelBias[0]);
  Serial.print(", ");
  Serial.print(calib.accelBias[1]);
  Serial.print(", ");
  Serial.println(calib.accelBias[2]);
  Serial.println("Gyro biases X/Y/Z: ");
  Serial.print(calib.gyroBias[0]);
  Serial.print(", ");
  Serial.print(calib.gyroBias[1]);
  Serial.print(", ");
  Serial.println(calib.gyroBias[2]);

  IMU.init(calib, IMU_ADDRESS);
#endif
  //-----------------------------------------

}
unsigned long int lastData = millis();

int val1 = 0;
int val2 = 0;
int val3 = 0;
int val4 = 0;
int val5 = 0;
int val6 = 0;
int x = 0;
unsigned long int lagg = millis();
void loop()
{
  if (millis() - lagg > 20)
    DEBUG_SERIAL.println(millis() - lagg);
  lagg = millis();
  
  IMU.update();
  IMU.getAccel(&IMUAccel);
  IMU.getGyro(&IMUGyro);
  
  bool temp = false;
  if (val3 == 1)
    temp = true;
  
  radiusDrive(val1, val2, temp);

  if (Serial2.available()) {
    DEBUG_SERIAL.println("Something");
    lastData = millis();
    String data = Serial2.readStringUntil('\n'); // Read until newline
    if (sscanf(data.c_str(), "%d,%d,%d", &val1, &val2, &val3) == 3) {
        DEBUG_SERIAL.print("Received LOADS OF DATA: ");
        DEBUG_SERIAL.print(val1);
        DEBUG_SERIAL.print(", ");
        DEBUG_SERIAL.print(val2);
        DEBUG_SERIAL.print(", ");
        DEBUG_SERIAL.println(val3);
    }
    else if (sscanf(data.c_str(), "%d,%d", &val5, &val6) == 2) {
        DEBUG_SERIAL.print("Received VALUE 5/6: ");
        DEBUG_SERIAL.print(val5);
        DEBUG_SERIAL.print(", ");
        DEBUG_SERIAL.println(val6);

        if (val5 == 3){
          val4 = val6;
          val5 = 0;
          val6 = 0;
        }
    }
    
    if (val5 == 1){
      stop();
      pointTurn(val6);
      Serial2.println("DONE");
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;
      val5 = 0;
      val6 = 0;
    }
    else if (val5 == 2){
      
      radiusDrive(10000, 1, false);
      delay(200);
  
      IMU.update();
      IMU.getAccel(&IMUAccel);
      IMU.getGyro(&IMUGyro);
      if (IMUAccel.accelZ > 0){
        if (IMUAccel.accelZ > 1.10){
        }
        if (IMUAccel.accelZ < 0.80){
          val6 = val6;
        }
      }
      else{
        if (IMUAccel.accelZ > -0.8){
        }
        if (IMUAccel.accelZ < -1.10){
          val6 = val6;
        }
      }
      
      resetODO();

      radiusDrive(10000, mathSign(val6) * 10, false);
      
      delay((static_cast<float>(abs(val6)) / 105.0) * 1000.0);

      stop();

      Serial2.println("DONE");
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;
      val5 = 0;
      val6 = 0;
    }
    else if (val5 == 4){
      keepTrackOfRotation(val6);
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;
      val5 = 0;
      val6 = 0;
    }
    else if(val5 == 9){
      DEBUG_SERIAL.println(val6);
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;
      val5 = 0;
      val6 = 0;
    }

    if(val4 == 2){
      for(int i = 0; i < 5; i++){
        if(dxl.setGoalPosition(MotorArm_ID, 650))
          break;
        else
          DEBUG_SERIAL.println("Failed");
      }
      
      delay(300);

      for(int i = 0; i < 5; i++){
        if(dxl.setGoalPosition(MotorWrist_ID, 700))
          break;
        else
          DEBUG_SERIAL.println("Failed");
      }

      delay(1000);

      for(int i = 0; i < 5; i++){
        if(dxl.setGoalPosition(MotorArm_ID, 850))
          break;
        else
          DEBUG_SERIAL.println("Failed");
      }
      
      for(int i = 0; i < 5; i++){
        if(dxl.setGoalPosition(MotorWrist_ID, 400))
          break;
        else
          DEBUG_SERIAL.println("Failed");
      }
      
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;
      val5 = 0;
      val6 = 0;
    }
    else if(val4 == 1){
      for(int i = 0; i < 5; i++){
        if(dxl.setGoalPosition(MotorWrist_ID, 800))
          break;
        else
          DEBUG_SERIAL.println("Failed");
      }
      
      delay(500);
      for(int i = 0; i < 5; i++){
        if(dxl.setGoalPosition(MotorArm_ID, 500))
          break;
        else
          DEBUG_SERIAL.println("Failed");
      }
      
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;
      val5 = 0;
      val6 = 0;
    }
    if(val4 == 3){
      for(int i = 0; i < 5; i++){
        if(dxl.setGoalPosition(MotorWrist_ID, 450))
          break;
        else
          DEBUG_SERIAL.println("Failed");
      }
      delay(1000);
      for(int i = 0; i < 5; i++){
        if(dxl.setGoalPosition(MotorWrist_ID, 400))
          break;
        else
          DEBUG_SERIAL.println("Failed");
      }
      delay(500);
      if (dxl.getPresentPosition(MotorWrist_ID) < 410){
        Serial2.println("FAILED");
        for(int i = 0; i < 5; i++){
          if(dxl.setGoalPosition(MotorWrist_ID, 700))
            break;
          else
            DEBUG_SERIAL.println("Failed");
        }
      }
      else{
        Serial2.println("DONE");
        for(int i = 0; i < 5; i++){
          if(dxl.setGoalPosition(MotorWrist_ID, 450))
            break;
          else
            DEBUG_SERIAL.println("Failed");
        }
      }
    

      delay(1000);
      for(int i = 0; i < 5; i++){
        if(dxl.setGoalPosition(MotorArm_ID, 850))
          break;
        else
          DEBUG_SERIAL.println("Failed");
      }
      
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;
      val5 = 0;
      val6 = 0;
    }
    else if(val4 == 5){
      digitalWrite(fanPin, HIGH);  // Turn pin 9 HIGH
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;
      val5 = 0;
      val6 = 0;
    }
    else if(val4 == 6){
      digitalWrite(fanPin, LOW);  // Turn pin 9 HIGH
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;
      val5 = 0;
      val6 = 0;
    }
    else if(val4 == 4){
      IMU.update();
      IMU.getAccel(&IMUAccel);
      IMU.getGyro(&IMUGyro);
    
      if (abs(IMUAccel.accelY) < 0.25){
        pointTurn(-180);
      }
      else{
        if(IMUAccel.accelY > 0){
          pointTurn(30);
          specialPointTurn(90);
          pointTurn(30);
        }
        else{
          pointTurn(-30);
          specialPointTurn(-90);
          pointTurn(-30);
        }
      }
    
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;
      val5 = 0;
      val6 = 0;
    }
    else if (val4 == 7){
      Serial.print(digitalRead(10));
      Serial.print(",");
      Serial.println(digitalRead(11));
      val1 = 0;
      val2 = 0;
      val3 = 0;
      val4 = 0;
      val5 = 0;
      val6 = 0;
    }
  }
}

bool isClose(float x1, float x2, float tolerance){
  if (abs(x1 - x2) < tolerance)
    return true;
  else
    return false;
}

void forwardDrive(int wantedSpeed){
  currentSpeedL = wantedSpeed;
  currentSpeedR = wantedSpeed;
  currentSpeedRear = wantedSpeed;
  setSpeedMotors(100);
}

void setSpeedMotors(float radius) {
  if (currentSpeedR < 0){
    for(int i = 0; i < 5; i++){
      if(dxl.setGoalVelocity(MotorR_ID, abs(currentSpeedR / (radiusWheelsF * 2 * pi)) * 9 * 60))
        break;
      else
        DEBUG_SERIAL.println("Failed");
    }
  }
  else{
    for(int i = 0; i < 5; i++){
      if(dxl.setGoalVelocity(MotorR_ID, 1024 + abs(currentSpeedR / (radiusWheelsF * 2 * pi)) * 9 * 60))
        break;
      else
        DEBUG_SERIAL.println("Failed");
    }
  }
  
  if (currentSpeedL < 0){
    for(int i = 0; i < 5; i++){
      if(dxl.setGoalVelocity(MotorL_ID, 1024 + abs((currentSpeedL / (radiusWheelsF * 2 * pi)) * 9 * 60)))
        break;
      else
        DEBUG_SERIAL.println("Failed");
    }
  }
  else{
    for(int i = 0; i < 5; i++){
      if(dxl.setGoalVelocity(MotorL_ID, abs((currentSpeedL / (radiusWheelsF * 2 * pi)) * 9 * 60)))
        break;
      else
        DEBUG_SERIAL.println("Failed");
    }
  }
    
  
  if (currentSpeedRear > 0){
    for(int i = 0; i < 5; i++){
      if(dxl.setGoalVelocity(MotorRear_ID, 1024 + abs((currentSpeedRear / (radiusWheelsR * 2 * pi)) * 9 * 60)))
        break;
      else
        DEBUG_SERIAL.println("FAiled");
    }
  }
  else{
    for(int i = 0; i < 5; i++){
      if(dxl.setGoalVelocity(MotorRear_ID, abs((currentSpeedRear / (radiusWheelsR * 2 * pi)) * 9 * 60)))
        break;
      else
        DEBUG_SERIAL.println("Failed");
    }
  }
}

float mapF(float x, float in_min, float in_max, float out_min, float out_max){
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

float getK(int x1, int x2, int y1, int y2){
  float deltaY = y2 - y1;
  float deltaX = x2 - x1;
  float k = deltaY / deltaX;

  return k;
}

void radiusDrive(float radius, int wantedSpeedOutside, bool reversed)
{
  float degreesPerSecond;
  if (reversed)
  {
    float outerRadius = radius + (wheelBaseWidth / 2);
    currentSpeedR = wantedSpeedOutside;
    //currentSpeedR += mathSign(wantedSpeedOutside - currentSpeedR) * 2;
    degreesPerSecond = currentSpeedR / (outerRadius * 2 * pi) * 360;
    float innerRadius = (radius - (wheelBaseWidth / 2));
    currentSpeedL = degreesPerSecond * ((innerRadius * 2 * pi) / 360);
  }
  else
  {
    float outerRadius = radius + (wheelBaseWidth / 2);
    currentSpeedL = wantedSpeedOutside;
    //currentSpeedL += mathSign(wantedSpeedOutside - currentSpeedL) * 2;
    degreesPerSecond = currentSpeedL / (outerRadius * 2 * pi) * 360;
    float innerRadius = (radius - (wheelBaseWidth / 2));
    currentSpeedR = degreesPerSecond * ((innerRadius * 2 * pi) / 360);
  }

  float radiusRear = radius + (sqrtf((radius * radius) + (wheelBaseLength * wheelBaseLength)) - radius);
  float angleOfAttack;
  if (reversed){
    angleOfAttack = -90 + 2 * (((atanf(radiusRear / wheelBaseLength)) * (180 / pi)) - 45);
  }
  else{
    angleOfAttack = 90 - 2 * (((atanf(radiusRear / wheelBaseLength)) * (180 / pi)) - 45);
  }
  currentSpeedRear = degreesPerSecond * ((radiusRear * 2 * pi) / 360);

  if (wantedSpeedOutside < 0)
    currentSpeedRear = currentSpeedRear;

  if (wantedSpeedOutside != 0){
    for(int i = 0; i < 5; i++){
      if(dxl.setGoalPosition(MotorT_ID, (158 - angleOfAttack) / 0.29))
        break;
      else
        DEBUG_SERIAL.println("Failed");
    }
  }
  
  if (currentSpeedRear > 28){
    radiusDrive(radius, wantedSpeedOutside - 1, reversed);
    return;
  }
  
  IMU.update();
  IMU.getAccel(&IMUAccel);
  IMU.getGyro(&IMUGyro);
  if (radius > -1){
    if (IMUAccel.accelZ > 0){
      if (IMUAccel.accelZ > 1.10){
        if (currentSpeedRear >= 0){
          currentSpeedRear = currentSpeedRear * 0.5;
          if (currentSpeedR > 0){
            currentSpeedR = currentSpeedR * 0.5;
          }

          if (currentSpeedL > 0){
            currentSpeedL = currentSpeedL * 0.5;
          }
        }
        else{
          currentSpeedR = currentSpeedR * 2.5;
          currentSpeedL = currentSpeedL * 2.5;
        }
        
      }
      if (IMUAccel.accelZ < 0.80 && radius > 10){
        if (currentSpeedRear >= 0){
          currentSpeedRear = currentSpeedRear * 2.5;
        }
        else{
          currentSpeedRear = currentSpeedRear * 0.5;
          if (currentSpeedR < 0){
            currentSpeedR = currentSpeedR * 0.5;
          }

          if (currentSpeedL < 0){
            currentSpeedL = currentSpeedL * 0.5;
          }
        }
        
      }
    }
    else{
      if (IMUAccel.accelZ > -0.8){
        if (currentSpeedRear >= 0){
          currentSpeedRear = currentSpeedRear * 0.5;
          if (currentSpeedR > 0){
            currentSpeedR = currentSpeedR * 0.5;
          }

          if (currentSpeedL > 0){
            currentSpeedL = currentSpeedL * 0.5;
          }
        }
        else{
          currentSpeedR = currentSpeedR * 2.5;
          currentSpeedL = currentSpeedL * 2.5;
        }
      }
      if (IMUAccel.accelZ < -1.10 && radius > 10){
        if (currentSpeedRear >= 0){
          currentSpeedRear = currentSpeedRear * 2.5;
        }
        else{
          currentSpeedRear = currentSpeedRear * 0.5;
          if (currentSpeedR < 0){
            currentSpeedR = currentSpeedR * 0.5;
          }

          if (currentSpeedL < 0){
            currentSpeedL = currentSpeedL * 0.5;
          }
        }
      }
    }
  }
  

  setSpeedMotors(radius);
}

void pointTurn(float degrees) {
  bool reversed = false;
  float degreesPerSecond = 0;
  float degreesDone = 0;
  unsigned long int millies = millis();

  if (degrees < 0)
    reversed = true;
  
  radiusDrive(0, 1, reversed);
  delay(300);

  unsigned long int start = millis();
  unsigned long int startPointTurn = millis();
  float rotatedDegrees = 0;
  while (abs(rotatedDegrees) <= abs(degrees)) {
    if (millis() - startPointTurn > 5000) {
      resetODO();
      while (abs(odoMeter) < 30) {
        forwardDrive(15);
        updateDistanceDriven();
        startPointTurn = millis();
      }
    }
    IMU.update();
    IMU.getAccel(&IMUAccel);
    IMU.getGyro(&IMUGyro);
    rotatedDegrees += (millis() - start) * IMUGyro.gyroX / 1000;
    start = millis();
    if (abs(rotatedDegrees - degrees) < 10)
      radiusDrive(0, 6, reversed);
    else
      radiusDrive(0, 8, reversed);
  }

  stop();
}

void specialPointTurn(float degrees) {
  bool reversed = false;
  float degreesPerSecond = 0;
  float degreesDone = 0;
  unsigned long int millies = millis();

  if (degrees < 0)
    reversed = true;
  
  radiusDrive(0, 1, reversed);
  delay(300);

  unsigned long int start = millis();
  unsigned long int startPointTurn = millis();
  float rotatedDegrees = 0;
  while (abs(rotatedDegrees) <= abs(degrees)) {
    if (millis() - startPointTurn > 3000) {
      resetODO();
      while (abs(odoMeter) < 30) {
        forwardDrive(15);
        updateDistanceDriven();
        startPointTurn = millis();
      }
    }
    IMU.update();
    IMU.getAccel(&IMUAccel);
    IMU.getGyro(&IMUGyro);
    rotatedDegrees += (millis() - start) * IMUGyro.gyroX / 1000;
    start = millis();
    if (abs(rotatedDegrees - degrees) < 10)
      radiusDrive(10, 6, reversed);
    else
      radiusDrive(10, 8, reversed);
  }

  stop();
}

int mathSign(int x){
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;
}

void updateDistanceDriven(){
  float temp = dxl.getPresentVelocity(MotorRear_ID);
  if (temp > 1014){
    temp = temp - 1024;
  }
  if (temp > 135){
    temp = 135;
  }
  temp = temp * 0.153;
  float speedRear = temp * (2.0f * pi * radiusWheelsR) / 60.0f;
  

  odoMeter += ((millis() - odoMillis) * (speedRear) / 100) * 2;
  Serial.print(speedRear);
  Serial.print(", ");
  Serial.println(odoMeter);
  odoMillis = millis();
}

void resetODO(){
  odoMeter = 0;
  odoMillis = millis();
}

void stop(){
  currentSpeedL = 0;
  currentSpeedR = 0;
  currentSpeedRear = 0;
  setSpeedMotors(100);
}

void keepTrackOfRotation(float degrees){
  bool reversed = false;
  if (degrees < 0){
    reversed = true;
  }
  radiusDrive(0, 1, reversed);
  delay(300);
  radiusDrive(0, 10, reversed);
  float degreesPerSecond = 0;
  float degreesDone = 0;
  unsigned long int millies = millis();
  
  unsigned long int start = millis();
  unsigned long int startPointTurn = millis();
  float rotatedDegrees = 0;
  while (abs(rotatedDegrees) <= abs(degrees)) {
    IMU.update();
    IMU.getAccel(&IMUAccel);
    IMU.getGyro(&IMUGyro);
    rotatedDegrees += (millis() - start) * IMUGyro.gyroX / 1000;
    start = millis();

    if (Serial2.available()) {
      String data = Serial2.readStringUntil('\n'); // Read until newline
      if (sscanf(data.c_str(), "%d,%d", &val5, &val6) == 2) {
          if (val6 == 10){
            return;
          }
      }
    }
  }
  DEBUG_SERIAL.println("EDNDING");

  stop();
  Serial2.println("DONE");
  return;
}
