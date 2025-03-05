#include "Adafruit_VL53L0X.h"
Adafruit_VL53L0X sensor2 = Adafruit_VL53L0X();

#include "Adafruit_TCS34725.h"
float red, green, blue;
uint16_t r, g, b, c;
int speed = 13;

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

//SETTINGS ACCELERATION AND SPEED
float currentSpeedL = 0;  //CM/S
float currentSpeedRear = 0;
float currentSpeedR = 0;  //CM/S

//Ball settings
bool hasBalls = false;
bool deliveredBalls = false;
bool outOfBallArea = false;
bool gotOut = false;
bool turnedRightLast = true;
int ballSize = 50;
int ballCounter = 0;
int totalBalls = 3;

Adafruit_TCS34725 colorSensorR = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_2_4MS, TCS34725_GAIN_4X);
Adafruit_TCS34725 colorSensorL = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_2_4MS, TCS34725_GAIN_4X);
Adafruit_TCS34725 colorSensorF = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_360MS, TCS34725_GAIN_4X);
Adafruit_TCS34725 colorSensorB = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_360MS, TCS34725_GAIN_4X);


//DEPTH SENSORS
bool useSensors = true;
bool gettingOut = false;
bool bypassMotorOptimization = false;

//SPECIFICATIONS CAR
#define wheelBaseWidth 10
#define wheelBaseLength 7
#define radiusWheelsF 3.12
#define radiusWheelsR 2.5  //2.4
#define pi 3.14

//Enviroment Settings
int whiteLimit = 200;
int maxBlack = 1023;
float intersectionWidth = 4.5;  //4,5
float lineWidth = 3;            //3

#include <Dynamixel2Arduino.h>
#define DXL_SERIAL Serial1
#define DEBUG_SERIAL Serial
const int DXL_DIR_PIN = 28;
const uint8_t MotorR_ID = 2;
const uint8_t MotorL_ID = 1;
const uint8_t MotorRear_ID = 3;
const float DXL_PROTOCOL_VERSION = 2.0;
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;


//*********************
//THESE SETTINGS ARE ONLY TO BE USED BY THE COMPUTER
#define pi 3.14

unsigned long int odoMillis = millis();

bool runMethod = true;
bool temp;  //Used to wait for methods
int values[8];
float width;
float position;
int distances[4];
float odoMeter = 0;
float prevPosition = 4;
float prevWidth = 0;
char colorR = 'W';
char colorL = 'W';
float standardWhiteValuesL = 0;
float standardWhiteValuesR = 0;
VL53L0X_RangingMeasurementData_t measure;


bool hasSeenLeft = false;
bool hasSeenRight = false;
bool hasSeenLeftInter = false;
bool hasSeenrightInter = false;
bool conductiveBottom = false;

void setup() {
  pinMode(18, OUTPUT);
  pinMode(19, INPUT);
  DEBUG_SERIAL.begin(115200);
  dxl.begin(1000000);
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  dxl.ping(MotorL_ID);
  dxl.torqueOff(MotorL_ID);
  dxl.setOperatingMode(MotorL_ID, OP_VELOCITY);
  dxl.torqueOn(MotorL_ID);

  dxl.ping(MotorRear_ID);
  dxl.torqueOff(MotorRear_ID);
  dxl.setOperatingMode(MotorRear_ID, OP_VELOCITY);
  dxl.torqueOn(MotorRear_ID);

  dxl.ping(MotorR_ID);
  dxl.torqueOff(MotorR_ID);
  dxl.setOperatingMode(MotorR_ID, OP_VELOCITY);
  dxl.torqueOn(MotorR_ID);
  dxl.setGoalCurrent(MotorR_ID, 1023);

  TCA9548A(1);
  if (colorSensorR.begin()) {
    //Serial.println("Found sensor");
  } else {
    Serial.println("No TCS34725 1 found ... check your connections");
  }

  TCA9548A(0);
  if (colorSensorL.begin()) {
    //Serial.println("Found sensor");
  } else {
    Serial.println("No TCS34725 2 found ... check your connections");
  }
  
  TCA9548A(6);
  if (!sensor2.begin()) {
    Serial.println(F("Failed to boot VL53L0X"));
  }
  sensor2.startRangeContinuous();


  TCA9548A(0);
  colorSensorL.getRGB(&red, &green, &blue);
  colorSensorL.getRawData(&r, &g, &b, &c);

  Serial.print(r);
  Serial.print("    ");
  Serial.print(g);
  Serial.print("    ");
  Serial.print(b);
  Serial.print("    ");
  Serial.println(c);

  standardWhiteValuesR = red;

  TCA9548A(1);
  colorSensorR.getRGB(&red, &green, &blue);
  colorSensorR.getRawData(&r, &g, &b, &c);


  standardWhiteValuesL = red;


  //IMU-----------------
  TCA9548A(4);
  Wire.begin();
  Wire.setClock(400000);  //400khz clock

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
}
unsigned long int lastMilli = millis();
unsigned long int resetMillisRight = millis();
unsigned long int resetMillisLeft = millis();
unsigned long int ramp = millis();
bool done = false;

void loop() {
  if (hasSeenLeftInter || hasSeenrightInter) {
    speed = 18;
    updateColors();
  } 
  else
    speed = 18;

  if (resetMillisRight + 150 <= millis()) {
    resetMillisRight = millis();
    hasSeenrightInter = false;
    hasSeenRight = false;
  }

  if (resetMillisLeft + 150 <= millis()) {
    resetMillisLeft = millis();
    hasSeenLeftInter = false;
    hasSeenLeft = false;
  }

  int start = millis();
  getArray();
  getWidthAndPos();
  getDistances();

  if (width > 4.5 && millis() - lastMilli > 2000){
    lastMilli = millis();
  }

  if (((hasSeenLeftInter || hasSeenrightInter) && colorR == 'G') || ((hasSeenLeftInter || hasSeenrightInter) && colorL == 'G')) {
    bypassMotorOptimization = true;
    Serial.println("reached");
    intersection:
    forwardDrive(speed);

    char tempr = colorR;
    char templ = colorL;

    for (int i = 0; i < 3; i++) {
      updateColors();
      if (colorR == 'G'){
        tempr == 'G';
      }
      if (colorL == 'G'){
        templ == 'G';
      }
    }

    if (tempr == 'G') {
      colorR = 'G';
    }

    if (templ == 'G') {
      colorL = 'G';
    }

    if ((hasSeenLeftInter || hasSeenrightInter) && colorR == 'G' && colorL == 'G') {

      stop();

      pointTurn(-180);

      hasSeenLeft = false;
      hasSeenRight = false;
      colorL = 'W';
      colorR = 'W';

      forwardDrive(10);
      delay(100);
    }
    else {
      if ((hasSeenLeftInter || hasSeenrightInter) && colorL == 'G') {
        resetODO();
        while (abs(odoMeter) < 20) {
          forwardDrive(speed);
          updateDistanceDriven();
        }
        stop();

        pointTurn(-90);
        hasSeenLeft = false;
        colorL = 'W';

        forwardDrive(10);
        delay(100);
      }

      if ((hasSeenLeftInter || hasSeenrightInter) && colorR == 'G') {
        stop();
        resetODO();
        while (abs(odoMeter) < 20) {
          forwardDrive(speed);
          updateDistanceDriven();
        }

        pointTurn(90);
        hasSeenRight = false;
        colorR = 'W';

        forwardDrive(10);
        delay(100);
      }
    }
    bypassMotorOptimization = false;
  } else if (distances[1] <= 50 && distances[1] != 0) {
    Serial.println("Object");
    stop();
    delay(100);

    getDistances();

    /*
    if (width == 0){
      while(true){
        getDistances();
        while (distances[1] > 70){
          getDistances();
          forwardDrive(12);
        }

        pointTurn(135);
      }
    }
    */

    resetODO();
    while (abs(odoMeter) < abs(30 - distances[1])) {
      forwardDrive(mathSign(distances[1] - 30) * 10);
      updateDistanceDriven();
    }

    bypassMotorOptimization = true;

    pointTurn(90);

    getDistances();

    stop();
    delay(500);

    if (distances[1] > 250) {
      getArray();
      getWidthAndPos();

      while (width < 2) {
        radiusDrive(16, 18, true);
        getArray();
        getWidthAndPos();
      }

      resetODO();
      while (abs(odoMeter) < 20) {
        forwardDrive(speed);
        updateDistanceDriven();
      }

      pointTurn(30);

      getDistances();

      distances[1] = 0;
      bypassMotorOptimization = false;
    } else {
      pointTurn(160);
      getArray();
      getWidthAndPos();

      while (width < 2) {
        radiusDrive(16, 18, false);
        getArray();
        getWidthAndPos();
      }

      resetODO();
      while (abs(odoMeter) < 20) {
        forwardDrive(speed);
        updateDistanceDriven();
      }

      pointTurn(-30);

      getDistances();

      distances[1] = 0;
      bypassMotorOptimization = false;
    }
    
  } else if (width <= lineWidth && width != 0) {
    if (position <= 2 && position != 0) {  // Change to adapt to previous line placement
      hasSeenLeftInter = true;
      resetMillisLeft = millis();
      hasSeenrightInter = false;
      speed = 6;
    }

    if (position >= 5) {
      hasSeenrightInter = true;
      resetMillisRight = millis();
      hasSeenLeftInter = false;
      speed = 6;
    }

    bool temp = false;
    if (position < 3.5)
      temp = true;
    radiusDrive(50 * pow(0.15, abs(position - 3.5)) + 5, speed, temp);

  } else if (width > lineWidth) {
    forwardDrive(6);

    if (position < 3.5) {  // Change to adapt to previous line placement
      hasSeenLeftInter = true;
      resetMillisLeft = millis();
      hasSeenrightInter = false;
    }

    if (position >= 3.5) {
      if (!hasSeenrightInter) {
         
      }
      hasSeenrightInter = true;
      resetMillisRight = millis();
      hasSeenLeftInter = false;
    }
  } else if (width == 0) {
    updateColors();
    if (r > g + b) {
      stop();
      delay(12000);
    }

    speed = 8;
    if (hasSeenrightInter) {
      resetODO();

      while (odoMeter < 15) {
        updateColors();
        forwardDrive(speed);
        updateDistanceDriven();
        
        if (r > g + b) {
          stop();
          delay(12000);
        }
        
        if ((colorR == 'B' && hasSeenRight) || (colorL == 'B' && hasSeenLeft)) {
          Serial.println("Shit 2");
          goto intersection;
        }
      }
      stop();

      getArray();
      getWidthAndPos();
      while ((position > 3.5 || width == 0) && (!hasSeenRight || colorR != 'B') && (!hasSeenLeft || colorL != 'B')) {
        radiusDrive(0, speed, false);
        getArray();
        getWidthAndPos();
      }
      stop();
    } else if (hasSeenLeftInter) {
      resetODO();

      while (odoMeter < 15) {
        forwardDrive(speed);
        updateDistanceDriven();
        updateColors();
        if (r > g + b) {
          stop();
          delay(12000);
        }

        if ((colorR == 'B' && hasSeenRight) || (colorL == 'B' && hasSeenLeft)) {
          goto intersection;
        }
      }
      stop();

      getArray();
      getWidthAndPos();
      while (position < 3.5 && (!hasSeenRight || colorR != 'B') && (!hasSeenLeft || colorL != 'B') && !conductiveBottom) {
        radiusDrive(0, speed, true);
        getArray();
        getWidthAndPos();
      }
      stop();
    } else {
      forwardDrive(speed);
    }
     
  }
  int stop = millis();
  
  //Serial.print(stop - start);
  //>Serial.println("ms");
  
}

char prevColorR = 'W';
char prevColorL = 'W';

bool doneRed = false;
unsigned long int colorRTimer = millis();
unsigned long int colorLTimer = millis();

void updateColors() {
  TCA9548A(0);
  colorSensorL.getRawData(&r, &g, &b, &c);
  
  if (g > b && g > r) {
    colorL = 'G';
    Serial.println("Found green L");
  }
  else
  {
    colorL = 'B';
  }

  TCA9548A(1);
  colorSensorR.getRawData(&r, &g, &b, &c);

  if (g > b && g > r) {
    colorR = 'G';
    Serial.println("Found green R");
  }
  else
  {
    colorR = 'B';
  }
}

bool isClose(float x1, float x2, float tolerance) {
  if (abs(x1 - x2) < tolerance)
    return true;
  else
    return false;
}

void forwardDrive(int wantedSpeed) {
  currentSpeedL = wantedSpeed;
  currentSpeedR = wantedSpeed;
  currentSpeedRear = wantedSpeed;
  setSpeedMotors();
}

float prevCurrentSpeedR = 0;
float prevCurrentSpeedRear = 0;
float prevCurrentSpeedL = 0;

void setSpeedMotors() {
  if (!isClose(currentSpeedR, prevCurrentSpeedR, 0.8)) {
    if (currentSpeedR < 0)
      dxl.setGoalVelocity(MotorR_ID, abs(currentSpeedR / (radiusWheelsF * 2 * pi)) * 9 * 60);
    else
      dxl.setGoalVelocity(MotorR_ID, 1024 + (currentSpeedR / (radiusWheelsF * 2 * pi)) * 9 * 60);
  }

  if (!isClose(currentSpeedL, prevCurrentSpeedL, 0.8)) {
    if (currentSpeedL < 0)
      dxl.setGoalVelocity(MotorL_ID, 1024 + abs((currentSpeedL / (radiusWheelsF * 2 * pi)) * 9 * 60));
    else
      dxl.setGoalVelocity(MotorL_ID, ((currentSpeedL / (radiusWheelsF * 2 * pi)) * 9 * 60));
  }

  if (!isClose(currentSpeedRear, prevCurrentSpeedRear, 0.8)) {
    if (currentSpeedRear < 0)
      dxl.setGoalVelocity(MotorRear_ID, 1024 + abs((currentSpeedRear / (radiusWheelsR * 2 * pi)) * 9 * 60));
    else
      dxl.setGoalVelocity(MotorRear_ID, ((currentSpeedRear / (radiusWheelsR * 2 * pi)) * 9 * 60));
  }

  if (bypassMotorOptimization) {
    if (!isClose(currentSpeedR, prevCurrentSpeedR, 0.8)) {
      if (currentSpeedR < 0)
        dxl.setGoalVelocity(MotorR_ID, abs(currentSpeedR / (radiusWheelsF * 2 * pi)) * 9 * 60);
      else
        dxl.setGoalVelocity(MotorR_ID, 1024 + (currentSpeedR / (radiusWheelsF * 2 * pi)) * 9 * 60);
    }

    if (!isClose(currentSpeedL, prevCurrentSpeedL, 0.8)) {
      if (currentSpeedL < 0)
        dxl.setGoalVelocity(MotorL_ID, 1024 + abs((currentSpeedL / (radiusWheelsF * 2 * pi)) * 9 * 60));
      else
        dxl.setGoalVelocity(MotorL_ID, ((currentSpeedL / (radiusWheelsF * 2 * pi)) * 9 * 60));
    }

    if (!isClose(currentSpeedRear, prevCurrentSpeedRear, 0.8)) {
      if (currentSpeedRear < 0)
        dxl.setGoalVelocity(MotorRear_ID, 1024 + abs((currentSpeedRear / (radiusWheelsR * 2 * pi)) * 9 * 60));
      else
        dxl.setGoalVelocity(MotorRear_ID, ((currentSpeedRear / (radiusWheelsR * 2 * pi)) * 9 * 60));
    }
  }

  prevCurrentSpeedR = currentSpeedR;
  prevCurrentSpeedRear = currentSpeedRear;
  prevCurrentSpeedL = currentSpeedL;
}

void getWidthAndPos() {
  float start = 0;
  float end = 0;

  for (int i = 0; i <= 7; i++) {
    if (values[i] - whiteLimit > 0) {
      start = i - (mapF(values[i], whiteLimit, maxBlack, 0, 1)) + 0.5;
      break;
    }
  }

  for (int i = 7; i >= 0; i--) {
    if (values[i] - whiteLimit > 0) {
      end = i + (mapF(values[i], whiteLimit, maxBlack, 0, 1)) - 0.5;
      break;
    }
  }

  if (width <= lineWidth - 1 && width != 0) {
    prevWidth = width;
    prevPosition = position;
  } else {
    prevWidth = 10;
  }

  width = end - start;
  position = (start + end) / 2;

  //Serial.println(position);
  //Serial.println(width);
}

float mapF(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

float getK(int x1, int x2, int y1, int y2) {
  float deltaY = y2 - y1;
  float deltaX = x2 - x1;
  float k = deltaY / deltaX;

  return k;
}

void getDistances() {
  if (true) {
    TCA9548A(6);
    sensor2.rangingTest(&measure, false);
    if (measure.RangeStatus != 4) {  // phase failures have incorrect data
      distances[1] = measure.RangeMilliMeter - 50;
    } else {
      distances[1] = 0;
    }
    /* Debugging
    Serial.println(distances[2]);
    Serial.println("---");
    Serial.println(distances[0]);
    Serial.println("---");
    Serial.println(distances[2] - distances[0]);
    */
  }
}

void getArray() {
  values[0] = analogRead(7);
  values[1] = analogRead(6);
  values[2] = analogRead(5);
  values[3] = analogRead(4);
  values[4] = analogRead(3);
  values[5] = analogRead(2);
  values[6] = analogRead(1);
  values[7] = analogRead(0);

  //Serial.println(width);





  /*Debugging
  Serial.print(values[0]);
  Serial.print("   ");
  Serial.print(values[1]);
  Serial.print("   ");
  Serial.print(values[2]);
  Serial.print("   ");
  Serial.print(values[3]);
  Serial.print("   ");
  Serial.print(values[4]);
  Serial.print("   ");
  Serial.print(values[5]);
  Serial.print("   ");
  Serial.print(values[6]);
  Serial.print("   ");
  Serial.println(values[7]);
  Serial.print("   ");
  */
}

void radiusDrive(float radius, int wantedSpeedOutside, bool reversed) {
  float degreesPerSecond;
  if (reversed) {
    float outerRadius = radius + (wheelBaseWidth / 2);
    currentSpeedR = wantedSpeedOutside;
    //currentSpeedR += mathSign(wantedSpeedOutside - currentSpeedR) * 2;
    degreesPerSecond = wantedSpeedOutside / (outerRadius * 2 * pi) * 360;
    float innerRadius = (radius - (wheelBaseWidth / 2));
    currentSpeedL = degreesPerSecond * ((innerRadius * 2 * pi) / 360);
  } else {
    float outerRadius = radius + (wheelBaseWidth / 2);
    currentSpeedL = wantedSpeedOutside;
    //currentSpeedL += mathSign(wantedSpeedOutside - currentSpeedL) * 2;
    degreesPerSecond = wantedSpeedOutside / (outerRadius * 2 * pi) * 360;
    float innerRadius = (radius - (wheelBaseWidth / 2));
    currentSpeedR = degreesPerSecond * ((innerRadius * 2 * pi) / 360);
  }

  float radiusRear = radius + (sqrtf((radius * radius) + (wheelBaseLength * wheelBaseLength)) - radius);
  float angleOfAttack = 90 - (atanf(radiusRear / wheelBaseLength));
  float speedTangent = degreesPerSecond * ((radiusRear * 2 * pi) / 360);
  currentSpeedRear = speedTangent * cosf(angleOfAttack);

  if (radius == 0) {
    currentSpeedRear = 0;
  }

  if (currentSpeedRear > 28) {
    radiusDrive(radius, wantedSpeedOutside - 1, reversed);
    return;
  }

  setSpeedMotors();
}

void pointTurn(float degrees) {
  bool reversed = false;
  float degreesPerSecond = 0;
  float degreesDone = 0;
  unsigned long int millies = millis();

  if (degrees < 0)
    reversed = true;

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
    Serial.print(rotatedDegrees);
    Serial.print("    ");
    Serial.println(IMUGyro.gyroX);
    radiusDrive(0, 12, reversed);
  }

  stop();
}

int mathSign(int x) {
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;
}

void updateDistanceDriven() {
  odoMeter += ((millis() - odoMillis) * ((currentSpeedL + currentSpeedR) / 2)) / 100;
  odoMillis = millis();
}

void resetODO() {
  odoMeter = 0;
  odoMillis = millis();
}

void stop() {
  currentSpeedL = 0;
  currentSpeedR = 0;
  currentSpeedRear = 0;
  setSpeedMotors();
}

void TCA9548A(uint8_t bus) {
  Wire.beginTransmission(0x70);  // TCA9548A address is 0x70
  Wire.write(1 << bus);          // send byte to select bus
  Wire.endTransmission();
}