#include <Arduino.h>
#include <ESP32Servo.h>

const int ENA_PIN    = 25;
const int IN1_PIN    = 26;
const int IN2_PIN    = 27;

const int SERVO1_PIN = 18;
const int SERVO2_PIN = 19;

const int PWM_FREQ = 5000;
const int PWM_RES  = 8;

Servo servo1;
Servo servo2;

// 모터 자동 반복 상태
bool motorAuto = false;
bool motorOnPhase = false;
int motorSpeed = 180;

// 시간 설정
unsigned long lastToggleTime = 0;
const unsigned long motorOnTime  = 200;
const unsigned long motorOffTime = 600;

void motorRun() {
  digitalWrite(IN1_PIN, HIGH);
  digitalWrite(IN2_PIN, LOW);
  ledcWrite(ENA_PIN, motorSpeed);
}

void motorStop() {
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);
  ledcWrite(ENA_PIN, 0);
}

void setup() {
  Serial.begin(115200);

  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);

  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);

  // 서보1
  servo1.setPeriodHertz(50);
  servo1.attach(SERVO1_PIN, 500, 2400);
  servo1.write(90);

  // 서보2
  servo2.setPeriodHertz(50);
  servo2.attach(SERVO2_PIN, 500, 2400);
  servo2.write(90);

  bool ok = ledcAttach(ENA_PIN, PWM_FREQ, PWM_RES);
  Serial.print("ledcAttach = ");
  Serial.println(ok ? "OK" : "FAIL");

  Serial.println("=== Servo1 ===");
  Serial.println("n = 90, d = 0, r = 180");

  Serial.println("=== Servo2 ===");
  Serial.println("j = 90, k = 0, l = 180");

  Serial.println("=== Motor ===");
  Serial.println("m = auto start");
  Serial.println("s = stop");
}

void loop() {
  // ===== 시리얼 명령 처리 =====
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '\n' || cmd == '\r') return;

    // ----- 서보1 -----
    if (cmd == 'i') {
      servo1.write(90);
      Serial.println("SERVO1 -> 90");
    }
    else if (cmd == 'o') {
      servo1.write(0);
      Serial.println("SERVO1 -> 0");
    }
    else if (cmd == 'p') {
      servo1.write(180);
      Serial.println("SERVO1 -> 180");
    }

    // ----- 서보2 -----
    else if (cmd == 'j') {
      servo2.write(90);
      Serial.println("SERVO2 -> 90");
    }
    else if (cmd == 'k') {
      servo2.write(0);
      Serial.println("SERVO2 -> 0");
    }
    else if (cmd == 'l') {
      servo2.write(180);
      Serial.println("SERVO2 -> 180");
    }

    // ----- 모터 -----
    else if (cmd == 'm') {
      motorAuto = true;
      motorOnPhase = true;
      lastToggleTime = millis();
      motorRun();
      Serial.println("MOTOR AUTO START");
    }
    else if (cmd == 's') {
      motorAuto = false;
      motorOnPhase = false;
      motorStop();
      Serial.println("MOTOR STOP");
    }
  }

  // ===== 모터 자동 반복 =====
  if (motorAuto) {
    unsigned long now = millis();

    if (motorOnPhase) {
      if (now - lastToggleTime >= motorOnTime) {
        motorOnPhase = false;
        lastToggleTime = now;
        motorStop();
      }
    } else {
      if (now - lastToggleTime >= motorOffTime) {
        motorOnPhase = true;
        lastToggleTime = now;
        motorRun();
      }
    }
  }
}


/*
#include <Arduino.h>
#include <ESP32Servo.h>

const int ENA_PIN   = 25;
const int IN1_PIN   = 26;
const int IN2_PIN   = 27;
const int SERVO_PIN = 18;

const int PWM_FREQ = 5000;
const int PWM_RES  = 8;

Servo myServo;

// 모터 자동 반복 상태
bool motorAuto = false;       // m 누르면 true, s 누르면 false
bool motorOnPhase = false;    // 현재 회전 중인지, 정지 중인지
int motorSpeed = 180;

// 시간 설정
unsigned long lastToggleTime = 0;
const unsigned long motorOnTime  = 200;  // 200ms 회전
const unsigned long motorOffTime = 600;  // 600ms 정지

void motorRun() {
  digitalWrite(IN1_PIN, HIGH);
  digitalWrite(IN2_PIN, LOW);
  ledcWrite(ENA_PIN, motorSpeed);
}

void motorStop() {
  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);
  ledcWrite(ENA_PIN, 0);
}

void setup() {
  Serial.begin(115200);

  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);

  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);

  myServo.setPeriodHertz(50);
  myServo.attach(SERVO_PIN, 500, 2400);

  bool ok = ledcAttach(ENA_PIN, PWM_FREQ, PWM_RES);
  Serial.print("ledcAttach = ");
  Serial.println(ok ? "OK" : "FAIL");

  myServo.write(90);

  Serial.println("n / d / r / m / s");
}

void loop() {
  // ===== 시리얼 명령 처리 =====
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '\n' || cmd == '\r') return;

    if (cmd == 'n') {
      myServo.write(90);
      Serial.println("NORMAL");
    }
    else if (cmd == 'd') {
      myServo.write(0);
      Serial.println("DEFECT");
    }
    else if (cmd == 'r') {
      myServo.write(180);
      Serial.println("RECHECK");
    }
    else if (cmd == 'm') {
      motorAuto = true;
      motorOnPhase = true;
      lastToggleTime = millis();
      motorRun();
      Serial.println("MOTOR AUTO START");
    }
    else if (cmd == 's') {
      motorAuto = false;
      motorOnPhase = false;
      motorStop();
      Serial.println("MOTOR STOP");
    }
  }

  // ===== 모터 자동 반복 =====
  if (motorAuto) {
    unsigned long now = millis();

    if (motorOnPhase) {
      // 현재 회전 중
      if (now - lastToggleTime >= motorOnTime) {
        motorOnPhase = false;
        lastToggleTime = now;
        motorStop();
      }
    } else {
      // 현재 정지 중
      if (now - lastToggleTime >= motorOffTime) {
        motorOnPhase = true;
        lastToggleTime = now;
        motorRun();
      }
    }
  }
}

*/