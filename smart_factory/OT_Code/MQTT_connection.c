#include <WiFi.h>
#include <PubSubClient.h>
#include <ESP32Servo.h>
#include <time.h>
#include <sys/time.h>

// =========================================================
// 1. 사용자 설정
// =========================================================
const char* ssid       = "rapa_classroom-4";
const char* password   = "rapa6074";

const char* mqttServer = "broker.hivemq.com";
const int   mqttPort   = 1883;

const char* subTopic    = "myfactory/0811/box/result";
const char* statusTopic = "myfactory/0811/box/status";

// NTP 시간 설정 (한국 시간)
const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 9 * 3600;   // UTC+9
const int   daylightOffset_sec = 0;

// =========================================================
// 2. 핀 설정
// =========================================================
const int ENA_PIN    = 25;
const int IN1_PIN    = 26;
const int IN2_PIN    = 27;

const int SERVO1_PIN = 18;
const int SERVO2_PIN = 19;

const int PWM_FREQ = 5000;
const int PWM_RES  = 8;

// =========================================================
// 3. 객체 생성
// =========================================================
WiFiClient espClient;
PubSubClient client(espClient);
Servo servo1;
Servo servo2;

// =========================================================
// 4. 서보 설정
// =========================================================
const int SERVO1_DEFAULT_ANGLE = 150;
const int SERVO1_ACTIVE_ANGLE  = 0;

const int SERVO2_DEFAULT_ANGLE = 30;
const int SERVO2_ACTIVE_ANGLE  = 180;

const unsigned long SERVO_HOLD_TIME = 150; //1500

// 중복 메시지 방지
String lastProcessedMessage = "";
unsigned long lastProcessedTime = 0;
const unsigned long DUPLICATE_BLOCK_MS = 2000; //2000

// 서보 동작 상태
bool servoActionActive = false;
int activeServo = 0; // 0:none, 1:servo1, 2:servo2
unsigned long servoActionStartTime = 0;

// 로그용 변수
unsigned long currentMessageReceivedTime = 0;
unsigned long currentServoStartTime = 0;
String currentMessage = "";

// =========================================================
// 5. 컨베이어 설정
// =========================================================
bool motorAuto = false;
bool motorOnPhase = false;
int motorSpeed = 174;

unsigned long lastToggleTime = 0;
const unsigned long motorOnTime  = 200;
const unsigned long motorOffTime = 600;

// =========================================================
// 6. 시간 함수
// =========================================================
String getIsoTimestamp() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);

  struct tm timeinfo;
  localtime_r(&tv.tv_sec, &timeinfo);

  char dateBuf[32];
  strftime(dateBuf, sizeof(dateBuf), "%Y-%m-%dT%H:%M:%S", &timeinfo);

  char finalBuf[40];
  snprintf(finalBuf, sizeof(finalBuf), "%s.%06ld", dateBuf, (long)tv.tv_usec);

  return String(finalBuf);
}

void setupTime() {
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

  Serial.print("[TIME] NTP 동기화 중");
  for (int i = 0; i < 20; i++) {
    struct tm timeinfo;
    if (getLocalTime(&timeinfo)) {
      Serial.println(" 성공");
      Serial.print("[TIME] 현재 시각: ");
      Serial.println(getIsoTimestamp());
      return;
    }
    delay(500);
    Serial.print(".");
  }
  Serial.println(" 실패");
}

// =========================================================
// 7. 모터 함수
// =========================================================
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

void startMotorAuto() {
  motorAuto = true;
  motorOnPhase = true;
  lastToggleTime = millis();
  motorRun();
  Serial.println("[MOTOR] 자동 시작");
  client.publish(statusTopic, "motor_auto_start");
}

void stopMotorAuto() {
  motorAuto = false;
  motorOnPhase = false;
  motorStop();
  Serial.println("[MOTOR] 정지");
  client.publish(statusTopic, "motor_stop");
}

// =========================================================
// 8. 서보 함수
// =========================================================
void resetServos() {
  servo1.write(SERVO1_DEFAULT_ANGLE);
  servo2.write(SERVO2_DEFAULT_ANGLE);
}

void startServoAction(int whichServo, int angle, const char* statusMsg) {
  servoActionActive = true;
  activeServo = whichServo;
  servoActionStartTime = millis();
  currentServoStartTime = servoActionStartTime;

  if (whichServo == 1) {
    servo1.write(angle);
  }
  else if (whichServo == 2) {
    servo2.write(angle);
  }

  client.publish(statusTopic, statusMsg);
}

// =========================================================
// 9. WiFi 연결
// =========================================================
void setupWifi() {
  delay(10);
  Serial.println();
  Serial.print("[WIFI] 연결 시도: ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("[WIFI] 연결 성공");
  Serial.print("[WIFI] IP: ");
  Serial.println(WiFi.localIP());
}

// =========================================================
// 10. MQTT 콜백
// =========================================================
void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  unsigned long now = millis();
  String recvTimestamp = getIsoTimestamp();

  currentMessage = message;
  currentMessageReceivedTime = now;

  // 수신 순간 타임스탬프 로그
  Serial.printf("[RECV] %s | topic=%s | message=%s\n",
                recvTimestamp.c_str(),
                topic,
                message.c_str());

  // 같은 메시지가 짧은 시간 안에 또 들어오면 무시
  if (message == lastProcessedMessage &&
      (now - lastProcessedTime) < DUPLICATE_BLOCK_MS) {
    Serial.printf("[%-10s] 수신=%lums 처리=중복무시\n",
                  message.c_str(), now);
    return;
  }

  // 서보가 현재 동작 중이면 새 메시지 무시
  if (servoActionActive) {
    Serial.printf("[%-10s] 수신=%lums 처리=동작중무시\n",
                  message.c_str(), now);
    return;
  }

  lastProcessedMessage = message;
  lastProcessedTime = now;

  if (message == "NORMAL") {
    resetServos();
    client.publish(statusTopic, "normal_pass");
    Serial.printf("[%-10s] 수신=%lums 처리=완료\n",
                  "NORMAL", now);
  }
  else if (message == "DEFECTIVE") {
    startServoAction(1, SERVO1_ACTIVE_ANGLE, "defect_sorting");
  }
  else if (message == "RE_INSPECT") {
    startServoAction(2, SERVO2_ACTIVE_ANGLE, "reinspect_sorting");
  }
  else {
    Serial.printf("[%-10s] 수신=%lums 처리=알수없음\n",
                  message.c_str(), now);
  }
}

// =========================================================
// 11. MQTT 재연결
// =========================================================
void reconnect() {
  while (!client.connected()) {
    Serial.print("[MQTT] 연결 시도 중...");

    String clientId = "ESP32Client-" + String(random(0xffff), HEX);

    if (client.connect(clientId.c_str())) {
      Serial.println("성공");
      client.subscribe(subTopic);
      client.publish(statusTopic, "connected");

      Serial.print("[MQTT] 구독 토픽: ");
      Serial.println(subTopic);
    } else {
      Serial.print("실패, rc=");
      Serial.print(client.state());
      Serial.println(" -> 5초 후 재시도");
      delay(5000);
    }
  }
}

// =========================================================
// 12. setup
// =========================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  randomSeed(micros());

  pinMode(IN1_PIN, OUTPUT);
  pinMode(IN2_PIN, OUTPUT);

  digitalWrite(IN1_PIN, LOW);
  digitalWrite(IN2_PIN, LOW);

  // 서보 초기화
  servo1.setPeriodHertz(50);
  servo1.attach(SERVO1_PIN, 500, 2400);
  servo1.write(SERVO1_DEFAULT_ANGLE);

  servo2.setPeriodHertz(50);
  servo2.attach(SERVO2_PIN, 500, 2400);
  servo2.write(SERVO2_DEFAULT_ANGLE);

  // PWM 초기화
  bool ok = ledcAttach(ENA_PIN, PWM_FREQ, PWM_RES);
  Serial.print("[PWM] ledcAttach = ");
  Serial.println(ok ? "OK" : "FAIL");

  Serial.println("=== SYSTEM READY ===");
  Serial.println("MQTT:");
  Serial.println("  NORMAL      -> 기본 위치");
  Serial.println("  DEFECTIVE   -> Servo1: 150 -> 30 -> 150");
  Serial.println("  RE_INSPECT  -> Servo2: 30 -> 150 -> 30");
  Serial.println();
  Serial.println("SERIAL CMD:");
  Serial.println("  m -> 컨베이어 자동 시작");
  Serial.println("  s -> 컨베이어 정지");
  Serial.println("  i -> servo1 150");
  Serial.println("  o -> servo1 30");
  Serial.println("  p -> servo1 180");
  Serial.println("  j -> servo2 30");
  Serial.println("  k -> servo2 150");
  Serial.println("  l -> servo2 180");

  setupWifi();
  setupTime();

  client.setServer(mqttServer, mqttPort);
  client.setCallback(callback);
}

// =========================================================
// 13. loop
// =========================================================
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  unsigned long now = millis();

  // -------------------------------------------------------
  // A. 시리얼 명령 처리
  // -------------------------------------------------------
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd != '\n' && cmd != '\r') {
      if (cmd == 'm') {
        startMotorAuto();
      }
      else if (cmd == 's') {
        stopMotorAuto();
      }
      else if (cmd == 'i') {
        servo1.write(SERVO1_DEFAULT_ANGLE);
        Serial.println("[SERVO1] 150");
      }
      else if (cmd == 'o') {
        servo1.write(SERVO1_ACTIVE_ANGLE);
        Serial.println("[SERVO1] 30");
      }
      else if (cmd == 'p') {
        servo1.write(180);
        Serial.println("[SERVO1] 180");
      }
      else if (cmd == 'j') {
        servo2.write(SERVO2_DEFAULT_ANGLE);
        Serial.println("[SERVO2] 30");
      }
      else if (cmd == 'k') {
        servo2.write(SERVO2_ACTIVE_ANGLE);
        Serial.println("[SERVO2] 150");
      }
      else if (cmd == 'l') {
        servo2.write(180);
        Serial.println("[SERVO2] 180");
      }
    }
  }

  // -------------------------------------------------------
  // B. 서보 자동 복귀
  // -------------------------------------------------------
  if (servoActionActive) {
    if (now - servoActionStartTime >= SERVO_HOLD_TIME) {
      unsigned long endTime = millis();
      unsigned long startDelay = currentServoStartTime - currentMessageReceivedTime;
      unsigned long totalTime = endTime - currentMessageReceivedTime;

      resetServos();
      servoActionActive = false;
      activeServo = 0;

      Serial.printf("[%-10s] 수신=%lums 시작지연=%lums 총시간=%lums 처리=완료\n",
                    currentMessage.c_str(),
                    currentMessageReceivedTime,
                    startDelay,
                    totalTime);

      client.publish(statusTopic, "servo_done");
    }
  }

  // -------------------------------------------------------
  // C. 컨베이어 자동 반복
  // -------------------------------------------------------
  if (motorAuto) {
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