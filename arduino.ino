// Nose-following single-servo controller
// Protocol (from PC): NOSE,<x_px>,<y_px>,<yaw_deg>,<pitch_deg>\r\n

#include <Servo.h>

// Single servo pin (use one servo only)
const uint8_t SERVO_PIN = 9;  // change as needed

// Control mode: use yaw (pan-like) or pitch (tilt-like)
enum ControlMode { USE_YAW, USE_PITCH };
ControlMode mode = USE_YAW;   // default: yaw

// Servo center and limits
int servoCenter = 90;
int servoMin    = 10,  servoMax = 170;

// Mapping from angles to servo range
// Assuming camera HFOV ~60°, VFOV ~35° (match PC side)
const float YAW_RANGE_DEG   = 30.0f;   // +/- range mapped from yaw
const float PITCH_RANGE_DEG = 20.0f;   // +/- range mapped from pitch

// Motion smoothing
const float SMOOTH_ALPHA = 0.25f; // 0..1 (higher = faster)
const float DEADZONE_DEG = 1.0f;  // ignore tiny movements

// Invert direction if needed
bool invertDir = false;  // set true if it moves opposite

Servo servo;
float targetServo = 90;
float smoothedServo = 90;

static inline int clampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  servo.attach(SERVO_PIN);
  servo.write(servoCenter);
  smoothedServo = targetServo = servoCenter;

  Serial.println("READY_SERVO");
}

static inline void updateServo() {
  // EMA smoothing
  smoothedServo = (1.0f - SMOOTH_ALPHA) * smoothedServo + SMOOTH_ALPHA * targetServo;

  int outVal = clampInt((int)round(smoothedServo), servoMin, servoMax);
  servo.write(outVal);
}

static inline void setTargetFromAngles(float yawDeg, float pitchDeg) {
  // choose source angle based on mode
  float a = (mode == USE_YAW) ? yawDeg : pitchDeg;
  if (fabs(a) < DEADZONE_DEG) a = 0;
  if (invertDir) a = -a;

  float range = (mode == USE_YAW) ? YAW_RANGE_DEG : PITCH_RANGE_DEG;
  float span  = (mode == USE_YAW) ? 45.0f : 35.0f; // movement span around center

  float norm = a / range; // -1..+1
  if (norm < -1) norm = -1; else if (norm > 1) norm = 1;
  targetServo = servoCenter + norm * span;
}

void loop() {
  // Read any incoming line
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      if (line.startsWith("NOSE,")) {
        int p1 = line.indexOf(',', 5);
        int p2 = line.indexOf(',', p1 + 1);
        int p3 = line.indexOf(',', p2 + 1);
        if (p1 > 0 && p2 > p1 && p3 > p2) {
          // x, y not used for control currently
          /* float x = */ line.substring(5, p1).toFloat();
          /* float y = */ line.substring(p1 + 1, p2).toFloat();
          float yaw   = line.substring(p2 + 1, p3).toFloat();
          float pitch = line.substring(p3 + 1).toFloat();

          setTargetFromAngles(yaw, pitch);
        }
      } else if (line.startsWith("CFG,")) {
        // Example: CFG,center,pan,95  or CFG,limit,tilt,25,155
        // Simple config interface
        int p1 = line.indexOf(',', 4);
        int p2 = line.indexOf(',', p1 + 1);
        if (p1 > 0 && p2 > p1) {
          String key = line.substring(4, p1);
          String sub = line.substring(p1 + 1, p2);
          String rest = line.substring(p2 + 1);
          if (key == "center") {
            int v = rest.toInt();
            if (sub == "servo") servoCenter = clampInt(v, 0, 180);
          } else if (key == "limit") {
            int comma2 = rest.indexOf(',');
            if (comma2 > 0) {
              int v1 = rest.substring(0, comma2).toInt();
              int v2 = rest.substring(comma2 + 1).toInt();
              if (sub == "servo") { servoMin = clampInt(v1, 0, 180); servoMax = clampInt(v2, 0, 180); }
            }
          } else if (key == "invert") {
            if (sub == "servo") invertDir = (rest == "1" || rest == "true");
          } else if (key == "mode") {
            // CFG,mode,source,yaw  or CFG,mode,source,pitch
            if (sub == "source") {
              if (rest == "yaw") mode = USE_YAW; else if (rest == "pitch") mode = USE_PITCH;
            }
          }
        }
      }
    }
  }

  updateServo();
}


