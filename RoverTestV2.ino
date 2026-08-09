/*
  Obstacle-Avoiding Rover (MDD3A driver)
  ---------------------------------------
  Algorithm:
    - Nothing ahead        -> go straight
    - Something ahead:
        - Left is clear    -> turn left
        - Left is blocked  -> turn right

  Wiring:
  3x Ultrasonic sensors GND -> Negative rail
  3x Ultrasonic sensors VCC -> Breadboard 5V (from Arduino 5V)

  Trig Right  (green) -> 10   Echo Right  -> 11
  Trig Middle (blue)  -> 2    Echo Middle -> 6
  Trig Left   (black) -> 12   Echo Left   -> 13

  MDD3A "2 PWM Input" mode:
  Motor 1 (assumed right side): M1A -> 9, M1B -> 3
  Motor 2 (assumed left side):  M2A -> 5, M2B -> 8

  If left/right or forward/backward come out reversed once running,
  swap the two pins for that motor, or swap turnLeft()/turnRight().
*/

// ---------------------------------------------------------------
// PIN DEFINITIONS
// ---------------------------------------------------------------

// Motor 1 (right side)
const int M1A = 9;
const int M1B = 3;

// Motor 2 (left side)
const int M2A = 5;
const int M2B = 8;

// Ultrasonic sensors
const int TRIG_MID   = 2;
const int ECHO_MID   = 6;
const int TRIG_RIGHT = 10;
const int ECHO_RIGHT = 11;
const int TRIG_LEFT  = 12;
const int ECHO_LEFT  = 13;

// ---------------------------------------------------------------
// TUNING CONSTANTS
// ---------------------------------------------------------------

const int DRIVE_SPEED = 120;   // 0-255
const int TURN_SPEED  = 120;   // 0-255

const int OBSTACLE_DISTANCE_CM = 30;   // trigger avoidance below this
const unsigned long TURN_TIME_MS = 500;

// =================================================================
// PART 1: CAR MOVEMENT (MDD3A - 2 PWM Input mode)
// =================================================================

void stopMotors() {
  analogWrite(M1A, 0);
  analogWrite(M1B, 0);
  analogWrite(M2A, 0);
  analogWrite(M2B, 0);
}

void moveForward(int speed) {
  analogWrite(M1A, speed);
  analogWrite(M1B, 0);
  analogWrite(M2A, speed);
  analogWrite(M2B, 0);
}

void moveBackward(int speed) {
  analogWrite(M1A, 0);
  analogWrite(M1B, speed);
  analogWrite(M2A, 0);
  analogWrite(M2B, speed);
}

void turnLeft(int speed) {
  analogWrite(M1A, speed);  // right side forward
  analogWrite(M1B, 0);
  analogWrite(M2A, 0);      // left side reverse
  analogWrite(M2B, speed);
}

void turnRight(int speed) {
  analogWrite(M1A, 0);      // right side reverse
  analogWrite(M1B, speed);
  analogWrite(M2A, speed);  // left side forward
  analogWrite(M2B, 0);
}

void setupMotors() {
  pinMode(M1A, OUTPUT);
  pinMode(M1B, OUTPUT);
  pinMode(M2A, OUTPUT);
  pinMode(M2B, OUTPUT);
  stopMotors();
}

// =================================================================
// PART 2: OBSTACLE AVOIDANCE
// =================================================================

void setupUltrasonic() {
  pinMode(TRIG_MID, OUTPUT);
  pinMode(ECHO_MID, INPUT);
  pinMode(TRIG_RIGHT, OUTPUT);
  pinMode(ECHO_RIGHT, INPUT);
  pinMode(TRIG_LEFT, OUTPUT);
  pinMode(ECHO_LEFT, INPUT);
}

// Returns distance in cm, or 400 (treated as "clear") if no echo received.
long readDistanceCM(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 30000); // 30ms timeout ~5m range

  if (duration == 0) {
    return 400;
  }

  return duration * 0.0343 / 2;
}

long getMiddleDistance() { return readDistanceCM(TRIG_MID, ECHO_MID); }
long getRightDistance()  { return readDistanceCM(TRIG_RIGHT, ECHO_RIGHT); }
long getLeftDistance()   { return readDistanceCM(TRIG_LEFT, ECHO_LEFT); }

void avoidObstacle() {
  stopMotors();
  delay(100);

  long leftDist = getLeftDistance();

  Serial.print("Left: ");
  Serial.print(leftDist);
  Serial.println(" cm");

  if (leftDist > OBSTACLE_DISTANCE_CM) {
    Serial.println("Turning left");
    turnLeft(TURN_SPEED);
  } else {
    Serial.println("Turning right");
    turnRight(TURN_SPEED);
  }

  delay(TURN_TIME_MS);
  stopMotors();
}

void setup() {
  Serial.begin(9600);
  setupMotors();
  setupUltrasonic();
  delay(1000); // let sensors settle
}

void loop() {
  long midDist = getMiddleDistance();

  Serial.print("Forward: ");
  Serial.print(midDist);
  Serial.println(" cm");

  if (midDist < OBSTACLE_DISTANCE_CM) {
    avoidObstacle();
  } else {
    moveForward(DRIVE_SPEED);
  }
}
