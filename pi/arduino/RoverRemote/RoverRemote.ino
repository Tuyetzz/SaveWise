/*
  Remote-Controlled Rover (MDD3A driver) — commanded by the Raspberry Pi
  ----------------------------------------------------------------------
  Successor to RoverTestV2.ino. The Pi (pi/rover_pi.py) holds a 3-bit
  command code on three GPIO pins; this sketch reads it and drives the
  motors. Code 5 hands control back to the original autonomous
  obstacle-avoidance loop.

  Command code (B2 B1 B0):
    0 stop | 1 forward | 2 backward | 3 left | 4 right | 5 autonomous
    6, 7   -> stop (7 is what the pullups read when the Pi is off or a
              wire is out — that makes "unplugged" fail safe)

  Wiring added on top of RoverTestV2:
    Pi GPIO17 (bit0) -> A0
    Pi GPIO27 (bit1) -> A1
    Pi GPIO22 (bit2) -> A2
    Pi GND           -> Arduino GND   (REQUIRED — common ground)
  The Pi's 3.3 V high is a valid HIGH for a 5 V Arduino input.

  Existing wiring (unchanged from RoverTestV2):
    3x ultrasonic: VCC -> 5V, GND -> GND
    Trig Right  (green) -> 10   Echo Right  -> 11
    Trig Middle (blue)  -> 2    Echo Middle -> 6
    Trig Left   (black) -> 12   Echo Left   -> 13
    MDD3A "2 PWM Input" mode:
    Motor 1 (right): M1A -> 9, M1B -> 3
    Motor 2 (left):  M2A -> 5, M2B -> 8

  Safety: a manual "forward" is still gated by the middle ultrasonic —
  the rover refuses to drive into anything closer than FORWARD_BLOCK_CM.
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

// Command bus from the Pi
const int CMD_B0 = A0;
const int CMD_B1 = A1;
const int CMD_B2 = A2;

// ---------------------------------------------------------------
// TUNING CONSTANTS
// ---------------------------------------------------------------

const int DRIVE_SPEED = 120;   // 0-255
const int TURN_SPEED  = 120;   // 0-255

const int OBSTACLE_DISTANCE_CM = 30;   // autonomous avoidance trigger
const int FORWARD_BLOCK_CM     = 20;   // manual forward hard stop
const unsigned long TURN_TIME_MS = 500;

// Command codes
const int CMD_STOP     = 0;
const int CMD_FORWARD  = 1;
const int CMD_BACKWARD = 2;
const int CMD_LEFT     = 3;
const int CMD_RIGHT    = 4;
const int CMD_AUTO     = 5;

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
// PART 2: ULTRASONIC SENSORS
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

// =================================================================
// PART 3: COMMAND BUS FROM THE PI
// =================================================================

void setupCommandBus() {
  // Pullups make the idle/unplugged state read 0b111 = 7 -> stop.
  pinMode(CMD_B0, INPUT_PULLUP);
  pinMode(CMD_B1, INPUT_PULLUP);
  pinMode(CMD_B2, INPUT_PULLUP);
}

int readCode() {
  return (digitalRead(CMD_B0) == HIGH ? 1 : 0)
       | (digitalRead(CMD_B1) == HIGH ? 2 : 0)
       | (digitalRead(CMD_B2) == HIGH ? 4 : 0);
}

// The Pi cannot set all three pins at once, so a change can briefly show a
// mixed code. Two identical reads 5 ms apart = settled; -1 = still changing.
int readCodeDebounced() {
  int first = readCode();
  delay(5);
  int second = readCode();
  return (first == second) ? first : -1;
}

// =================================================================
// PART 4: AUTONOMOUS MODE (the original RoverTestV2 behaviour)
// =================================================================

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

void autonomousStep() {
  long midDist = getMiddleDistance();

  if (midDist < OBSTACLE_DISTANCE_CM) {
    avoidObstacle();
  } else {
    moveForward(DRIVE_SPEED);
  }
}

// =================================================================
// MAIN
// =================================================================

int lastCode = CMD_STOP;

void setup() {
  Serial.begin(9600);
  setupMotors();
  setupUltrasonic();
  setupCommandBus();
  delay(1000); // let sensors settle
  Serial.println("RoverRemote ready — waiting for Pi commands");
}

void loop() {
  int code = readCodeDebounced();
  if (code < 0) {
    return; // mid-transition — keep doing what we were doing
  }
  if (code != lastCode) {
    Serial.print("command code: ");
    Serial.println(code);
    lastCode = code;
  }

  switch (code) {
    case CMD_FORWARD: {
      // Manual forward still refuses to ram what's straight ahead.
      long midDist = getMiddleDistance();
      if (midDist < FORWARD_BLOCK_CM) {
        stopMotors();
      } else {
        moveForward(DRIVE_SPEED);
      }
      break;
    }
    case CMD_BACKWARD:
      moveBackward(DRIVE_SPEED);
      break;
    case CMD_LEFT:
      turnLeft(TURN_SPEED);
      break;
    case CMD_RIGHT:
      turnRight(TURN_SPEED);
      break;
    case CMD_AUTO:
      autonomousStep();
      break;
    case CMD_STOP:
    default: // 6 and 7 included: unknown or unplugged -> stop
      stopMotors();
      break;
  }
}
