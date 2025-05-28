#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoJson.h>
#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int PAN_PIN = D4;
const int TILT_PIN = D5;
const int BUZZER_PIN = D3;

// Floor Positions
struct FloorPosition {
  int panAngle;
  int tiltAngle;
  const char* direction;
};

FloorPosition floorPositions[] = {
  {6, 145, "Back"},
  {92, 145, "Left"},
  {6, 35, "Front"},
  {92, 35, "Right"}
};

const int INITIAL_PAN = 6;
const int INITIAL_TILT = 89;

// Known WiFi Networks
struct WiFiNetwork {
  const char* ssid;
  const char* password;
};

// List of WiFi credentials
WiFiNetwork wifiList[] = {
  {"CALVIN-Student", "CITStudentsOnly"} // Add more networks here if needed
};

const int NUM_WIFI = sizeof(wifiList) / sizeof(WiFiNetwork);

ESP8266WebServer server(80);

void setup() {
  Serial.begin(115200);
  
  // Initialize buzzer
  pinMode(BUZZER_PIN, OUTPUT);
  
  readyBeep();

  // Connect to WiFi automatically
  connectToWiFi();

  // Attach servos
  panServo.attach(PAN_PIN, 500, 2500);
  tiltServo.attach(TILT_PIN, 500, 2500);
  
  // Reset to initial position on startup
  panServo.write(INITIAL_PAN);
  tiltServo.write(INITIAL_TILT);
  delay(1000);  // Give servos time to reach position
  
  // Detach pan servo to prevent shaking while maintaining position
  panServo.detach();
  
  // Define API endpoint
  server.on("/sort", HTTP_POST, handleSortRequest);

  // Define STATUS endpoint
  server.on("/status", HTTP_GET, handleStatusRequest);

  // Start server
  server.begin();
  Serial.println("Server started");
  printHelp();
  delay(1500);
  readyBeep();
}

void loop() {
  server.handleClient();
  
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "ip") {
      Serial.println("Device IP: " + WiFi.localIP().toString());
    } 
    else if (command == "wifi") {
      Serial.println("Connected WiFi SSID: " + WiFi.SSID());
    }
    else if (command == "help") {
      printHelp();
    }
    else if (command.startsWith("sort ")) {
      // Extract floor number from command
      int floorNumber = command.substring(5).toInt();
      if (floorNumber >= 17 && floorNumber <= 20) {
        sortPackage(floorNumber);
        Serial.println("Sorted package to floor " + String(floorNumber));
      } else {
        errorBeep();
        Serial.println("Invalid floor number! Please use 17-20.");
      }
    }
    else if (command.length() == 2 && isDigit(command[0]) && isDigit(command[1])) {
      // Handle direct floor number input (e.g. "17")
      int floorNumber = command.toInt();
      if (floorNumber >= 17 && floorNumber <= 20) {
        sortPackage(floorNumber);
        Serial.println("Sorted package to floor " + String(floorNumber));
      } else {
        errorBeep();
        Serial.println("Invalid floor number! Please use 17-20.");
      }
    }
    else {
      Serial.println("Unknown command. Type 'help' for available commands.");
    }
  }
}

void printHelp() {
  Serial.println("\n===== Package Sorter Commands =====");
  Serial.println("17-20    : Sort package to specific floor");
  Serial.println("sort 17  : Alternative way to sort package");
  Serial.println("ip       : Display device IP address");
  Serial.println("wifi     : Display connected WiFi name");
  Serial.println("help     : Show this help message");
  Serial.println("=================================\n");
}

// Auto Connect to Known WiFi
void connectToWiFi() {
  Serial.println("Scanning for WiFi networks...");
  int numNetworks = WiFi.scanNetworks();

  for (int i = 0; i < numNetworks; i++) {
    String detectedSSID = WiFi.SSID(i);
    Serial.println("Found: " + detectedSSID);

    for (int j = 0; j < NUM_WIFI; j++) {
      if (detectedSSID == wifiList[j].ssid) {
        Serial.println("Connecting to " + detectedSSID);
        WiFi.begin(wifiList[j].ssid, wifiList[j].password);

        int attempts = 0;
        while (WiFi.status() != WL_CONNECTED && attempts < 20) {
          delay(500);
          Serial.print(".");
          attempts++;
        }

        if (WiFi.status() == WL_CONNECTED) {
          Serial.println("\nConnected! IP: " + WiFi.localIP().toString());
          return;
        } else {
          Serial.println("\nFailed to connect, trying next...");
        }
      }
    }
  }

  Serial.println("No known WiFi found. Retrying in 10 seconds...");
  delay(10000);
  connectToWiFi(); // Retry connection
}

// Status request handler
void handleStatusRequest() {
  StaticJsonDocument<200> jsonResponse;
  
  // Check WiFi connection
  if (WiFi.status() == WL_CONNECTED) {
    jsonResponse["status"] = "Active";
    jsonResponse["ip"] = WiFi.localIP().toString();
    jsonResponse["ssid"] = WiFi.SSID();
    
    // For pan servo, check if it's attached first
    if (panServo.attached()) {
      jsonResponse["pan_angle"] = panServo.read();
    } else {
      jsonResponse["pan_angle"] = "detached";
    }
    
    // Check tilt servo position
    jsonResponse["tilt_angle"] = tiltServo.read();
  } else {
    jsonResponse["status"] = "Inactive";
  }
  
  String response;
  serializeJson(jsonResponse, response);
  server.send(200, "application/json", response);
}

// Handle HTTP request from Flask
void handleSortRequest() {
  if (server.hasArg("plain")) {
    StaticJsonDocument<200> doc;
    deserializeJson(doc, server.arg("plain"));
    int floorNumber = doc["floor"];

    if (floorNumber >= 17 && floorNumber <= 20) {
      sortPackage(floorNumber);
      server.send(200, "application/json", "{\"status\": \"success\"}");
    } else {
      errorBeep();
      server.send(400, "application/json", "{\"error\": \"Invalid floor\"}");
    }
  } else {
    server.send(400, "application/json", "{\"error\": \"No data received\"}");
  }
}

void sortPackage(int floorNumber) {
  int index = floorNumber - 17;
  Serial.println("Sorting package for floor " + String(floorNumber) + " (" + floorPositions[index].direction + ")");

  // Attach pan servo before movement
  panServo.attach(PAN_PIN, 500, 2500);
  delay(100); // Brief delay to ensure servo is properly attached
  
  // Move pan servo to position
  panServo.write(floorPositions[index].panAngle);
  delay(1000); // Wait for pan servo to reach position
  
  // Detach pan servo to stop power and reduce shaking
  panServo.detach();
  
  // Move tilt servo (which stays attached)
  tiltServo.write(floorPositions[index].tiltAngle);
  delay(2000); // Wait for tilt servo to complete its movement

  resetPosition();
  tone(BUZZER_PIN, 1000, 1000);
}

void resetPosition() {
  // Implement a slower movement back to center position
  // Get current position
  int currentTiltPosition = tiltServo.read();
  int targetPosition = INITIAL_TILT;
  
  // Calculate movement direction
  int step = (currentTiltPosition < targetPosition) ? 1 : -1;
  
  // Move servo slowly to the target position
  Serial.println("Resetting position slowly...");
  for (int pos = currentTiltPosition; pos != targetPosition; pos += step) {
    tiltServo.write(pos);
    delay(5); // Small delay between increments - adjusted for slightly faster movement
  }
  
  // Ensure final position is set precisely
  tiltServo.write(INITIAL_TILT);
  delay(500);
  // Pan servo stays detached and at its last position
}

void errorBeep() {
  Serial.println("Error: Invalid floor!");
  for (int i = 0; i < 3; i++) {
    tone(BUZZER_PIN, 3000, 100); // Shorter error beeps
    delay(200);
  }
}

void readyBeep() {
  for (int i = 0; i < 3; i++) {
    tone(BUZZER_PIN, 3000, 500); // Beep for 500ms
    delay(600);                  // Wait 600ms
  }
}
