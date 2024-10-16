#include <DallasTemperature.h>
#include "DHT.h"
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <PubSubClient.h>

// Inisialisasi konstanta
#define ONE_WIRE_BUS 2
#define S0 16
#define S1 5
#define S2 4
#define S3 0
#define SIG A0
#define DHT_PIN 13
#define DHT_TYPE DHT22
#define NUTRIENT_DETECTOR_PIN 14
#define FERTILIZER_DETECTOR_PIN 12
#define RELAY_NUTRIENT_PIN 3
#define RELAY_FERTILIZER_PIN 1
#define SSID "skypianAcces"
#define PASSWORD "Monitor123"
// #define MQTTHOST "broker.hivemq.com"
#define MQTTHOST "192.168.0.188"
#define MQTTPORT  1883

// Inisialisasi variabel
float waterTemperature;
float tds;
float pH;
float humidity;
float airTemperature;
int nutrientDetector;
int fertilizerDetector;
int nutrientPumpStatus;
int fertilizerPumpStatus;
String systemController = "0|0";
String dataSensor;

OneWire oneWire(ONE_WIRE_BUS);

DallasTemperature sensors(&oneWire);

DHT dht(DHT_PIN, DHT_TYPE);
// Inisialisasi library wifi WiFiClient
WiFiClient espClient;
// Inisialisasi library MQTT PubSubClient
PubSubClient client(espClient);

void setup() {
  Serial.begin(115200);
  pinMode(S0,OUTPUT);
  pinMode(S1,OUTPUT);
  pinMode(S2,OUTPUT);
  pinMode(S3,OUTPUT);
  pinMode(SIG, INPUT);
  pinMode(RELAY_NUTRIENT_PIN, OUTPUT);
  pinMode(RELAY_FERTILIZER_PIN, OUTPUT);

  sensors.begin();

  dht.begin();

  setupWifi();
  // Mengatur host dan port MQTT
  client.setServer(MQTTHOST, MQTTPORT);
  // Mengatur fungsi callback mana yang akan digunakan
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  delay(200);
  // Mengendalikan pompa cairan dan pompa pupuk berdasarkan data dari MQTT
  systemManager(systemController);

  waterTemperature = getWaterTemperature();

  inputMultiplexer(0);
  tds = analogRead(SIG);
  delay(100);

  inputMultiplexer(1);
  pH = getpHValue(SIG);
  delay(200);

  humidity = dht.readHumidity();
  airTemperature = dht.readTemperature();
  nutrientDetector = liquidDetection(NUTRIENT_DETECTOR_PIN);
  fertilizerDetector = liquidDetection(FERTILIZER_DETECTOR_PIN);
  nutrientPumpStatus = checkRelayStatus(RELAY_NUTRIENT_PIN);
  fertilizerPumpStatus = checkRelayStatus(RELAY_FERTILIZER_PIN);

  dataSensor = String(waterTemperature)+"|"+String(tds)+"|"+String(pH)+"|"+String(humidity)+"|"+String(airTemperature)+"|"+String(nutrientDetector)+"|"+String(fertilizerDetector)+"|"+String(nutrientPumpStatus)+"|"+String(fertilizerPumpStatus);
  if(dataSensor) {
    // Serial out sensorData in Serial Monitor
    Serial.println(dataSensor);
    // Publish data ke MQTT
    client.publish("AeroponicsSensor", String(dataSensor).c_str(), true);
  }
  delay(500);
}

float getWaterTemperature() {
  sensors.requestTemperatures();
  float tempC = sensors.getTempCByIndex(0);
  return tempC;
}

void inputMultiplexer(int channel){
  if(channel == 0){
    digitalWrite(S0,LOW); digitalWrite(S1,LOW); digitalWrite(S2,LOW); digitalWrite(S3,LOW);
  }
  else if(channel == 1){
    digitalWrite(S0,HIGH); digitalWrite(S1,LOW); digitalWrite(S2,LOW); digitalWrite(S3,LOW);
  }
  else{
    digitalWrite(S0,LOW); digitalWrite(S1,LOW); digitalWrite(S2,LOW); digitalWrite(S3,LOW);
  }
}

float getpHValue(int pin) {
  float sensorValue = analogRead(pin);
  float phValue = (-0.02642 * sensorValue) + 16.73;
  if (phValue > 6){
    phValue = phValue - 1;
  }

  return sensorValue;
}

int liquidDetection(int pin)
{
  if(digitalRead(pin)){
    return 1;
  }else{
    return 0;
  }
}

int checkRelayStatus(int pin)
{
  int relayStatus = digitalRead(pin);
  if(relayStatus == HIGH){
    return 0;
  }else{
    return 1;
  }
}

// Fungsi setup konfigurasi wifi
void setupWifi(){
  delay(10);
  Serial.println();
  Serial.print("connect to");
  Serial.println(SSID);
  // Set mode wifi => WIFI_STA berarti wifi akan melakukan koneksi otomatis dengan jaringan sekitar
  // WiFi.mode(WIFI_STA);
  WiFi.mode(WIFI_AP);
  // Mulai koneksi ke wifi dengan nama wifi dan password yang telah diatur sebelumnya
  WiFi.begin(SSID, PASSWORD);


  // Cek jika wifi tidak terkoneksi
  while(WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print(".");
    digitalWrite(RELAY_NUTRIENT_PIN, LOW);
    digitalWrite(RELAY_FERTILIZER_PIN, HIGH);
  }
  // randomSeed(micros());
  // Tampilkan jika wifi telah terhubung
  Serial.println("");
  Serial.println("WiFi Connected");
  Serial.println("IP Address : ");
  Serial.println(WiFi.localIP());
}


// callback untuk menerima data dari MQTT
void callback(char* topic, byte* payload, unsigned int length) {
  payload[length] = '\0';
  // Cek apakah data yang diterima dari topic "systemController"
  if(strcmp(topic, "systemController") == 0){
    systemController = String((char*) payload);
  }
}


// Fungsi untuk melakukan rekoneksi secara otomatis jika koneksi ke MQTT terputus
void reconnect() {
  // Loop until we're reconnected
  // Cek apakah MQTT tidak terkonek
  while (!client.connected()) {
    digitalWrite(RELAY_NUTRIENT_PIN, LOW);
    digitalWrite(RELAY_FERTILIZER_PIN, HIGH);
    // Create a random client ID
    String clientId = "ESP8266Client-";
    clientId += String(random(0xffff), HEX);
    // Attempt to connect
    if (client.connect(clientId.c_str())) {
      subscribeMQTT();
    } else {
      // Wait 5 seconds before retrying
      digitalWrite(RELAY_NUTRIENT_PIN, LOW);
      digitalWrite(RELAY_FERTILIZER_PIN, HIGH);
      delay(2000);
    }
  }
}

void subscribeMQTT()
{
    // 1 is a QoS type, Qos 1
    client.subscribe("systemController", 1);
}

void systemManager(String system)
{
 
  String parts[2]; // Array untuk menyimpan potongan string
  int partIndex = 0; // Indeks saat ini di dalam array
  const char separator = '|';


  // Iterasi melalui setiap karakter dalam string
  for (int i = 0; i < system.length(); i++) {
    char currentChar = system.charAt(i);
    if (currentChar == separator) {
      partIndex++; // Pindah ke potongan string berikutnya
    } else {
      parts[partIndex] += currentChar; // Tambahkan karakter ke potongan saat ini
    }
  }

  digitalWrite(RELAY_NUTRIENT_PIN, parts[0].toInt());
  digitalWrite(RELAY_FERTILIZER_PIN, parts[1].toInt());
}