// ============================================================
// FIRMWARE FINAL ESP32 + ADXL355 


#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <math.h>

// Para keepalive con lwIP
#include <lwip/sockets.h>
#include <lwip/netdb.h>

// =============== CONFIG ===============
#define CAPTURE_MINUTES   2
#define TARGET_ODR_HZ     250.0

// Escalonado de envio: sensor N espera (N-1)*SEND_STAGGER_MS.
// Con envio de ~10 s por sensor, 12000 ms evita solapamiento total.
// 
#define SEND_STAGGER_MS   12000

#define I2C_SDA 8
#define I2C_SCL 9
#define DRDY_PIN 7

#define ADXL355_ADDR  0x1D
#define REG_FILTER    0x28
#define REG_RANGE     0x2C
#define REG_POWER_CTL 0x2D
#define REG_XDATA3    0x08

// ===== R SENSOR: 1 al 8 =====
#define SENSOR_ID 8

const char* STA_SSID  = "1";
const char* STA_PASS  = "12345678";
const char* SERVER_IP = " 10.191.217.99";
const uint16_t TCP_PORT = 5000;

// =============== DRDY ISR ===============
volatile bool dataReady = false;
volatile uint8_t first_drdy_flag = 0;
void IRAM_ATTR handleDataReady() {
  dataReady = true;
  if (!first_drdy_flag) first_drdy_flag = 1;
}

// =============== I2C helpers ===============
static inline void writeReg(uint8_t r, uint8_t v) {
  Wire.beginTransmission(ADXL355_ADDR);
  Wire.write(r); Wire.write(v);
  Wire.endTransmission();
}

static inline int32_t convertRaw(uint8_t h, uint8_t m, uint8_t l) {
  int32_t raw = ((int32_t)h << 12) | ((int32_t)m << 4) | (l >> 4);
  if (raw & 0x80000) raw |= 0xFFF00000;
  return raw;
}

static inline bool readXYZ_safe(int32_t* x, int32_t* y, int32_t* z) {
  Wire.beginTransmission(ADXL355_ADDR);
  Wire.write(REG_XDATA3);
  if (Wire.endTransmission(false) != 0) return false;
  uint8_t got = Wire.requestFrom(ADXL355_ADDR, (uint8_t)9);
  if (got != 9) return false;
  uint8_t d[9];
  for (int i = 0; i < 9; i++) d[i] = Wire.read();
  *x = convertRaw(d[0], d[1], d[2]);
  *y = convertRaw(d[3], d[4], d[5]);
  *z = convertRaw(d[6], d[7], d[8]);
  return true;
}

// =============== Buffers dinamicos ===============
int16_t *ax = nullptr, *ay = nullptr, *az = nullptr;
uint32_t BUF_CAP = 0;

bool allocate_buffers_fit(uint32_t wantSamples) {
  const int STEP = 1000;
  for (int32_t n = (int32_t)wantSamples; n > 0; n -= STEP) {
    if (ax) { heap_caps_free(ax); ax = nullptr; }
    if (ay) { heap_caps_free(ay); ay = nullptr; }
    if (az) { heap_caps_free(az); az = nullptr; }
    ax = (int16_t*)heap_caps_malloc(n * sizeof(int16_t), MALLOC_CAP_8BIT);
    ay = (int16_t*)heap_caps_malloc(n * sizeof(int16_t), MALLOC_CAP_8BIT);
    az = (int16_t*)heap_caps_malloc(n * sizeof(int16_t), MALLOC_CAP_8BIT);
    if (ax && ay && az) { BUF_CAP = (uint32_t)n; return true; }
  }
  BUF_CAP = 0; return false;
}

// =============== TCP / SYNC ===============
WiFiClient client;
int64_t g_time_offset_us = 0;

bool recvLine(WiFiClient &c, String &out, uint32_t timeout_ms = 1000) {
  uint32_t t0 = millis(); out = "";
  while (millis() - t0 < timeout_ms) {
    while (c.available()) {
      char ch = c.read();
      if (ch == '\n') return true;
      if (ch != '\r') out += ch;
    }
    delay(1);
  }
  return false;
}

bool doHandshakeAsClient(WiFiClient &c, int rounds = 8) {
  int ok = 0;
  for (int i = 0; i < rounds; i++) {
    String line;
    if (!recvLine(c, line, 2000)) { if (!recvLine(c, line, 200)) break; }
    line.trim();
    if (line.startsWith("SYNC_REQ")) {
      int64_t t_us = esp_timer_get_time();
      c.printf("TS %lld\n", (long long)t_us);
      String resp; if (!recvLine(c, resp, 1000)) return false; resp.trim();
      if (resp.startsWith("OFF ")) {
        int64_t off = atoll(resp.substring(4).c_str());
        g_time_offset_us = off; ok++;
      } else if (resp.startsWith("SYNC_DONE")) break;
      else return false;
    } else if (line.startsWith("SYNC_DONE")) break;
  }
  return ok > 0;
}

void ensureWifi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(STA_SSID, STA_PASS);
  Serial.printf("[WiFi] Conectando a '%s'", STA_SSID);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 150) {
    Serial.print(".");
    delay(100);
    tries++;
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED)
    Serial.printf("[WiFi] OK. IP=%s\n", WiFi.localIP().toString().c_str());
  else
    Serial.println("[WiFi] ERROR: no se logro conectar.");
}

void purgeServerInput() {
  if (!client.connected()) return;
  int guard = 0;
  while (client.connected() && client.available() && guard < 32768) {
    (void)client.read();
    guard++;
    delay(0);
  }
}

// Configurar keepalive usando lwIP directamente
void configureTcpKeepAlive() {
  if (!client.connected()) return;

  int sock = client.fd();
  if (sock < 0) return;

  int keepAlive = 1;    // Habilitar keepalive
  int keepIdle = 5;     // Empezar a sondear tras 5 s de inactividad
  int keepInterval = 3; // Intervalo entre sondas: 3 s
  int keepCount = 3;    // Número de sondas antes de desconectar

  lwip_setsockopt(sock, SOL_SOCKET,  SO_KEEPALIVE,  &keepAlive,    sizeof(keepAlive));
  lwip_setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE,  &keepIdle,     sizeof(keepIdle));
  lwip_setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, &keepInterval, sizeof(keepInterval));
  lwip_setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT,   &keepCount,    sizeof(keepCount));

  Serial.println("[TCP] Keepalive configurado.");
}

void ensureTcp() {
  static unsigned long lastTry = 0;
  if (client.connected()) return;
  if (millis() - lastTry < 2000) return;
  lastTry = millis();

  Serial.printf("[TCP] Conectando a %s:%u ...\n", SERVER_IP, TCP_PORT);
  if (!client.connect(SERVER_IP, TCP_PORT)) {
    Serial.println("[TCP] ERROR: no conecta.");
    return;
  }

  client.setNoDelay(true);
  configureTcpKeepAlive();

  if (doHandshakeAsClient(client, 10))
    Serial.printf("[SYNC] OK. offset_us=%lld\n", (long long)g_time_offset_us);
  else
    Serial.println("[SYNC] AVISO: handshake fallo.");

  client.printf("HELLO SENSOR %d READY\n", SENSOR_ID);
  purgeServerInput();
}

// ============================================================
// Reconexion forzada.
// Cierra el socket viejo y abre uno nuevo limpio.
// ============================================================
void forceReconnect() {
  Serial.println("[TCP] Reconexion forzada (socket fresco)...");
  client.stop();
  delay(200);
  ensureWifi();
  ensureTcp();
  uint32_t t0 = millis();
  while (!client.connected() && millis() - t0 < 15000) {
    delay(2100);   // > 2000 ms del rate-limit interno de ensureTcp
    ensureTcp();
  }
}

// =============== Envio robusto ===============
static bool write_all(WiFiClient& c, const uint8_t* buf, size_t len,
                      unsigned long tout_ms = 15000, int max_retries = 2) {
  size_t sent = 0;
  int retries = 0;
  unsigned long t0 = millis();
  while (sent < len) {
    int n = c.write(buf + sent, len - sent);
    if (n > 0) {
      sent += (size_t)n;
      t0 = millis();
      continue;
    }
    if (!c.connected()) return false;
    if (millis() - t0 > tout_ms) {
      if (retries++ >= max_retries) return false;
      delay(100);
      t0 = millis();
    }
    delay(1);
    yield();
  }
  return true;
}

// =============== CONTROL DESDE SERVIDOR ===============
String cmdBuf;
volatile bool want_start = false;

void resetCommandState() {
  want_start = false;
  cmdBuf = "";
  purgeServerInput();
}

void pollServerCommands() {
  while (client.connected() && client.available()) {
    char ch = client.read();
    if (ch == '\n') {
      String cmd = cmdBuf; cmdBuf = ""; cmd.trim();
      String up = cmd; up.toUpperCase();

      if (up.startsWith("INICIAR")) {
        want_start = true;
        Serial.println("[CMD] INICIAR recibido.");
        client.printf("ACK INICIAR %d\n", SENSOR_ID);

      } else if (up.startsWith("REINICIAR")) {
        Serial.println("[CMD] REINICIAR -> ESP.restart()");
        client.printf("ACK REINICIAR %d\n", SENSOR_ID);
        Serial.flush();
        delay(50);
        ESP.restart();

      } else if (up.startsWith("RESYNC")) {
        client.printf("ACK RESYNC %d\n", SENSOR_ID);
        client.printf("RESYNC\n");
        if (doHandshakeAsClient(client, 6))
          Serial.printf("[SYNC] Re-sincronizado. offset_us=%lld\n",
                        (long long)g_time_offset_us);
      }

    } else if (ch != '\r') {
      cmdBuf += ch;
      if (cmdBuf.length() > 200) cmdBuf = "";
    }
  }
}

// =============== CAPTURA ===============
uint32_t run_capture_and_report(uint64_t &t_start_out, uint64_t &t_end_out) {
  const uint64_t capture_us = (uint64_t)CAPTURE_MINUTES * 60ULL * 1000000ULL;

  uint64_t t_start = esp_timer_get_time();
  uint64_t t_deadline = t_start + capture_us;

  bool first_msg = false;
  uint32_t n = 0;
  uint32_t last_heartbeat = 0;

  while ((esp_timer_get_time() < t_deadline) && (n < BUF_CAP)) {
    // Heartbeat cada 30 s durante la captura
    if (millis() - last_heartbeat > 30000) {
      if (client.connected()) {
        client.printf("HEARTBEAT %d\n", n);
      }
      last_heartbeat = millis();
    }

    if (!dataReady) { delayMicroseconds(50); continue; }
    dataReady = false;
    if (!first_msg && first_drdy_flag) {
      Serial.println("# Primer DRDY detectado!");
      first_msg = true;
    }
    int32_t xr, yr, zr;
    if (!readXYZ_safe(&xr, &yr, &zr)) continue;

    const double LSB_PER_G = 256000.0;
    long xi = lround((xr / LSB_PER_G) * 1000.0);
    long yi = lround((yr / LSB_PER_G) * 1000.0);
    long zi = lround((zr / LSB_PER_G) * 1000.0);
    if (xi < -32768) xi = -32768; if (xi > 32767) xi = 32767;
    if (yi < -32768) yi = -32768; if (yi > 32767) yi = 32767;
    if (zi < -32768) zi = -32768; if (zi > 32767) zi = 32767;

    ax[n] = (int16_t)xi;
    ay[n] = (int16_t)yi;
    az[n] = (int16_t)zi;
    n++;
  }

  uint64_t t_end = esp_timer_get_time();
  double total_s = (t_end - t_start) / 1e6;
  double fs_eff  = (n > 1) ? (double)(n - 1) / total_s : 0.0;

  Serial.println("# ===== CAPTURA FINALIZADA =====");
  Serial.printf("# Tiempo real: %.3f s (objetivo: %d min)\n",
                total_s, (int)CAPTURE_MINUTES);
  Serial.printf("# Muestras guardadas: %u (capacidad: %u)\n", n, BUF_CAP);
  Serial.printf("# Frecuencia efectiva: %.3f Hz\n", fs_eff);

  t_start_out = t_start; t_end_out = t_end;
  return n;
}

// =============== ENVIO OPTIMIZADO ===============
void send_buffer_to_server(uint32_t n, uint64_t t_start_us) {

  // ── CORRECCIÓN 1 ────────────────────────────────────────────────────────
  // Solo reconectar si el socket realmente cayó durante la captura.
  // Antes se llamaba forceReconnect() incondicionalmente, lo que mataba
  // la conexión buena y generaba el ciclo "Desconectado → Reconectado → HELLO".
  if (!client.connected()) {
    Serial.println("[SEND] Conexión caída; reconectando...");
    forceReconnect();
  }
  // ────────────────────────────────────────────────────────────────────────

  if (!client.connected()) {
    Serial.println("[SEND] Sin TCP tras reconexion; omito envio.");
    return;
  }

  // Escalonado secuencial: evita que los 8 saturen el servidor a la vez
  uint32_t stagger_ms = (uint32_t)(SENSOR_ID - 1) * SEND_STAGGER_MS;
  if (stagger_ms > 0) {
    Serial.printf("[SEND] Escalonado: espero %u ms (Sensor %d)\n",
                  stagger_ms, SENSOR_ID);

    uint32_t t0s = millis();
    uint32_t last_hb_stagger = millis();   // para heartbeat durante la espera

    while (millis() - t0s < stagger_ms) {
      if (client.connected()) {
        pollServerCommands();

        // ── CORRECCIÓN 2 ──────────────────────────────────────────────────
        // Heartbeat cada 20 s durante el escalonado.
        // El servidor tiene timeout 300 s, pero el silencio de hasta 84 s
        // (sensor 8) podía provocar cortes en algunos SO o routers con NAT.
        if (millis() - last_hb_stagger > 20000) {
          client.printf("HEARTBEAT STAGGER %d\n", SENSOR_ID);
          last_hb_stagger = millis();
        }
        // ──────────────────────────────────────────────────────────────────

      } else {
        // ── CORRECCIÓN 3 ──────────────────────────────────────────────────
        // Si el socket muere durante la espera del escalonado, reconectar
        // de inmediato en vez de seguir esperando en bucle sin conexión.
        Serial.println("[SEND] Conexión caída durante escalonado; reconectando...");
        forceReconnect();
        last_hb_stagger = millis();
        // ──────────────────────────────────────────────────────────────────
      }

      delay(10);
      yield();
    }

    // Verificación final tras el escalonado
    if (!client.connected()) forceReconnect();
  }

  const double Ts_us = 1e6 / TARGET_ODR_HZ;

  // Buffer grande: ~52 llamadas TCP en vez de ~420 para 28000 muestras
  static char out[16384];
  size_t pos = 0;
  uint32_t i = 0;
  const int MAX_RECONNECTS = 5;
  int reconnects = 0;

  Serial.printf("[SEND] Enviando %u muestras...\n", n);
  uint32_t t_send_start = millis();

  while (i < n) {
    if (!client.connected()) {
      if (reconnects >= MAX_RECONNECTS) {
        Serial.println("[SEND] Conexion perdida definitivamente.");
        return;
      }
      forceReconnect();
      if (!client.connected()) {
        reconnects++;
        delay(500);
        continue;
      }
      reconnects = 0;
    }

    pos = 0;
    while (i < n && pos < sizeof(out) - 64) {
      const double G = 9.80665;
      double x = (ax[i] / 1000.0) * G;
      double y = (ay[i] / 1000.0) * G;
      double z = (az[i] / 1000.0) * G;

      int wrote = snprintf(out + pos, sizeof(out) - pos,
                           "%d,%.6f,%.6f,%.6f\n",
                           SENSOR_ID, x, y, z);
      if (wrote <= 0) break;
      pos += (size_t)wrote;
      i++;
    }

    if (pos > 0) {
      if (!write_all(client, (const uint8_t*)out, pos)) {
        Serial.println("[SEND] write_all fallo. Reconectando...");
        client.stop();
        delay(200);
        continue;
      }
    }

    if ((i % 5000) == 0 && i > 0) {
      float elapsed = (millis() - t_send_start) / 1000.0f;
      Serial.printf("[SEND] %u/%u (%.0f muestras/s)\n",
                    i, n, i / elapsed);
    }
    yield();
  }

  float total_s = (millis() - t_send_start) / 1000.0f;
  Serial.printf("[SEND] Completado: %u muestras en %.2f s (%.0f muestras/s)\n",
                n, total_s, n / total_s);
}

// =============== SETUP ===============
void setup() {
  Serial.begin(115200);
  delay(300);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  writeReg(REG_POWER_CTL, 0x01); delay(5);
  writeReg(REG_RANGE,     0x01);
  writeReg(REG_FILTER,    0x04);
  delay(5);
  writeReg(REG_POWER_CTL, 0x00);
  delay(50);

  pinMode(DRDY_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(DRDY_PIN), handleDataReady, RISING);

  const uint32_t wantSamples =
      (uint32_t)((uint32_t)CAPTURE_MINUTES * 60UL * (uint32_t)TARGET_ODR_HZ + 1);
  if (!allocate_buffers_fit(wantSamples)) {
    Serial.println("# ERROR: no se pudo reservar RAM.");
    while (true) delay(1000);
  }
  Serial.printf("# Capacidad: %u muestras | Sensor ID: %d | Escalonado: %d ms\n",
                BUF_CAP, SENSOR_ID, (int)((SENSOR_ID - 1) * SEND_STAGGER_MS));

  ensureWifi();
  ensureTcp();
  Serial.println("# Listo. Esperando INICIAR/REINICIAR/RESYNC.");
}

// =============== LOOP ===============
void loop() {
  // Monitoreo periódico de conexión
  static unsigned long last_conn_check = 0;

  if (millis() - last_conn_check > 5000) {
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[WARN] WiFi desconectado, reconectando...");
      ensureWifi();
    }
    if (!client.connected()) {
      Serial.println("[WARN] TCP desconectado, reconectando...");
      ensureTcp();
    }
    last_conn_check = millis();
  }

  if (WiFi.status() != WL_CONNECTED) ensureWifi();
  ensureTcp();

  if (client.connected()) {
    pollServerCommands();
  }

  if (want_start) {
    want_start = false;
    uint64_t t0, t1;
    uint32_t n = run_capture_and_report(t0, t1);
    send_buffer_to_server(n, t0);
    resetCommandState();
    first_drdy_flag = 0;
    Serial.println("# Listo. Esperando INICIAR/REINICIAR/RESYNC.");
  }

  delay(10);
}