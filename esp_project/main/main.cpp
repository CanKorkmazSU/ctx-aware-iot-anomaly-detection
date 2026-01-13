#include "model.h"
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
#include "driver/usb_serial_jtag.h"
#include "esp_err.h"
#include "freertos/task.h"
}

// Global Agent
static NanoFL agent;
static std::vector<std::vector<float>> reservoir;
static constexpr size_t K_RES_SIZE = 20;

// Protocol: N F L 0 (Data), N F L W (Weights), N F L R (Request)
// Payload: 100 floats (400 bytes)
static constexpr int PAYLOAD_BYTES = 100 * sizeof(float);

// --- Simple Serial Helpers ---
static void write_data(const void *data, size_t key_len) {
  usb_serial_jtag_write_bytes((const uint8_t *)data, key_len,
                              pdMS_TO_TICKS(100));
}

static void write_str(const char *s) {
  if (s)
    write_data(s, strlen(s));
}

static bool read_bytes(void *dest, size_t n, int timeout_ms) {
  size_t got = 0;
  TickType_t end = xTaskGetTickCount() + pdMS_TO_TICKS(timeout_ms);
  while (got < n) {
    if (xTaskGetTickCount() > end)
      return false;
    int r = usb_serial_jtag_read_bytes((uint8_t *)dest + got, n - got, 1);
    if (r > 0)
      got += r;
    else
      vTaskDelay(1);
  }
  return true;
}

// --- Frame Processing ---
static void process_frame(const std::vector<float> &data) {
  float score = agent.predict(data);

  // Online Training (Only on "Normal" data)
  if (score < 0.15f) {
    agent.train(data);
    if (reservoir.size() >= K_RES_SIZE)
      reservoir.erase(reservoir.begin());
    reservoir.push_back(data);
  }

  // Periodic Log
  static int count = 0;
  if (++count % 20 == 0) {
    char buf[64];
    sprintf(buf, "OK score=%.4f n=%d\n", score, count);
    write_str(buf);
  }
}

extern "C" void app_main(void) {
  usb_serial_jtag_driver_config_t cfg = {4096, 4096};
  usb_serial_jtag_driver_install(&cfg);
  write_str("NanoFL Ready (100-float).\n");

  uint8_t magic[4];

  while (true) {
    // 1. Wait for Magic 'N' 'F' 'L'
    if (!read_bytes(magic, 1, 100))
      continue;
    if (magic[0] != 'N')
      continue;
    if (!read_bytes(magic + 1, 2, 100))
      continue;
    if (magic[1] != 'F' || magic[2] != 'L')
      continue;

    // 2. Read Type
    if (!read_bytes(magic + 3, 1, 100))
      continue;
    char type = magic[3];

    if (type == '0') { // DATA
      uint16_t len;
      if (!read_bytes(&len, 2, 100) || len != PAYLOAD_BYTES) {
        write_str("ERR_LEN\n");
        continue;
      }
      std::vector<float> buf(100);
      if (read_bytes(buf.data(), PAYLOAD_BYTES, 500)) {
        process_frame(buf);
      }

    } else if (type == 'W') { // SET WEIGHTS (FedAvg / FedALA)
      uint16_t bytes;
      if (!read_bytes(&bytes, 2, 100))
        continue;

      std::vector<float> w(bytes / 4);
      if (read_bytes(w.data(), bytes, 3000)) {
        // Adaptive Aggregation (FedALA) if we have validation data
        if (reservoir.size() > 5) {
          agent.adaptive_aggregate(w, reservoir);
          write_str("OK_ALA\n");
        } else {
          agent.set_weights(w);
          write_str("OK_Direct\n");
        }
      }

    } else if (type == 'R') { // GET WEIGHTS
      std::vector<float> w = agent.get_weights();
      uint16_t len = w.size() * 4;
      write_data("NFLW", 4);
      write_data(&len, 2);
      write_data(w.data(), len);
    }
  }
}
