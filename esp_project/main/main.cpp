#include "model.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
#include "driver/usb_serial_jtag.h"
#include "esp_err.h"
#include "freertos/task.h"
}

// 1. Instantiate Wrapper
static NanoFL agent;

static constexpr uint8_t kMagic[4] = {'N', 'F', 'L', '0'}; // Standard Frame
static constexpr uint8_t kMagicSetWeights[4] = {'N', 'F', 'L',
                                                'W'}; // Host -> Dev: Weights
static constexpr uint8_t kMagicGetWeights[4] = {'N', 'F', 'L',
                                                'R'}; // Host -> Dev: Request

static constexpr uint16_t kPayloadBytes = 25 * sizeof(float); // 5x5 float32

static bool read_exact(uint8_t *dst, size_t n, TickType_t timeout_ticks) {
  const TickType_t start = xTaskGetTickCount();
  size_t got = 0;
  while (got < n) {
    const TickType_t elapsed = xTaskGetTickCount() - start;
    if (elapsed >= timeout_ticks) {
      return false;
    }

    const TickType_t remaining = timeout_ticks - elapsed;
    int r = usb_serial_jtag_read_bytes(
        dst + got, static_cast<uint32_t>(n - got), remaining);
    if (r < 0)
      return false;
    if (r == 0)
      continue;
    got += static_cast<size_t>(r);
  }
  return true;
}

static void write_text(const char *s) {
  if (!s)
    return;
  usb_serial_jtag_write_bytes(reinterpret_cast<const uint8_t *>(s),
                              static_cast<uint32_t>(std::strlen(s)), 0);
}

static uint16_t read_le_u16(const uint8_t b[2]) {
  return static_cast<uint16_t>(static_cast<uint16_t>(b[0]) |
                               (static_cast<uint16_t>(b[1]) << 8));
}

static void process_one_sample(const float sample[25]) {
  std::vector<float> v(25);
  for (int i = 0; i < 25; i++)
    v[static_cast<size_t>(i)] = sample[static_cast<size_t>(i)];

  float score = agent.predict(v);
  // Simple online learning: only train on "normal" frames.
  if (score < 0.15f) {
    agent.train(v);
  }

  // Reduce USB traffic: only print periodically
  static int sample_count = 0;
  sample_count++;
  if (sample_count % 20 == 0) {
    char line[96];
    std::snprintf(line, sizeof(line), "OK score=%.6f (N=%d)\n",
                  static_cast<double>(score), sample_count);
    write_text(line);
  }
}

// Helper to ensure all bytes are written
static void write_all_bytes(const uint8_t *data, size_t len,
                            TickType_t timeout_ticks) {
  size_t written = 0;
  TickType_t start = xTaskGetTickCount();

  while (written < len) {
    if ((xTaskGetTickCount() - start) > timeout_ticks)
      break;

    int r = usb_serial_jtag_write_bytes(data + written, len - written,
                                        pdMS_TO_TICKS(100));
    if (r > 0) {
      written += r;
    } else {
      vTaskDelay(1); // Yield if stalled
    }
  }
}

extern "C" void app_main(void) {
  usb_serial_jtag_driver_config_t cfg = {
      .tx_buffer_size = 4096,
      .rx_buffer_size = 4096,
  };
  ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));

  write_text("NanoFL ready. Send frames: magic 'NFL0' + u16_le payload_len + "
             "payload (25 float32).\n");

  size_t matched = 0; // 0=Waiting N, 1=Waiting F, 2=Waiting L, 3=Waiting Type
  uint8_t b = 0;

  while (true) {
    // Blocking read with timeout (implicitly yields to OS)
    int r = usb_serial_jtag_read_bytes(&b, 1, pdMS_TO_TICKS(100));
    if (r <= 0)
      continue;

    if (matched < 3) {
      // Expecting N, F, L
      if (b == kMagic[matched]) {
        matched++;
      } else {
        matched = (b == kMagic[0]) ? 1 : 0;
      }
    } else {
      // matched == 3, expecting type: '0' (Data), 'W' (Set), 'R' (Get)
      matched = 0; // Reset state for next packet

      if (b == '0') {
        // --- DATA FRAME ---
        uint8_t len_buf[2];
        if (!read_exact(len_buf, sizeof(len_buf), pdMS_TO_TICKS(200)))
          continue;
        uint16_t payload_len = read_le_u16(len_buf);

        if (payload_len != kPayloadBytes) {
          // Drain unknown payload
          static uint8_t trash[64];
          uint16_t remaining = payload_len;
          while (remaining > 0) {
            uint16_t chunk =
                remaining > sizeof(trash) ? sizeof(trash) : remaining;
            if (!read_exact(trash, chunk, pdMS_TO_TICKS(50)))
              break;
            remaining -= chunk;
          }
          write_text("ERR bad_len\n");
          continue;
        }

        std::array<uint8_t, kPayloadBytes> payload{};
        if (!read_exact(payload.data(), payload.size(), pdMS_TO_TICKS(500))) {
          write_text("ERR timeout\n");
          continue;
        }
        float sample[25];
        std::memcpy(sample, payload.data(), payload.size());
        process_one_sample(sample);

      } else if (b == 'W') {
        // --- SET WEIGHTS ---
        uint8_t len_buf[2];
        if (!read_exact(len_buf, sizeof(len_buf), pdMS_TO_TICKS(200)))
          continue;
        uint16_t w_bytes = read_le_u16(len_buf);

        std::vector<uint8_t> w_buf(w_bytes);
        if (!read_exact(w_buf.data(), w_bytes, pdMS_TO_TICKS(5000))) {
          write_text("ERR timeout_w\n");
          continue;
        }

        size_t num_floats = w_bytes / sizeof(float);
        std::vector<float> weights(num_floats);
        std::memcpy(weights.data(), w_buf.data(), w_bytes);
        agent.set_weights(weights);
        write_text("OK SET_WEIGHTS\n");

      } else if (b == 'R') {
        // --- GET WEIGHTS ---
        std::vector<float> w = agent.export_weights();
        uint16_t w_len = static_cast<uint16_t>(w.size() * sizeof(float));

        // Header
        write_all_bytes(kMagicSetWeights, 4, pdMS_TO_TICKS(500));
        write_all_bytes(reinterpret_cast<uint8_t *>(&w_len), 2,
                        pdMS_TO_TICKS(500));
        write_all_bytes(reinterpret_cast<uint8_t *>(w.data()), w_len,
                        pdMS_TO_TICKS(5000));

      } else if (b == 'N') {
        // Retry match?
        matched = 1;
      }
    }
  }
}