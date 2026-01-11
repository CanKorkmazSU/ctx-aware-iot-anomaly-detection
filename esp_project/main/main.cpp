#include "model.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_err.h"
}

// We declare these symbols explicitly so we don't depend on the driver header.
// They are provided by ESP-IDF when USB Serial/JTAG console is enabled.
extern "C" int usb_serial_jtag_read_bytes(uint8_t *buf, uint32_t length, TickType_t ticks_to_wait);
extern "C" int usb_serial_jtag_write_bytes(const uint8_t *buf, uint32_t length, TickType_t ticks_to_wait);

// 1. Instantiate Wrapper
static NanoFL agent;

static constexpr uint8_t kMagic[4] = {'N','F','L','0'};
static constexpr uint16_t kPayloadBytes = 25 * sizeof(float); // 5x5 float32

static bool read_exact(uint8_t *dst, size_t n, TickType_t timeout_ticks) {
    size_t got = 0;
    while (got < n) {
        int r = usb_serial_jtag_read_bytes(dst + got, static_cast<uint32_t>(n - got), timeout_ticks);
        if (r < 0) return false;
        if (r == 0) continue;
        got += static_cast<size_t>(r);
    }
    return true;
}

static void write_text(const char *s) {
    if (!s) return;
    usb_serial_jtag_write_bytes(reinterpret_cast<const uint8_t *>(s), static_cast<uint32_t>(std::strlen(s)), 0);
}

static uint16_t read_le_u16(const uint8_t b[2]) {
    return static_cast<uint16_t>(static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8));
}

static void process_one_sample(const float sample[25]) {
    std::vector<float> v(25);
    for (int i = 0; i < 25; i++) v[static_cast<size_t>(i)] = sample[static_cast<size_t>(i)];

    float score = agent.predict(v);
    // Simple online learning: only train on "normal" frames.
    if (score < 0.15f) {
        agent.train(v);
    }

    char line[96];
    std::snprintf(line, sizeof(line), "OK score=%.6f\n", static_cast<double>(score));
    write_text(line);
}

extern "C" void app_main(void) {
    write_text("NanoFL ready. Send frames: magic 'NFL0' + u16_le payload_len + payload (25 float32).\n");

    size_t matched = 0;
    uint8_t b = 0;

    while (true) {
        // Re-sync on the magic header.
        int r = usb_serial_jtag_read_bytes(&b, 1, pdMS_TO_TICKS(100));
        if (r <= 0) {
            continue;
        }

        if (b == kMagic[matched]) {
            matched++;
            if (matched < sizeof(kMagic)) continue;

            // Full magic matched.
            matched = 0;

            uint8_t len_buf[2];
            if (!read_exact(len_buf, sizeof(len_buf), pdMS_TO_TICKS(200))) {
                continue;
            }
            uint16_t payload_len = read_le_u16(len_buf);

            if (payload_len != kPayloadBytes) {
                // Skip unknown payloads (best-effort) and continue searching.
                // This keeps the protocol extensible.
                static uint8_t trash[64];
                uint16_t remaining = payload_len;
                while (remaining > 0) {
                    uint16_t chunk = remaining > sizeof(trash) ? sizeof(trash) : remaining;
                    if (!read_exact(trash, chunk, pdMS_TO_TICKS(200))) break;
                    remaining = static_cast<uint16_t>(remaining - chunk);
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
            static_assert(sizeof(sample) == kPayloadBytes);
            std::memcpy(sample, payload.data(), payload.size());
            process_one_sample(sample);
        } else {
            matched = (b == kMagic[0]) ? 1 : 0;
        }
    }
}