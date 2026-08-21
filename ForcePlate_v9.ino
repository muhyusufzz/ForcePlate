#include <Arduino.h>
#include <SPI.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <stdlib.h>
#include "Protocentral_ADS1220.h"
#include "esp_timer.h"

// ==========================================================
// FORCE PLATE FIRMWARE v8.5 CRC-FRAMED / TURBO-CMJ / FAST-READ / MUX-TIMESTAMPED
// v8.5 mempertahankan akuisisi ADC/MUX dan faktor kalibrasi. Perubahan hanya
// pada transport serial streaming: setiap frame diberi prefix @FP1 dan CRC16-CCITT,
// lalu Serial.write dipastikan mengirim seluruh baris sebelum frame berikutnya.
// Mapping serial:
// L1, L2, L3, L4, R1, R2, R3, R4
// ==========================================================

#define ADS1_CS_PIN 5
#define ADS1_DRDY_PIN 22
#define ADS2_CS_PIN 17
#define ADS2_DRDY_PIN 16
#define ADS3_CS_PIN 4
#define ADS3_DRDY_PIN 21
#define ADS4_CS_PIN 15
#define ADS4_DRDY_PIN 25

Protocentral_ADS1220 ads[4];

const uint8_t CS_PINS[4] = {
  ADS1_CS_PIN, ADS2_CS_PIN, ADS3_CS_PIN, ADS4_CS_PIN
};

const uint8_t DRDY_PINS[4] = {
  ADS1_DRDY_PIN, ADS2_DRDY_PIN, ADS3_DRDY_PIN, ADS4_DRDY_PIN
};

const uint32_t SERIAL_BAUD = 921600;
const uint32_t ADC_TIMEOUT_US = 12000;

const int32_t ADC_WARN_POS = 7969176;
const int32_t ADC_WARN_NEG = -7969176;

// Tare robust.
const uint16_t TARE_WARMUP_CYCLES = 20;
const uint16_t TARE_TARGET_SAMPLES = 320;
const uint16_t TARE_MIN_VALID_SAMPLES = 280;
const uint16_t TARE_MAX_CYCLES = 520;
const int32_t TARE_WARNING_SPAN_COUNTS = 30000;
const int32_t TARE_HARD_FAIL_SPAN_COUNTS = 150000;
const uint16_t TARE_MAX_SATURATION_EVENTS = 2;

// Zero-return verification setelah tare. Firmware mengirim residual ADC counts;
// konversi ke kg tetap dilakukan hanya di processing.py agar calibration source tunggal.
const uint16_t ZERO_CHECK_WARMUP_CYCLES = 8;
const uint16_t ZERO_CHECK_TARGET_FRAMES = 160;
const uint16_t ZERO_CHECK_MAX_CYCLES = 240;

int64_t tare_offsets[8] = {0};
uint32_t frame_id = 0;
bool tare_valid = false;

int32_t tare_samples[8][TARE_TARGET_SAMPLES];
int32_t scratch_samples[TARE_TARGET_SAMPLES];

enum DaqState {
  STATE_IDLE,
  STATE_STREAMING
};

DaqState daq_state = STATE_IDLE;
uint32_t last_ready_announcement_ms = 0;

enum AcquisitionProfile {
  PROFILE_STANDING_600_NORMAL,
  PROFILE_CMJ_2000_TURBO
};

AcquisitionProfile active_profile = PROFILE_STANDING_600_NORMAL;

// IMPORTANT UNTUK FILE .ino:
// Tipe kustom didefinisikan sebelum deklarasi fungsi dan seluruh prototype
// ditulis manual. Ini mencegah Arduino sketch preprocessor membuat prototype
// otomatis yang memakai ADCResult sebelum tipe tersebut dikenali.
struct ADCResult {
  int32_t value;
  bool timeout;
  bool saturated;
};

// -------------------- EXPLICIT FUNCTION PROTOTYPES --------------------
const char* profileName(AcquisitionProfile profile);
uint16_t profileNominalSps(AcquisitionProfile profile);
void configureAdcProfile(AcquisitionProfile profile, bool announce);
int compareInt32(const void* a, const void* b);
bool waitAllDRDY(
  const uint64_t start_time_us[4],
  ADCResult results[4],
  uint64_t ready_time_us[4]
);
void startMuxConversion(
  uint8_t mux_setting,
  uint64_t start_time_us[4]
);
void readMuxGroup(
  uint8_t mux_setting,
  ADCResult results[4],
  uint64_t& timestamp_us
);
void readFullFrame(
  bool reverse_order,
  ADCResult mux1[4],
  uint64_t& t_mux1,
  ADCResult mux2[4],
  uint64_t& t_mux2
);
void mapMuxResults(
  ADCResult mux1[4],
  ADCResult mux2[4],
  ADCResult* mapped[8]
);
uint16_t crc16Ccitt(const uint8_t* data, size_t length);
void writeSerialAll(const uint8_t* data, size_t length);
void performTare();
void performZeroCheck();
void printSystemMetadata();
void handleCommand(char command);
void setup();
void loop();
// ---------------------------------------------------------------------

const char* profileName(AcquisitionProfile profile) {
  return profile == PROFILE_CMJ_2000_TURBO
    ? "CMJ_2000_SPS_TURBO"
    : "STANDING_600_SPS_NORMAL";
}

uint16_t profileNominalSps(AcquisitionProfile profile) {
  return profile == PROFILE_CMJ_2000_TURBO ? 2000 : 600;
}

void configureAdcProfile(AcquisitionProfile profile, bool announce) {
  daq_state = STATE_IDLE;
  tare_valid = false;
  frame_id = 0;

  for (uint8_t i = 0; i < 4; i++) {
    if (profile == PROFILE_CMJ_2000_TURBO) {
      // ProtoCentral exposes DR_1000SPS and MODE_TURBO separately.
      // Pada ADS1220, kombinasi ini menghasilkan nominal 2000 SPS.
      ads[i].set_data_rate(DR_1000SPS);
      ads[i].set_OperationMode(MODE_TURBO);
    } else {
      ads[i].set_data_rate(DR_600SPS);
      ads[i].set_OperationMode(MODE_NORMAL);
    }
    ads[i].set_pga_gain(PGA_GAIN_128);
    ads[i].set_conv_mode_single_shot();
  }

  active_profile = profile;

  if (announce) {
    Serial.printf(
      "#PROFILE_SET,%s,NOMINAL_SPS=%u,PGA=128,CONV=SINGLE_SHOT,OPERATING_MODE=%s,TARE_REQUIRED\n",
      profileName(active_profile),
      static_cast<unsigned>(profileNominalSps(active_profile)),
      active_profile == PROFILE_CMJ_2000_TURBO ? "TURBO" : "NORMAL"
    );
  }
}

int compareInt32(const void* a, const void* b) {
  const int32_t va = *static_cast<const int32_t*>(a);
  const int32_t vb = *static_cast<const int32_t*>(b);
  return (va > vb) - (va < vb);
}

bool waitAllDRDY(
  const uint64_t start_time_us[4],
  ADCResult results[4],
  uint64_t ready_time_us[4]
) {
  bool seen_high[4] = {false, false, false, false};
  bool finished[4] = {false, false, false, false};
  uint8_t finished_count = 0;
  const uint64_t wait_start_us = esp_timer_get_time();

  // Setelah START pada single-shot, DRDY seharusnya HIGH selama konversi dan
  // kembali LOW ketika data baru siap. Menunggu transisi HIGH->LOW mencegah
  // level LOW lama dari konversi sebelumnya dianggap sebagai data baru.
  while (finished_count < 4) {
    const uint64_t now_us = esp_timer_get_time();

    for (uint8_t i = 0; i < 4; i++) {
      if (finished[i]) {
        continue;
      }

      const int pin_state = digitalRead(DRDY_PINS[i]);
      if (!seen_high[i] && pin_state == HIGH) {
        seen_high[i] = true;
      }

      if (seen_high[i] && pin_state == LOW) {
        ready_time_us[i] = now_us;
        finished[i] = true;
        finished_count++;
        continue;
      }

      if (now_us - start_time_us[i] > ADC_TIMEOUT_US) {
        results[i].timeout = true;
        ready_time_us[i] = now_us;
        finished[i] = true;
        finished_count++;
      }
    }

    if (now_us - wait_start_us > ADC_TIMEOUT_US + 5000ULL) {
      for (uint8_t i = 0; i < 4; i++) {
        if (!finished[i]) {
          results[i].timeout = true;
          ready_time_us[i] = now_us;
          finished[i] = true;
          finished_count++;
        }
      }
      break;
    }

    // Busy polling sengaja tanpa delay pada window konversi yang sangat pendek.
    // Ini mengurangi overhead frame tanpa mengubah data rate, gain, atau faktor
    // kalibrasi. Command serial tetap diproses pada batas antar-frame.
  }

  return finished_count == 4;
}

void startMuxConversion(
  uint8_t mux_setting,
  uint64_t start_time_us[4]
) {
  for (uint8_t i = 0; i < 4; i++) {
    ads[i].select_mux_channels(mux_setting);
  }

  // Timestamp tiap ADC diambil di sekitar command START masing-masing.
  for (uint8_t i = 0; i < 4; i++) {
    const uint64_t before_start_us = esp_timer_get_time();
    ads[i].Start_Conv();
    const uint64_t after_start_us = esp_timer_get_time();
    start_time_us[i] = (before_start_us + after_start_us) / 2ULL;
  }
}

void readMuxGroup(
  uint8_t mux_setting,
  ADCResult results[4],
  uint64_t& timestamp_us
) {
  uint64_t start_time_us[4] = {0, 0, 0, 0};
  uint64_t ready_time_us[4] = {0, 0, 0, 0};

  for (uint8_t i = 0; i < 4; i++) {
    results[i] = {0, false, false};
  }

  startMuxConversion(mux_setting, start_time_us);
  waitAllDRDY(start_time_us, results, ready_time_us);

  uint64_t effective_sum_us = 0;
  uint8_t effective_count = 0;

  for (uint8_t i = 0; i < 4; i++) {
    if (results[i].timeout) {
      continue;
    }

    // FAST READ v8.1:
    // Read_Data_Samples() pada library ProtoCentral memasukkan delay 100 us
    // sebelum dan sesudah transfer pada setiap ADC. Read_Data()+DataToInt()
    // memakai API public yang sama dengan transaksi SPI yang lebih singkat.
    // Ini mengurangi overhead antar kelompok MUX tanpa mengubah wiring.
    ads[i].Read_Data();
    results[i].value = ads[i].DataToInt();
    results[i].saturated = (
      results[i].value >= ADC_WARN_POS ||
      results[i].value <= ADC_WARN_NEG
    );

    // ADS1220 mengintegrasikan sinyal selama konversi. Waktu efektif sampel
    // didekati titik tengah antara command START dan DRDY data-ready.
    const uint64_t effective_us =
      (start_time_us[i] + ready_time_us[i]) / 2ULL;
    effective_sum_us += effective_us;
    effective_count++;
  }

  timestamp_us = effective_count > 0
    ? effective_sum_us / static_cast<uint64_t>(effective_count)
    : esp_timer_get_time();
}

void readFullFrame(
  bool reverse_order,
  ADCResult mux1[4],
  uint64_t& t_mux1,
  ADCResult mux2[4],
  uint64_t& t_mux2
) {
  // Standing mempertahankan urutan lama. Pada profil CMJ, urutan dibalik
  // setiap frame agar kelompok AIN01 dan AIN23 bergantian menjadi kelompok
  // pertama. Timestamp tetap disimpan berdasarkan kelompok kanal, bukan
  // berdasarkan urutan baca. Interpolasi di processing.py dapat memakai
  // informasi ini tanpa mengubah format serial.
  if (reverse_order) {
    readMuxGroup(MUX_AIN2_AIN3, mux2, t_mux2);
    readMuxGroup(MUX_AIN0_AIN1, mux1, t_mux1);
  } else {
    readMuxGroup(MUX_AIN0_AIN1, mux1, t_mux1);
    readMuxGroup(MUX_AIN2_AIN3, mux2, t_mux2);
  }
}

void mapMuxResults(
  ADCResult mux1[4],
  ADCResult mux2[4],
  ADCResult* mapped[8]
) {
  mapped[0] = &mux1[0];  // L1
  mapped[1] = &mux1[1];  // L2
  mapped[2] = &mux2[0];  // L3
  mapped[3] = &mux2[1];  // L4
  mapped[4] = &mux1[2];  // R1
  mapped[5] = &mux1[3];  // R2
  mapped[6] = &mux2[2];  // R3
  mapped[7] = &mux2[3];  // R4
}

uint16_t crc16Ccitt(const uint8_t* data, size_t length) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < length; i++) {
    crc ^= static_cast<uint16_t>(data[i]) << 8;
    for (uint8_t bit = 0; bit < 8; bit++) {
      if ((crc & 0x8000U) != 0U) {
        crc = static_cast<uint16_t>((crc << 1) ^ 0x1021U);
      } else {
        crc = static_cast<uint16_t>(crc << 1);
      }
    }
  }
  return crc;
}

void writeSerialAll(const uint8_t* data, size_t length) {
  // HardwareSerial::write() mengembalikan jumlah byte yang diterima driver.
  // Pastikan seluruh satu frame selesai diantrikan sebelum akuisisi frame baru.
  // Ini menghindari baris terpotong bila TX buffer sedang penuh.
  size_t sent = 0;
  while (sent < length) {
    const size_t written = Serial.write(data + sent, length - sent);
    if (written > 0) {
      sent += written;
    } else {
      delay(1);
    }
  }
}

void performTare() {
  daq_state = STATE_IDLE;
  tare_valid = false;
  Serial.println("#TARE_START");

  uint16_t valid_counts[8] = {0};
  uint16_t timeout_counts[8] = {0};
  uint16_t saturation_counts[8] = {0};

  // Buang sampel awal untuk settling MUX/PGA.
  for (uint16_t cycle = 0; cycle < TARE_WARMUP_CYCLES; cycle++) {
    ADCResult mux1[4], mux2[4];
    uint64_t t_mux1 = 0;
    uint64_t t_mux2 = 0;
    const bool reverse_order = (
      active_profile == PROFILE_CMJ_2000_TURBO && (cycle & 1U) != 0U
    );

    readFullFrame(reverse_order, mux1, t_mux1, mux2, t_mux2);
  }

  for (uint16_t cycle = 0; cycle < TARE_MAX_CYCLES; cycle++) {
    bool complete = true;

    for (uint8_t ch = 0; ch < 8; ch++) {
      if (valid_counts[ch] < TARE_TARGET_SAMPLES) {
        complete = false;
        break;
      }
    }

    if (complete) {
      break;
    }

    ADCResult mux1[4], mux2[4];
    uint64_t t_mux1 = 0;
    uint64_t t_mux2 = 0;
    const bool reverse_order = (
      active_profile == PROFILE_CMJ_2000_TURBO && (cycle & 1U) != 0U
    );

    readFullFrame(reverse_order, mux1, t_mux1, mux2, t_mux2);

    ADCResult* mapped[8];
    mapMuxResults(mux1, mux2, mapped);

    for (uint8_t ch = 0; ch < 8; ch++) {
      const ADCResult& result = *mapped[ch];

      if (result.timeout) {
        timeout_counts[ch]++;
        continue;
      }

      if (result.saturated) {
        saturation_counts[ch]++;
        continue;
      }

      if (valid_counts[ch] < TARE_TARGET_SAMPLES) {
        tare_samples[ch][valid_counts[ch]++] = result.value;
      }
    }
  }

  uint32_t insufficient_mask = 0;
  uint32_t unstable_mask = 0;
  uint32_t warning_mask = 0;
  uint32_t timeout_mask = 0;
  uint32_t saturation_mask = 0;
  uint32_t saturation_fail_mask = 0;

  for (uint8_t ch = 0; ch < 8; ch++) {
    if (timeout_counts[ch] > 0) {
      timeout_mask |= (1UL << ch);
    }

    if (saturation_counts[ch] > 0) {
      saturation_mask |= (1UL << ch);
    }

    if (saturation_counts[ch] > TARE_MAX_SATURATION_EVENTS) {
      saturation_fail_mask |= (1UL << ch);
    }

    if (valid_counts[ch] < TARE_MIN_VALID_SAMPLES) {
      insufficient_mask |= (1UL << ch);

      Serial.printf(
        "#TARE_CH,%u,VALID=%u,TIMEOUT=%u,SAT=%u,"
        "STATUS=INSUFFICIENT\n",
        static_cast<unsigned>(ch + 1),
        static_cast<unsigned>(valid_counts[ch]),
        static_cast<unsigned>(timeout_counts[ch]),
        static_cast<unsigned>(saturation_counts[ch])
      );

      continue;
    }

    qsort(
      tare_samples[ch],
      valid_counts[ch],
      sizeof(int32_t),
      compareInt32
    );

    const uint16_t n = valid_counts[ch];
    const uint16_t median_index = n / 2;
    const int32_t median_value = tare_samples[ch][median_index];

    const uint16_t p05_index = static_cast<uint16_t>(
      (static_cast<uint32_t>(n - 1) * 5U) / 100U
    );

    const uint16_t p95_index = static_cast<uint16_t>(
      (static_cast<uint32_t>(n - 1) * 95U) / 100U
    );

    const int64_t robust_span =
      static_cast<int64_t>(tare_samples[ch][p95_index]) -
      static_cast<int64_t>(tare_samples[ch][p05_index]);

    for (uint16_t i = 0; i < n; i++) {
      const int64_t difference =
        static_cast<int64_t>(tare_samples[ch][i]) -
        static_cast<int64_t>(median_value);

      scratch_samples[i] = static_cast<int32_t>(
        difference >= 0 ? difference : -difference
      );
    }

    qsort(
      scratch_samples,
      n,
      sizeof(int32_t),
      compareInt32
    );

    const int32_t mad = scratch_samples[median_index];

    if (robust_span > TARE_WARNING_SPAN_COUNTS) {
      warning_mask |= (1UL << ch);
    }

    if (robust_span > TARE_HARD_FAIL_SPAN_COUNTS) {
      unstable_mask |= (1UL << ch);
    }

    tare_offsets[ch] = static_cast<int64_t>(median_value);

    Serial.printf(
      "#TARE_CH,%u,VALID=%u,TIMEOUT=%u,SAT=%u,"
      "MEDIAN=%ld,MAD=%ld,SPAN90=%lld\n",
      static_cast<unsigned>(ch + 1),
      static_cast<unsigned>(valid_counts[ch]),
      static_cast<unsigned>(timeout_counts[ch]),
      static_cast<unsigned>(saturation_counts[ch]),
      static_cast<long>(median_value),
      static_cast<long>(mad),
      static_cast<long long>(robust_span)
    );
  }

  const bool hard_failure =
    insufficient_mask != 0 ||
    unstable_mask != 0 ||
    saturation_fail_mask != 0;

  if (hard_failure) {
    Serial.printf(
      "#TARE_FAILED,INSUFFICIENT=0x%02lX,"
      "UNSTABLE=0x%02lX,TIMEOUT=0x%02lX,"
      "SAT_FAIL=0x%02lX\n",
      static_cast<unsigned long>(insufficient_mask),
      static_cast<unsigned long>(unstable_mask),
      static_cast<unsigned long>(timeout_mask),
      static_cast<unsigned long>(saturation_fail_mask)
    );
    return;
  }

  if (
    warning_mask != 0 ||
    timeout_mask != 0 ||
    saturation_mask != 0
  ) {
    Serial.printf(
      "#TARE_WARNING,NOISE=0x%02lX,"
      "TIMEOUT=0x%02lX,SAT=0x%02lX\n",
      static_cast<unsigned long>(warning_mask),
      static_cast<unsigned long>(timeout_mask),
      static_cast<unsigned long>(saturation_mask)
    );
  }

  tare_valid = true;
  Serial.println("#TARE_OK");
}


void performZeroCheck() {
  if (daq_state == STATE_STREAMING) {
    Serial.println("#ZERO_DENIED,STREAMING");
    return;
  }

  if (!tare_valid) {
    Serial.println("#ZERO_DENIED,NOT_TARED");
    return;
  }

  Serial.println("#ZERO_START");

  // Warm-up singkat agar pergantian MUX/PGA sudah settle.
  for (uint16_t cycle = 0; cycle < ZERO_CHECK_WARMUP_CYCLES; cycle++) {
    ADCResult mux1[4], mux2[4];
    uint64_t t_mux1 = 0;
    uint64_t t_mux2 = 0;
    const bool reverse_order = (
      active_profile == PROFILE_CMJ_2000_TURBO && (cycle & 1U) != 0U
    );
    readFullFrame(reverse_order, mux1, t_mux1, mux2, t_mux2);
  }

  int64_t sum_counts[8] = {0};
  uint16_t valid_frames = 0;
  uint32_t timeout_mask = 0;
  uint32_t saturation_mask = 0;

  for (
    uint16_t cycle = 0;
    cycle < ZERO_CHECK_MAX_CYCLES &&
    valid_frames < ZERO_CHECK_TARGET_FRAMES;
    cycle++
  ) {
    ADCResult mux1[4], mux2[4];
    uint64_t t_mux1 = 0;
    uint64_t t_mux2 = 0;
    const bool reverse_order = (
      active_profile == PROFILE_CMJ_2000_TURBO && (cycle & 1U) != 0U
    );

    readFullFrame(reverse_order, mux1, t_mux1, mux2, t_mux2);

    ADCResult* mapped[8];
    mapMuxResults(mux1, mux2, mapped);

    bool frame_valid = true;
    int64_t corrected[8] = {0};

    for (uint8_t ch = 0; ch < 8; ch++) {
      const ADCResult& result = *mapped[ch];

      if (result.timeout) {
        timeout_mask |= (1UL << ch);
        frame_valid = false;
        continue;
      }

      if (result.saturated) {
        saturation_mask |= (1UL << ch);
        frame_valid = false;
        continue;
      }

      corrected[ch] =
        static_cast<int64_t>(result.value) -
        tare_offsets[ch];
    }

    if (!frame_valid) {
      continue;
    }

    for (uint8_t ch = 0; ch < 8; ch++) {
      sum_counts[ch] += corrected[ch];
    }
    valid_frames++;
  }

  if (valid_frames < 40) {
    Serial.printf(
      "#ZERO_FAILED,VALID=%u,TIMEOUT_MASK=0x%02lX,SAT_MASK=0x%02lX\n",
      static_cast<unsigned>(valid_frames),
      static_cast<unsigned long>(timeout_mask),
      static_cast<unsigned long>(saturation_mask)
    );
    return;
  }

  int64_t mean_counts[8] = {0};
  for (uint8_t ch = 0; ch < 8; ch++) {
    mean_counts[ch] =
      sum_counts[ch] /
      static_cast<int64_t>(valid_frames);
  }

  Serial.printf(
    "#ZERO_RESULT,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,"
    "VALID=%u,TIMEOUT_MASK=0x%02lX,SAT_MASK=0x%02lX\n",
    static_cast<long long>(mean_counts[0]),
    static_cast<long long>(mean_counts[1]),
    static_cast<long long>(mean_counts[2]),
    static_cast<long long>(mean_counts[3]),
    static_cast<long long>(mean_counts[4]),
    static_cast<long long>(mean_counts[5]),
    static_cast<long long>(mean_counts[6]),
    static_cast<long long>(mean_counts[7]),
    static_cast<unsigned>(valid_frames),
    static_cast<unsigned long>(timeout_mask),
    static_cast<unsigned long>(saturation_mask)
  );
}

void printSystemMetadata() {
  Serial.println("#FW_VERSION,8.5_CRC_FRAMED_TURBO_CMJ_FAST_READ_MUX_TIMESTAMPED");
  Serial.printf(
    "#ADC_CONFIG,%s,NOMINAL_SPS=%u,PGA_128,SINGLE_SHOT,OPERATING_MODE=%s\n",
    profileName(active_profile),
    static_cast<unsigned>(profileNominalSps(active_profile)),
    active_profile == PROFILE_CMJ_2000_TURBO ? "TURBO" : "NORMAL"
  );
  Serial.println("#TIMESTAMP_MODE,PER_ADC_START_TO_DRDY_MIDPOINT_AVERAGED_PER_MUX_GROUP");
  Serial.println("#MUX_POLICY,STANDING_FIXED_ORDER;CMJ_ALTERNATING_ORDER;TIMESTAMPS_GROUP_SPECIFIC;PROCESSING_ALIGNMENT_REQUIRED;CMJ_TURBO_FAST_READ");
  Serial.println("#ZERO_CHECK,COMMAND_Z_RETURNS_TARE_CORRECTED_COUNTS");
  Serial.println("#ADC_READ_PATH,Read_Data+DataToInt_FAST_SPI_TRANSACTION");
  Serial.println("#SERIAL_BAUD,921600");
  Serial.println("#STREAM_PROTOCOL,FP1_CSV_CRC16_CCITT_FALSE");
  Serial.println("#CHANNEL_MAP,L1=ADS1_AIN01,L2=ADS2_AIN01,L3=ADS1_AIN23,L4=ADS2_AIN23,R1=ADS3_AIN01,R2=ADS4_AIN01,R3=ADS3_AIN23,R4=ADS4_AIN23");
}

void handleCommand(char command) {
  if (command == '\r' || command == '\n') {
    return;
  }

  if (command == '?') {
    Serial.println("#SYSTEM_READY");
    return;
  }

  if (command == 'I' || command == 'i') {
    printSystemMetadata();
    Serial.println("#SYSTEM_READY");
    return;
  }

  if (command == 'C' || command == 'c') {
    if (daq_state == STATE_STREAMING) {
      Serial.println("#PROFILE_DENIED,STREAMING");
      return;
    }
    configureAdcProfile(PROFILE_CMJ_2000_TURBO, true);
    return;
  }

  if (command == 'G' || command == 'g') {
    if (daq_state == STATE_STREAMING) {
      Serial.println("#PROFILE_DENIED,STREAMING");
      return;
    }
    configureAdcProfile(PROFILE_STANDING_600_NORMAL, true);
    return;
  }

  if (command == 'T' || command == 't') {
    if (daq_state == STATE_STREAMING) {
      Serial.println("#TARE_DENIED,STREAMING");
      return;
    }
    performTare();
    return;
  }

  if (command == 'Z' || command == 'z') {
    performZeroCheck();
    return;
  }

  if (command == 'S' || command == 's') {
    if (!tare_valid) {
      Serial.println("#STREAM_DENIED,NOT_TARED");
      return;
    }

    frame_id = 0;
    daq_state = STATE_STREAMING;
    Serial.println("#STREAM_STARTED");
    return;
  }

  if (command == 'X' || command == 'x') {
    daq_state = STATE_IDLE;
    Serial.println("#STREAM_STOPPED");
    return;
  }

  Serial.print("#UNKNOWN_COMMAND,");
  Serial.println(command);
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(500);
  SPI.begin();

  for (uint8_t i = 0; i < 4; i++) {
    ads[i].begin(CS_PINS[i], DRDY_PINS[i]);
  }

  // Default aman untuk standing. processing.py akan memilih profil CMJ
  // nominal 2000 SPS (DR_1000SPS + MODE_TURBO) sebelum tare ketika mode CMJ dijalankan.
  configureAdcProfile(PROFILE_STANDING_600_NORMAL, false);

  printSystemMetadata();
  Serial.println("#SYSTEM_READY");
  last_ready_announcement_ms = millis();
}

void loop() {
  while (Serial.available() > 0) {
    handleCommand(Serial.read());
  }

  if (daq_state == STATE_IDLE) {
    if (millis() - last_ready_announcement_ms >= 1000) {
      Serial.println("#SYSTEM_READY");
      last_ready_announcement_ms = millis();
    }

    delay(1);
    return;
  }

  uint32_t status_mask = 0;
  int32_t final_data[8] = {0};

  ADCResult mux1[4], mux2[4];
  uint64_t t_mux1 = 0;
  uint64_t t_mux2 = 0;

  const bool reverse_order = (
    active_profile == PROFILE_CMJ_2000_TURBO && (frame_id & 1UL) != 0UL
  );
  readFullFrame(reverse_order, mux1, t_mux1, mux2, t_mux2);

  ADCResult* mapped[8];
  mapMuxResults(mux1, mux2, mapped);

  for (uint8_t ch = 0; ch < 8; ch++) {
    const ADCResult& result = *mapped[ch];
    const uint8_t timeout_bit = ch;
    const uint8_t saturation_bit = ch + 8;

    if (result.timeout) {
      final_data[ch] = 0;
      status_mask |= (1UL << timeout_bit);
      continue;
    }

    int64_t corrected =
      static_cast<int64_t>(result.value) -
      tare_offsets[ch];

    if (corrected > INT32_MAX) {
      corrected = INT32_MAX;
      status_mask |= (1UL << saturation_bit);
    }

    if (corrected < INT32_MIN) {
      corrected = INT32_MIN;
      status_mask |= (1UL << saturation_bit);
    }

    final_data[ch] = static_cast<int32_t>(corrected);

    if (result.saturated) {
      status_mask |= (1UL << saturation_bit);
    }
  }

  const uint32_t next_frame_id = frame_id + 1UL;

  char payload[300];
  const int payload_length = snprintf(
    payload,
    sizeof(payload),
    "FP1,%lu,%llu,%llu,%ld,%ld,%ld,%ld,"
    "%ld,%ld,%ld,%ld,%lu",
    static_cast<unsigned long>(next_frame_id),
    static_cast<unsigned long long>(t_mux1),
    static_cast<unsigned long long>(t_mux2),
    static_cast<long>(final_data[0]),
    static_cast<long>(final_data[1]),
    static_cast<long>(final_data[2]),
    static_cast<long>(final_data[3]),
    static_cast<long>(final_data[4]),
    static_cast<long>(final_data[5]),
    static_cast<long>(final_data[6]),
    static_cast<long>(final_data[7]),
    static_cast<unsigned long>(status_mask)
  );

  if (
    payload_length <= 0 ||
    payload_length >= static_cast<int>(sizeof(payload))
  ) {
    // Tidak menaikkan frame_id bila payload internal gagal dibentuk.
    return;
  }

  const uint16_t crc = crc16Ccitt(
    reinterpret_cast<const uint8_t*>(payload),
    static_cast<size_t>(payload_length)
  );

  char line_buffer[340];
  const int line_length = snprintf(
    line_buffer,
    sizeof(line_buffer),
    "@%s*%04X\n",
    payload,
    static_cast<unsigned>(crc)
  );

  if (
    line_length <= 0 ||
    line_length >= static_cast<int>(sizeof(line_buffer))
  ) {
    return;
  }

  writeSerialAll(
    reinterpret_cast<const uint8_t*>(line_buffer),
    static_cast<size_t>(line_length)
  );
  frame_id = next_frame_id;

}


