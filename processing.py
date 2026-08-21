from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
try:
    import serial
except ImportError:
    serial = None
from scipy.signal import butter, sosfiltfilt




DASHBOARD_VERSION = "1.3"


def finite_number(value: Any, default: float = float("nan")) -> float:
    """Konversi aman ke float tanpa mengubah algoritma pengukuran."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


TRIAL_STATUS_PASS = "PASS"
TRIAL_STATUS_NOTE = "USABLE_WITH_NOTE"
TRIAL_STATUS_REPEAT = "REPEAT_REQUIRED"


def status_level(status: Any) -> str:
    """Kelompokkan status untuk pewarnaan dashboard.

    Status akhir v8.5 dibedakan menjadi:
    - PASS: data utama dan QC sekunder tidak bermasalah.
    - USABLE_WITH_NOTE: hasil utama dapat dipakai, tetapi ada catatan QC sekunder.
    - REPEAT_REQUIRED: core validity / akuisisi / keselamatan tidak memenuhi syarat.

    Status komponen lama seperti CAUTION dan REVIEW tetap dipertahankan agar
    detail diagnosis tidak hilang.
    """
    text = str(status or "").strip().upper().replace(" ", "_")
    if any(token in text for token in (
        "REPEAT_REQUIRED", "REPEAT", "FAIL", "ERROR", "INVALID"
    )):
        return "danger"
    if any(token in text for token in (
        "USABLE_WITH_NOTE", "WITH_NOTE", "PASS_WITH_CAUTION",
        "WITH_REVIEW", "CAUTION", "WARNING", "INFORMATIONAL"
    )):
        return "warning"
    if "REVIEW" in text:
        return "danger"
    if any(token in text for token in ("PASS", "USABLE", "CONSISTENT", "OK")):
        return "success"
    return "neutral"


def final_trial_status(
    repeat_reasons: Iterable[Any],
    note_reasons: Iterable[Any],
) -> str:
    """Gabungkan alasan QC menjadi keputusan akhir tiga tingkat.

    Fungsi ini sengaja tidak mengubah threshold pengukuran. Yang berubah hanya
    cara keputusan akhir dibentuk: masalah core memerlukan pengulangan,
    sedangkan masalah sekunder disimpan sebagai catatan tanpa memaksa ulang.
    """
    repeat = [item for item in repeat_reasons if str(item).strip()]
    notes = [item for item in note_reasons if str(item).strip()]
    if repeat:
        return TRIAL_STATUS_REPEAT
    if notes:
        return TRIAL_STATUS_NOTE
    return TRIAL_STATUS_PASS


def status_requires_repeat(status: Any) -> bool:
    """True hanya untuk keputusan akhir yang memang membutuhkan pengulangan."""
    return str(status or "").strip().upper().replace(" ", "_") == TRIAL_STATUS_REPEAT


def status_palette(status: Any) -> dict[str, str]:
    """Palet warna dashboard berdasarkan status hasil."""
    level = status_level(status)
    palettes = {
        "success": {"background": "#DCFCE7", "foreground": "#166534", "border": "#86EFAC"},
        "warning": {"background": "#FEF3C7", "foreground": "#92400E", "border": "#FCD34D"},
        "danger": {"background": "#FEE2E2", "foreground": "#991B1B", "border": "#FCA5A5"},
        "neutral": {"background": "#E5E7EB", "foreground": "#374151", "border": "#D1D5DB"},
    }
    return palettes[level]


def humanize_status(status: Any) -> str:
    """Ubah kode status menjadi label yang mudah dibaca tanpa mengubah nilainya."""
    text = str(status or "-").strip().upper().replace("_", " ")
    return " ".join(text.split()) or "-"


_CHECK_LABELS = {
    "record_integrity": "Integritas rekaman",
    "both_plates_loaded": "Kedua plate terbebani",
    "physical_bounds": "CoP berada dalam batas fisik",
    "channel_force_valid": "Gaya tiap kanal valid",
    "hampel_ratio": "Rasio koreksi outlier",
    "total_mass_accuracy": "Akurasi massa total",
    "side_mass_accuracy": "Akurasi massa tiap sisi",
    "grf_total_cv": "Variasi GRF total",
    "grf_side_cv": "Variasi GRF kiri/kanan",
    "cop_drift": "Drift CoP",
    "endpoint_shift": "Pergeseran titik awal–akhir",
    "cop_sd": "Simpangan baku CoP",
    "cop_robust_range": "Rentang robust CoP",
    "mean_velocity": "Kecepatan rata-rata CoP",
    "block_step": "Pergeseran CoP antarblok",
    "narrowband_motion": "Gerakan narrowband",
    "load_centering": "Pemusatan beban",
    "expected_cop_accuracy": "Akurasi CoP referensi",
}


def humanize_check_name(name: Any) -> str:
    """Label ramah pengguna untuk nama pemeriksaan QC."""
    key = str(name or "-").strip()
    if key in _CHECK_LABELS:
        return _CHECK_LABELS[key]
    return key.replace("_", " ").strip().capitalize() or "-"


def compact_reason_text(reasons: Iterable[Any], limit: int = 3) -> str:
    """Ringkas daftar alasan untuk banner; detail lengkap tetap ada di tab QC."""
    cleaned = [humanize_check_name(item) for item in reasons if str(item).strip()]
    if not cleaned:
        return "-"
    shown = cleaned[: max(1, int(limit))]
    remaining = len(cleaned) - len(shown)
    suffix = f" (+{remaining} lainnya)" if remaining > 0 else ""
    return ", ".join(shown) + suffix


def format_metric(value: Any, unit: str = "", decimals: int = 2, missing: str = "-") -> str:
    """Format angka untuk kartu hasil dan tabel GUI."""
    number = finite_number(value)
    if not np.isfinite(number):
        return missing
    suffix = f" {unit}" if unit else ""
    return f"{number:.{int(decimals)}f}{suffix}"


def derived_output_path(source_filename: str | Path, suffix: str, extension: str) -> Path:
    """Bangun nama file turunan secara konsisten dari CSV pengukuran."""
    source = Path(source_filename)
    ext = extension if str(extension).startswith(".") else f".{extension}"
    return source.with_suffix("").with_name(source.stem + suffix).with_suffix(ext)


def open_path_with_default_app(path: str | Path) -> tuple[bool, str]:
    """Buka file/folder dengan aplikasi bawaan OS tanpa membuat program kelima."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return False, f"File tidak ditemukan: {target}"

    try:
        if sys.platform.startswith("win"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return True, str(target)


def write_operator_note(
    source_filename: str | Path,
    mode: str,
    status: str,
    note: str,
) -> Path:
    """Simpan catatan operator sebagai file teks pendamping hasil."""
    output = derived_output_path(source_filename, f"_{mode.lower()}_operator_note", ".txt")
    timestamp = datetime.now().isoformat(timespec="seconds")
    content = (
        f"Mode: {mode}\n"
        f"Status: {status}\n"
        f"Timestamp: {timestamp}\n"
        f"Source data: {Path(source_filename).name}\n\n"
        f"{str(note).strip()}\n"
    )
    output.write_text(content, encoding="utf-8")
    return output


def write_key_value_csv(
    output_path: str | Path,
    rows: Iterable[tuple[Any, Any, Any]],
) -> Path:
    """Simpan tabel ringkasan sederhana dengan kolom metric, value, unit."""
    output = Path(output_path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Value", "Unit"])
        writer.writerows(rows)
    return output


GRAVITY = 9.80665
SERIAL_PORT = "COM9"
BAUD_RATE = 921600
SOFTWARE_VERSION = "8.5_CRC_FRAMED_GAP_ROBUST"

# ==============================================================
# KALIBRASI HASIL REFINEMENT 12 KONDISI STATIS — 11 AGUSTUS 2026
#
# Dasar:
# - 2 beban referensi: 15.4 kg dan 35.4 kg
# - plate LEFT dan RIGHT diuji bergantian
# - posisi CENTER, FRONT, BACK
# - 6 kondisi per plate
#
# Faktor ditetapkan per-plate (bukan per-sensor) karena posisi pusat beban
# tidak diketahui cukup presisi untuk mengestimasi delapan faktor individual.
# Urutan kanal: L1, L2, L3, L4, R1, R2, R3, R4.
# Satuan: ADC counts per kilogram.
# ==============================================================
CALIBRATION_VERSION = "STATIC_15P4_35P4_12COND_20260811"
CAL_FACTORS = np.array(
    [
        17553.7, 17553.7, 17553.7, 17553.7,
        17591.5, 17591.5, 17591.5, 17591.5,
    ],
    dtype=float,
)

CHANNEL_SIGN = np.ones(8, dtype=float)

# Geometri sesuai rancangan:
# top plate 225 mm x 400 mm; pusat load cell 20 mm dari tepi.
PLATE_WIDTH_M = 0.225
PLATE_LENGTH_M = 0.400
PLATE_CENTER_DISTANCE_M = 0.300
SENSOR_X_LOCAL_M = np.array(
    [-0.0925, -0.0925, 0.0925, 0.0925,
     -0.0925, -0.0925, 0.0925, 0.0925],
    dtype=float,
)
SENSOR_Y_LOCAL_M = np.array(
    [0.1800, -0.1800, 0.1800, -0.1800,
     0.1800, -0.1800, 0.1800, -0.1800],
    dtype=float,
)

FIXED_CONFIG: dict[str, Any] = {
    "serial_port": SERIAL_PORT,
    "baud_rate": BAUD_RATE,
    "gravity_m_s2": GRAVITY,
    "require_crc_stream_protocol": True,
    "calibration_version": CALIBRATION_VERSION,
    "calibration_source": "12 static conditions: 15.4/35.4 kg, LEFT/RIGHT, CENTER/FRONT/BACK",
    "counts_per_kg": CAL_FACTORS.tolist(),
    "channel_sign": CHANNEL_SIGN.tolist(),
    "sensor_x_local_m": SENSOR_X_LOCAL_M.tolist(),
    "sensor_y_local_m": SENSOR_Y_LOCAL_M.tolist(),
    "plate_width_m": PLATE_WIDTH_M,
    "plate_length_m": PLATE_LENGTH_M,
    "plate_center_distance_m": PLATE_CENTER_DISTANCE_M,
    "min_plate_force_n": 20.0,
    "min_total_force_n": 40.0,

    # Recovery/zero gate berdasarkan temuan static validation:
    # beberapa kondisi baru kembali <= ~0.10 kg setelah settling kedua.
    "pre_tare_settle_s": 5.0,
    "post_tare_zero_settle_s": 2.0,
    "zero_check_retry_delay_s": 5.0,
    "zero_check_max_attempts": 2,
    "zero_total_limit_kg": 0.10,
    "zero_side_limit_kg": 0.10,

    # MUX alignment v8.0:
    # AIN01 dan AIN23 tetap sequential karena satu ADS1220 membaca dua load cell.
    # Rasio raw MUX separation terhadap frame tidak lagi dijadikan syarat PASS/FAIL
    # karena secara arsitektur nilainya cenderung mendekati ~1/2 frame.
    # QC CMJ memakai separation absolut + alignment coverage + alternating order.
    "mux_alignment_method": "linear_to_group_midpoint",
    "mux_alignment_min_coverage": 0.98,
    "cmj_mux_skew_abs_caution_us": 1500.0,
    "cmj_mux_skew_abs_review_us": 2500.0,
    "cmj_mux_fraction_informational_only": True,
    "cmj_temporal_target_frame_rate_hz": 300.0,

    # Integritas serial/gap v8.5. Gap kecil terisolasi tidak otomatis membuat
    # seluruh trial gagal karena integral impulse menggunakan timestamp aktual.
    # Gap besar/beruntun tetap merupakan core failure.
    "cmj_gap_hard_max_consecutive_missing": 3,
    "cmj_gap_hard_missing_fraction": 0.0025,
    "cmj_gap_hard_duration_s": 0.010,
    "standing_gap_hard_max_consecutive_missing": 8,
    "standing_gap_hard_missing_fraction": 0.002,
    "standing_gap_hard_duration_s": 0.025,

    # Landing bilateral.
    "cmj_side_landing_threshold_n": 25.0,
    "cmj_side_landing_persistence_s": 0.015,
    "cmj_staggered_landing_threshold_s": 0.015,

    "standing_force_filter_hz": 5.0,
    "cmj_force_filter_hz": 0.0,
    "standing_adc_nominal_sps": 600,
    "cmj_adc_nominal_sps": 2000,
    "cmj_min_effective_sampling_hz": 200.0,
    "cmj_impact_filter_hz": 60.0,
    "cmj_impact_filter_max_fraction_fs": 0.28,
    # Impact characterization v8.5:
    # 60 Hz tetap menjadi cutoff UTAMA yang dilaporkan. 40/50 Hz hanya untuk
    # sensitivity audit internal; perbedaannya tidak memaksa pengulangan bila
    # core validity CMJ tetap baik.
    "cmj_impact_sensitivity_cutoffs_hz": [40.0, 50.0, 60.0],
    "cmj_impact_peak_variation_caution_percent": 5.0,
    # Landing impulse dilaporkan karena lebih robust terhadap pilihan cutoff
    # dibanding satu titik peak force.
    "cmj_landing_impulse_windows_s": [0.050, 0.100],
    "cmj_rfd_variation_caution_percent": 20.0,
    "cmj_landing_second_peak_ratio_caution": 0.50,

    # Kebijakan keputusan akhir v8.5:
    # - acquisition / primary impulse / temporal hard review / safety limit
    #   -> REPEAT_REQUIRED
    # - flight-time, bilateral biomechanics, impact sensitivity, landing-rate,
    #   post-landing audit -> USABLE_WITH_NOTE
    # Threshold individual tetap dipertahankan agar transparansi QC tidak hilang.
    # QC v7.0: estimasi zero-force saat flight memakai bagian tengah flight,
    # sehingga sampel transisi take-off/landing tidak menaikkan offset secara palsu.
    "cmj_flight_zero_guard_s": 0.020,
    "cmj_flight_zero_min_samples": 12,
    "cmj_flight_offset_caution_percent_bw": 1.0,
    "cmj_flight_offset_review_percent_bw": 3.0,
    # Loading-rate 20–80% tetap dilaporkan, tetapi pada frame rate efektif yang bergantung pada dua kelompok MUX dapat hanya terdiri dari beberapa sampel. Karena itu v7.0 juga
    # menghitung kemiringan linear awal 30 ms dan 50 ms.
    "cmj_early_loading_windows_s": [0.030, 0.050],
    "cmj_early_loading_primary_window_s": 0.050,
    "cmj_early_loading_min_samples": 6,
    "cmj_early_loading_r2_caution": 0.80,
    "cmj_rfd_20_80_min_samples": 6.0,
    "cmj_live_event_tolerance_samples": 2.0,
    "cmj_mux_alternation_min_fraction": 0.80,
    "cmj_mux_signed_bias_max_fraction": 0.20,
    "human_standing_record_duration_s": 40.0,
    "human_analysis_start_s": 7.0,
    "human_analysis_duration_s": 30.0,
    "human_post_monitor_s": 3.0,
    "static_rigid_record_duration_s": 55.0,
    "static_liquid_record_duration_s": 85.0,
    "cmj_record_duration_s": 35.0,
    "cmj_min_prejump_s": 2.0,
    "cmj_flight_threshold_n": 25.0,
    "cmj_landing_threshold_n": 50.0,
    "cmj_event_persistence_s": 0.025,
    "cmj_post_search_delay_s": 0.60,
    "cmj_post_landing_min_s": 2.50,
    "cmj_post_landing_max_s": 7.0,
    "cmj_post_stable_window_s": 0.80,
    "cmj_post_total_cv_limit": 0.015,
    "cmj_post_side_cv_limit": 0.050,
    "cmj_post_total_slope_limit_n_s": 10.0,
    "cmj_post_side_slope_limit_n_s": 20.0,
    "cmj_post_cop_range_limit_m": 0.025,
    "cmj_post_cop_slope_limit_m_s": 0.010,
    "cmj_post_stable_hold_s": 0.40,
    "cmj_post_capture_tail_s": 0.75,
    "cmj_live_event_median_samples": 3,
    "cmj_force_drift_max_fraction_bw": 0.030,
    # Audit closure post-landing tidak mengubah velocity pra-landing.
    # Kelayakan koreksi dinilai dari gaya ekuivalen yang dibutuhkan, bukan
    # hanya dari besar residual velocity yang dapat membesar pada rekaman panjang.
    "cmj_post_closure_max_equiv_force_fraction_bw": 0.030,
    "cmj_post_closure_min_duration_s": 0.35,
    # Dipertahankan untuk kompatibilitas file lama; v7.0 tidak lagi memakai
    # residual velocity tunggal sebagai syarat utama penerapan audit closure.
    "cmj_zupt_max_residual_m_s": 0.100,
    "cmj_terminal_velocity_limit_m_s": 0.030,
    "cmj_ballistic_velocity_caution_m_s": 0.080,
    "cmj_ballistic_velocity_review_m_s": 0.150,
    "cmj_primary_method": "impulse_momentum",
    "cmj_baseline_guard_s": 0.15,
    "cmj_primary_baseline_cv_limit": 0.010,
    "cmj_primary_baseline_slope_limit_n_s": 5.0,
    "cmj_impulse_height_threshold_caution_cm": 0.75,
    "cmj_impulse_height_threshold_review_cm": 1.50,
    "design_max_force_n": 4905.0,
}


SENSOR_NAMES = ("L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4")
LEFT_INDICES = np.array([0, 1, 2, 3], dtype=int)
RIGHT_INDICES = np.array([4, 5, 6, 7], dtype=int)
MUX1_CHANNELS = (0, 1, 4, 5)
MUX2_CHANNELS = (2, 3, 6, 7)

ESP32_BOOT_WAIT_S = 2.5
READY_TIMEOUT_S = 15.0
TARE_TIMEOUT_S = 30.0
STREAM_TIMEOUT_S = 5.0
SERIAL_SILENCE_TIMEOUT_S = 5.0
VALID_FRAME_TIMEOUT_S = 5.0
SUBJECT_WAIT_TIMEOUT_S = 120.0
MAX_TARE_ATTEMPTS = 3


def load_config() -> dict[str, Any]:
    """Mengembalikan konfigurasi tetap dengan faktor kalibrasi yang sama."""
    config: dict[str, Any] = {}
    for key, value in FIXED_CONFIG.items():
        config[key] = list(value) if isinstance(value, list) else value
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    for key in (
        "counts_per_kg",
        "channel_sign",
        "sensor_x_local_m",
        "sensor_y_local_m",
    ):
        values = np.asarray(config[key], dtype=float)
        if values.shape != (8,):
            raise ValueError(f"{key} harus berisi tepat 8 nilai.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key} mengandung nilai tidak valid.")

    calibration = np.asarray(config["counts_per_kg"], dtype=float)
    if np.any(calibration <= 0):
        raise ValueError("Semua counts_per_kg harus lebih besar dari nol.")

    signs = np.asarray(config["channel_sign"], dtype=float)
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("channel_sign hanya boleh berisi -1 atau 1.")

    if float(config["plate_center_distance_m"]) <= 0:
        raise ValueError("plate_center_distance_m harus positif.")
    if float(config["plate_width_m"]) <= 0:
        raise ValueError("plate_width_m harus positif.")
    if float(config["plate_length_m"]) <= 0:
        raise ValueError("plate_length_m harus positif.")

    for key in (
        "cmj_gap_hard_max_consecutive_missing",
        "standing_gap_hard_max_consecutive_missing",
    ):
        if int(config[key]) < 1:
            raise ValueError(f"{key} minimal 1.")
    for key in (
        "cmj_gap_hard_missing_fraction",
        "standing_gap_hard_missing_fraction",
    ):
        value = float(config[key])
        if not 0.0 < value <= 0.05:
            raise ValueError(f"{key} harus >0 dan <=0.05.")
    for key in (
        "cmj_gap_hard_duration_s",
        "standing_gap_hard_duration_s",
    ):
        if float(config[key]) <= 0.0:
            raise ValueError(f"{key} harus positif.")

    human_start = float(config["human_analysis_start_s"])
    human_duration = float(config["human_analysis_duration_s"])
    human_post = float(config["human_post_monitor_s"])
    human_record = float(config["human_standing_record_duration_s"])
    if min(human_start, human_duration, human_post) < 0.0:
        raise ValueError("Parameter waktu standing manusia tidak boleh negatif.")
    if human_duration <= 0.0:
        raise ValueError("human_analysis_duration_s harus positif.")
    required_record = human_start + human_duration + human_post
    if human_record + 1e-9 < required_record:
        raise ValueError(
            "human_standing_record_duration_s terlalu pendek untuk window "
            "analisis dan post-monitor."
        )


    cmj_min_prejump = float(config["cmj_min_prejump_s"])
    cmj_post_delay = float(config["cmj_post_search_delay_s"])
    cmj_post_min = float(config["cmj_post_landing_min_s"])
    cmj_post_max = float(config["cmj_post_landing_max_s"])
    cmj_post_window = float(config["cmj_post_stable_window_s"])
    cmj_post_hold = float(config["cmj_post_stable_hold_s"])
    cmj_post_tail = float(config["cmj_post_capture_tail_s"])
    if min(
        cmj_min_prejump,
        cmj_post_delay,
        cmj_post_min,
        cmj_post_window,
        cmj_post_hold,
        cmj_post_tail,
    ) < 0.0:
        raise ValueError("Parameter waktu CMJ tidak boleh negatif.")
    if cmj_post_min <= 0.0 or cmj_post_max < cmj_post_min:
        raise ValueError("Batas waktu post-landing CMJ tidak valid.")
    if cmj_post_window <= 0.0 or cmj_post_window > cmj_post_max:
        raise ValueError("Window stabil post-landing CMJ tidak valid.")
    if cmj_post_hold + cmj_post_tail > cmj_post_max:
        raise ValueError("Hold dan tail post-landing CMJ terlalu panjang.")
    if float(config["cmj_flight_threshold_n"]) <= 0.0:
        raise ValueError("cmj_flight_threshold_n harus positif.")
    if float(config["cmj_landing_threshold_n"]) <= float(config["cmj_flight_threshold_n"]):
        raise ValueError("Threshold landing CMJ harus lebih besar dari threshold flight.")
    if int(config["cmj_live_event_median_samples"]) < 1:
        raise ValueError("cmj_live_event_median_samples minimal 1.")
    if float(config["cmj_force_drift_max_fraction_bw"]) <= 0.0:
        raise ValueError("cmj_force_drift_max_fraction_bw harus positif.")
    if float(config["cmj_zupt_max_residual_m_s"]) <= 0.0:
        raise ValueError("cmj_zupt_max_residual_m_s harus positif.")
    if not 0.0 < float(config["cmj_post_closure_max_equiv_force_fraction_bw"]) <= 0.10:
        raise ValueError(
            "cmj_post_closure_max_equiv_force_fraction_bw harus antara 0 dan 0.10."
        )
    if float(config["cmj_post_closure_min_duration_s"]) <= 0.0:
        raise ValueError("cmj_post_closure_min_duration_s harus positif.")
    if float(config["cmj_terminal_velocity_limit_m_s"]) <= 0.0:
        raise ValueError("cmj_terminal_velocity_limit_m_s harus positif.")
    if float(config["cmj_ballistic_velocity_caution_m_s"]) <= 0.0:
        raise ValueError("cmj_ballistic_velocity_caution_m_s harus positif.")
    if float(config["cmj_ballistic_velocity_review_m_s"]) <= float(
        config["cmj_ballistic_velocity_caution_m_s"]
    ):
        raise ValueError(
            "cmj_ballistic_velocity_review_m_s harus lebih besar dari batas caution."
        )
    if str(config.get("cmj_primary_method", "")).lower() != "impulse_momentum":
        raise ValueError("cmj_primary_method harus 'impulse_momentum'.")
    if float(config["cmj_baseline_guard_s"]) < 0.0:
        raise ValueError("cmj_baseline_guard_s tidak boleh negatif.")
    if float(config["cmj_primary_baseline_cv_limit"]) <= 0.0:
        raise ValueError("cmj_primary_baseline_cv_limit harus positif.")
    if float(config["cmj_primary_baseline_slope_limit_n_s"]) <= 0.0:
        raise ValueError("cmj_primary_baseline_slope_limit_n_s harus positif.")
    if float(config["cmj_impulse_height_threshold_caution_cm"]) <= 0.0:
        raise ValueError("Batas caution sensitivitas tinggi impulse harus positif.")
    if float(config["cmj_impulse_height_threshold_review_cm"]) <= float(config["cmj_impulse_height_threshold_caution_cm"]):
        raise ValueError("Batas review sensitivitas tinggi impulse harus lebih besar dari caution.")
    if not 0.10 <= float(config["cmj_impact_filter_max_fraction_fs"]) <= 0.40:
        raise ValueError("cmj_impact_filter_max_fraction_fs harus antara 0.10 dan 0.40.")
    impact_cutoffs = np.asarray(config["cmj_impact_sensitivity_cutoffs_hz"], dtype=float)
    if impact_cutoffs.ndim != 1 or impact_cutoffs.size < 2:
        raise ValueError("cmj_impact_sensitivity_cutoffs_hz minimal berisi dua cutoff.")
    if not np.all(np.isfinite(impact_cutoffs)) or np.any(impact_cutoffs <= 0.0):
        raise ValueError("Cutoff sensitivitas impact harus positif dan finite.")
    if float(config["cmj_impact_peak_variation_caution_percent"]) <= 0.0:
        raise ValueError("Batas variasi peak impact harus positif.")
    landing_impulse_windows = np.asarray(
        config.get("cmj_landing_impulse_windows_s", [0.050, 0.100]),
        dtype=float,
    )
    if landing_impulse_windows.ndim != 1 or landing_impulse_windows.size < 1:
        raise ValueError("cmj_landing_impulse_windows_s minimal berisi satu window.")
    if (
        not np.all(np.isfinite(landing_impulse_windows))
        or np.any(landing_impulse_windows <= 0.0)
    ):
        raise ValueError("Semua landing impulse window harus positif dan finite.")
    if np.any(np.diff(np.sort(landing_impulse_windows)) <= 0.0):
        raise ValueError("Landing impulse window harus unik.")
    if float(config["cmj_rfd_variation_caution_percent"]) <= 0.0:
        raise ValueError("Batas variasi RFD harus positif.")
    if not 0.0 < float(config["cmj_landing_second_peak_ratio_caution"]) < 1.0:
        raise ValueError("Rasio peak landing kedua harus antara 0 dan 1.")
    if float(config["cmj_flight_zero_guard_s"]) < 0.0:
        raise ValueError("cmj_flight_zero_guard_s tidak boleh negatif.")
    if int(config["cmj_flight_zero_min_samples"]) < 5:
        raise ValueError("cmj_flight_zero_min_samples minimal 5.")
    flight_offset_caution = float(config["cmj_flight_offset_caution_percent_bw"])
    flight_offset_review = float(config["cmj_flight_offset_review_percent_bw"])
    if flight_offset_caution <= 0.0 or flight_offset_review <= flight_offset_caution:
        raise ValueError("Batas offset flight harus positif dan review > caution.")
    loading_windows = np.asarray(config["cmj_early_loading_windows_s"], dtype=float)
    if loading_windows.ndim != 1 or loading_windows.size < 1:
        raise ValueError("cmj_early_loading_windows_s minimal berisi satu window.")
    if not np.all(np.isfinite(loading_windows)) or np.any(loading_windows <= 0.0):
        raise ValueError("Semua early loading window harus positif dan finite.")
    if float(config["cmj_early_loading_primary_window_s"]) <= 0.0:
        raise ValueError("cmj_early_loading_primary_window_s harus positif.")
    if int(config["cmj_early_loading_min_samples"]) < 4:
        raise ValueError("cmj_early_loading_min_samples minimal 4.")
    r2_limit = float(config["cmj_early_loading_r2_caution"])
    if not 0.0 < r2_limit < 1.0:
        raise ValueError("cmj_early_loading_r2_caution harus antara 0 dan 1.")
    if float(config["cmj_rfd_20_80_min_samples"]) < 3.0:
        raise ValueError("cmj_rfd_20_80_min_samples minimal 3 sampel.")
    if float(config["cmj_live_event_tolerance_samples"]) <= 0.0:
        raise ValueError("cmj_live_event_tolerance_samples harus positif.")
    if not 0.0 < float(config["cmj_mux_alternation_min_fraction"]) <= 1.0:
        raise ValueError("cmj_mux_alternation_min_fraction harus antara 0 dan 1.")
    if not 0.0 <= float(config["cmj_mux_signed_bias_max_fraction"]) <= 1.0:
        raise ValueError("cmj_mux_signed_bias_max_fraction harus antara 0 dan 1.")

    if float(config["pre_tare_settle_s"]) < 0.0:
        raise ValueError("pre_tare_settle_s tidak boleh negatif.")
    if float(config["post_tare_zero_settle_s"]) < 0.0:
        raise ValueError("post_tare_zero_settle_s tidak boleh negatif.")
    if float(config["zero_check_retry_delay_s"]) < 0.0:
        raise ValueError("zero_check_retry_delay_s tidak boleh negatif.")
    if int(config["zero_check_max_attempts"]) < 1:
        raise ValueError("zero_check_max_attempts minimal 1.")
    if float(config["zero_total_limit_kg"]) <= 0.0:
        raise ValueError("zero_total_limit_kg harus positif.")
    if float(config["zero_side_limit_kg"]) <= 0.0:
        raise ValueError("zero_side_limit_kg harus positif.")
    if not 0.90 <= float(config["mux_alignment_min_coverage"]) <= 1.0:
        raise ValueError("mux_alignment_min_coverage harus antara 0.90 dan 1.00.")
    mux_caution_us = float(config["cmj_mux_skew_abs_caution_us"])
    mux_review_us = float(config["cmj_mux_skew_abs_review_us"])
    if not 0.0 < mux_caution_us < mux_review_us:
        raise ValueError("Batas MUX skew absolut harus 0 < caution < review.")
    if float(config["cmj_temporal_target_frame_rate_hz"]) <= 0.0:
        raise ValueError("cmj_temporal_target_frame_rate_hz harus positif.")
    if float(config["cmj_side_landing_threshold_n"]) <= 0.0:
        raise ValueError("cmj_side_landing_threshold_n harus positif.")
    if float(config["cmj_side_landing_persistence_s"]) <= 0.0:
        raise ValueError("cmj_side_landing_persistence_s harus positif.")
    if float(config["cmj_staggered_landing_threshold_s"]) <= 0.0:
        raise ValueError("cmj_staggered_landing_threshold_s harus positif.")

    if float(config["design_max_force_n"]) <= 0.0:
        raise ValueError("design_max_force_n harus positif.")


def butter_lowpass_filter(data, cutoff, fs, order=4):
    values = np.asarray(data, dtype=float)

    if (
        values.shape[0] < 30
        or not np.isfinite(fs)
        or fs <= 0
        or cutoff <= 0
    ):
        return values.copy()

    normalized_cutoff = cutoff / (0.5 * fs)
    if not 0.0 < normalized_cutoff < 1.0:
        return values.copy()

    sos = butter(order, normalized_cutoff, btype="low", output="sos")
    return sosfiltfilt(sos, values, axis=0)


def counts_to_force_channels(raw_counts, config: dict[str, Any] | None = None):
    cfg = load_config() if config is None else config
    counts = np.asarray(raw_counts, dtype=float)

    if counts.ndim == 1:
        if counts.shape != (8,):
            raise ValueError("raw_counts harus berisi tepat 8 kanal.")
        counts = counts.reshape(1, 8)
        squeeze = True
    elif counts.ndim == 2 and counts.shape[1] == 8:
        squeeze = False
    else:
        raise ValueError("raw_counts harus berbentuk 8 atau N x 8.")

    counts_per_kg = np.asarray(cfg["counts_per_kg"], dtype=float)
    signs = np.asarray(cfg["channel_sign"], dtype=float)
    gravity = float(cfg["gravity_m_s2"])

    force_n = counts * signs / counts_per_kg * gravity
    return force_n[0] if squeeze else force_n


def _global_sensor_coordinates(config: dict[str, Any]):
    x_local = np.asarray(config["sensor_x_local_m"], dtype=float)
    y_global = np.asarray(config["sensor_y_local_m"], dtype=float)
    half_distance = float(config["plate_center_distance_m"]) / 2.0

    x_global = x_local.copy()
    x_global[LEFT_INDICES] -= half_distance
    x_global[RIGHT_INDICES] += half_distance
    return x_global, y_global


def calculate_signals_from_force_channels(
    force_channels_n,
    config: dict[str, Any] | None = None,
):
    cfg = load_config() if config is None else config
    force_n = np.asarray(force_channels_n, dtype=float)

    if force_n.ndim != 2 or force_n.shape[1] != 8:
        raise ValueError("force_channels_n harus berbentuk N x 8.")

    fz_l = np.sum(force_n[:, LEFT_INDICES], axis=1)
    fz_r = np.sum(force_n[:, RIGHT_INDICES], axis=1)
    grf = fz_l + fz_r

    x_global, y_global = _global_sensor_coordinates(cfg)
    x_local = np.asarray(cfg["sensor_x_local_m"], dtype=float)

    min_plate_force = float(cfg["min_plate_force_n"])
    min_total_force = float(cfg["min_total_force_n"])

    cop_l_ap = np.full(len(force_n), np.nan)
    cop_l_ml = np.full(len(force_n), np.nan)
    cop_r_ap = np.full(len(force_n), np.nan)
    cop_r_ml = np.full(len(force_n), np.nan)
    cop_ap = np.full(len(force_n), np.nan)
    cop_ml = np.full(len(force_n), np.nan)

    valid_l = fz_l > min_plate_force
    valid_r = fz_r > min_plate_force
    valid_total = grf > min_total_force

    if np.any(valid_l):
        left_force = force_n[valid_l][:, LEFT_INDICES]
        cop_l_ap[valid_l] = (
            left_force @ y_global[LEFT_INDICES]
        ) / fz_l[valid_l]
        cop_l_ml[valid_l] = (
            left_force @ x_local[LEFT_INDICES]
        ) / fz_l[valid_l]

    if np.any(valid_r):
        right_force = force_n[valid_r][:, RIGHT_INDICES]
        cop_r_ap[valid_r] = (
            right_force @ y_global[RIGHT_INDICES]
        ) / fz_r[valid_r]
        cop_r_ml[valid_r] = (
            right_force @ x_local[RIGHT_INDICES]
        ) / fz_r[valid_r]

    if np.any(valid_total):
        valid_force = force_n[valid_total]
        cop_ap[valid_total] = (valid_force @ y_global) / grf[valid_total]
        cop_ml[valid_total] = (valid_force @ x_global) / grf[valid_total]

    return {
        "force_channels_n": force_n,
        "fz_l": fz_l,
        "fz_r": fz_r,
        "grf": grf,
        "cop_l_ap": cop_l_ap,
        "cop_l_ml": cop_l_ml,
        "cop_r_ap": cop_r_ap,
        "cop_r_ml": cop_r_ml,
        "cop_ap": cop_ap,
        "cop_ml": cop_ml,
    }


def calculate_forces_and_cop(raw_counts, config: dict[str, Any] | None = None):
    force = counts_to_force_channels(raw_counts, config).reshape(1, 8)
    result = calculate_signals_from_force_channels(force, config)
    return {
        key: float(value[0])
        for key, value in result.items()
        if key != "force_channels_n"
    }


def calculate_forces_and_cop_batch(
    raw_counts,
    config: dict[str, Any] | None = None,
    fs: float | None = None,
    force_cutoff_hz: float = 0.0,
):
    cfg = load_config() if config is None else config
    force_raw = counts_to_force_channels(raw_counts, cfg)
    force_used = force_raw.copy()

    if (
        fs is not None
        and np.isfinite(fs)
        and fs > 0
        and force_cutoff_hz > 0
    ):
        force_used = butter_lowpass_filter(
            force_raw,
            cutoff=force_cutoff_hz,
            fs=fs,
            order=4,
        )

    result = calculate_signals_from_force_channels(force_used, cfg)
    result["force_channels_raw_n"] = force_raw
    result["force_channels_filtered_n"] = force_used
    return result


def _interpolate_monotonic(source_time, values, target_time):
    source_time = np.asarray(source_time, dtype=float)
    values = np.asarray(values, dtype=float)
    target_time = np.asarray(target_time, dtype=float)

    keep = np.concatenate(([True], np.diff(source_time) > 0))
    source_unique = source_time[keep]
    values_unique = values[keep]

    if len(source_unique) < 2:
        return np.full(len(target_time), np.nan)

    return np.interp(
        target_time,
        source_unique,
        values_unique,
        left=np.nan,
        right=np.nan,
    )


def align_mux_channels(t_mux1_us, t_mux2_us, raw_counts):
    t1 = np.asarray(t_mux1_us, dtype=float)
    t2 = np.asarray(t_mux2_us, dtype=float)
    raw = np.asarray(raw_counts, dtype=float)

    if (
        len(t1) != len(t2)
        or len(t1) != len(raw)
        or raw.ndim != 2
        or raw.shape[1] != 8
    ):
        raise ValueError("Data MUX dan raw count tidak sejajar.")

    common_time_us = (t1 + t2) / 2.0
    aligned = np.full(raw.shape, np.nan, dtype=float)

    for channel in MUX1_CHANNELS:
        aligned[:, channel] = _interpolate_monotonic(
            t1, raw[:, channel], common_time_us
        )

    for channel in MUX2_CHANNELS:
        aligned[:, channel] = _interpolate_monotonic(
            t2, raw[:, channel], common_time_us
        )

    valid = np.isfinite(common_time_us) & np.all(np.isfinite(aligned), axis=1)
    return common_time_us[valid], aligned[valid], valid


def get_sampling_diagnostics(frame_arr, time_arr):
    """Ringkas kualitas timing dan gap berdasarkan frame-id + timestamp aktual.

    v8.5 membedakan *jumlah frame hilang* dari *keparahan gap*. Satu atau dua
    frame yang hilang terisolasi pada stream panjang tidak diperlakukan sama
    dengan burst gap puluhan milidetik. Ini penting karena integrasi impulse
    menggunakan timestamp aktual, sementara gap besar tetap dapat merusak event
    detection/filtering dan harus ditandai keras.
    """
    frames = np.asarray(frame_arr, dtype=np.int64).ravel()
    timestamps = np.asarray(time_arr, dtype=float).ravel()

    if len(frames) != len(timestamps):
        raise ValueError(
            f"Frame dan waktu tidak sejajar: {len(frames)} vs {len(timestamps)}."
        )

    empty = {
        "fs": np.nan,
        "mean_dt": np.nan,
        "jitter_sd": np.nan,
        "analysis_frame_gaps": 0,
        "gap_events": 0,
        "max_consecutive_missing": 0,
        "missing_fraction": 0.0,
        "max_gap_interval_s": 0.0,
        "estimated_max_missing_duration_s": 0.0,
        "duplicates": 0,
        "out_of_order": 0,
    }
    if len(frames) < 2:
        return empty.copy()

    frame_delta = np.diff(frames)
    time_delta = np.diff(timestamps)
    valid = (frame_delta > 0) & (time_delta > 0) & np.isfinite(time_delta)

    duplicates = int(np.sum(frame_delta == 0))
    out_of_order = int(np.sum(frame_delta < 0))
    missing_each = np.clip(frame_delta - 1, 0, None)
    missing_total = int(np.sum(missing_each))
    gap_mask = missing_each > 0
    gap_events = int(np.sum(gap_mask))
    max_missing = int(np.max(missing_each)) if gap_events else 0

    if not np.any(valid):
        result = empty.copy()
        result.update({
            "analysis_frame_gaps": missing_total,
            "gap_events": gap_events,
            "max_consecutive_missing": max_missing,
            "duplicates": duplicates,
            "out_of_order": out_of_order,
        })
        return result

    dt_per_frame = time_delta[valid] / frame_delta[valid]
    median_dt = float(np.median(dt_per_frame))
    fs = float(1.0 / median_dt) if median_dt > 0 else np.nan

    expected_span = int(max(1, frames[-1] - frames[0] + 1))
    missing_fraction = float(missing_total / expected_span)
    max_gap_interval_s = (
        float(np.max(time_delta[gap_mask & np.isfinite(time_delta)]))
        if np.any(gap_mask & np.isfinite(time_delta))
        else 0.0
    )
    estimated_missing_duration = (
        float(max_missing * median_dt)
        if np.isfinite(median_dt) and median_dt > 0
        else 0.0
    )

    return {
        "fs": fs,
        "mean_dt": median_dt,
        "jitter_sd": float(np.std(dt_per_frame)),
        "analysis_frame_gaps": missing_total,
        "gap_events": gap_events,
        "max_consecutive_missing": max_missing,
        "missing_fraction": missing_fraction,
        "max_gap_interval_s": max_gap_interval_s,
        "estimated_max_missing_duration_s": estimated_missing_duration,
        "duplicates": duplicates,
        "out_of_order": out_of_order,
    }


def classify_gap_severity(
    diagnostics: dict[str, Any],
    *,
    hard_max_consecutive_missing: int,
    hard_missing_fraction: float,
    hard_duration_s: float,
) -> tuple[str, list[str]]:
    """Klasifikasi gap: PASS, NOTE, atau REPEAT_REQUIRED.

    Tidak mengisi/mengarang sampel yang hilang. Fungsi hanya menentukan apakah
    gap yang benar-benar terdeteksi cukup kecil untuk tetap dipakai dengan
    catatan atau cukup besar sehingga trial harus diulang.
    """
    missing = int(diagnostics.get("analysis_frame_gaps", 0) or 0)
    events = int(diagnostics.get("gap_events", 0) or 0)
    max_missing = int(diagnostics.get("max_consecutive_missing", 0) or 0)
    fraction = float(diagnostics.get("missing_fraction", 0.0) or 0.0)
    duration = float(diagnostics.get("estimated_max_missing_duration_s", 0.0) or 0.0)
    duplicates = int(diagnostics.get("duplicates", 0) or 0)
    out_of_order = int(diagnostics.get("out_of_order", 0) or 0)

    reasons: list[str] = []
    if out_of_order > 0:
        reasons.append(f"terdapat {out_of_order} frame out-of-order")
    if duplicates > 0:
        reasons.append(f"terdapat {duplicates} frame duplikat")

    hard = (
        out_of_order > 0
        or max_missing > int(hard_max_consecutive_missing)
        or fraction > float(hard_missing_fraction)
        or duration > float(hard_duration_s)
    )
    if hard:
        if missing > 0:
            reasons.append(
                "gap frame melebihi batas aman "
                f"(hilang={missing}, event={events}, max-beruntun={max_missing}, "
                f"fraksi={fraction * 100.0:.3f}%, estimasi durasi maks={duration * 1000.0:.2f} ms)"
            )
        return "REPEAT_REQUIRED", reasons

    if missing > 0 or duplicates > 0:
        reasons.append(
            "gap kecil terisolasi pada stream "
            f"(hilang={missing}, event={events}, max-beruntun={max_missing}, "
            f"fraksi={fraction * 100.0:.3f}%, estimasi durasi maks={duration * 1000.0:.2f} ms)"
        )
        return "NOTE", reasons

    return "PASS", reasons


def get_serial_diagnostics(all_received_frames):
    frames = np.asarray(all_received_frames, dtype=np.int64).ravel()
    if len(frames) < 2:
        return {
            "serial_lost_frames": 0,
            "gap_events": 0,
            "max_consecutive_missing": 0,
            "missing_fraction": 0.0,
            "duplicates": 0,
            "out_of_order": 0,
        }

    delta = np.diff(frames)
    missing_each = np.clip(delta - 1, 0, None)
    missing_total = int(np.sum(missing_each))
    gap_events = int(np.sum(missing_each > 0))
    expected_span = int(max(1, frames[-1] - frames[0] + 1))
    return {
        "serial_lost_frames": missing_total,
        "gap_events": gap_events,
        "max_consecutive_missing": int(np.max(missing_each)) if gap_events else 0,
        "missing_fraction": float(missing_total / expected_span),
        "duplicates": int(np.sum(delta == 0)),
        "out_of_order": int(np.sum(delta < 0)),
    }


def calculate_confidence_ellipse(cop_ap, cop_ml):
    ap_values = np.asarray(cop_ap, dtype=float)
    ml_values = np.asarray(cop_ml, dtype=float)
    valid = np.isfinite(ap_values) & np.isfinite(ml_values)
    ap = ap_values[valid]
    ml = ml_values[valid]

    if len(ap) < 10:
        return np.nan, np.nan, np.nan, np.nan

    covariance = np.cov(ml, ap)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    chi_square_95 = 5.991
    major_axis = 2.0 * np.sqrt(chi_square_95 * eigenvalues[0])
    minor_axis = 2.0 * np.sqrt(chi_square_95 * eigenvalues[1])
    area = np.pi * chi_square_95 * np.sqrt(eigenvalues[0] * eigenvalues[1])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    return float(area), float(major_axis), float(minor_axis), float(angle)


def open_forceplate_serial(config: dict[str, Any] | None = None):
    if serial is None:
        print("[ERROR] pyserial belum terpasang. Jalankan: pip install pyserial numpy scipy matplotlib")
        return None
    cfg = load_config() if config is None else config
    port = str(cfg["serial_port"])
    baud = int(cfg["baud_rate"])

    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=0.25,
            write_timeout=2.0,
        )
        # Beberapa driver Windows mendukung pembesaran buffer serial.
        # Pemanggilan ini bersifat best-effort agar tidak merusak port yang
        # tidak menyediakan set_buffer_size().
        try:
            ser.set_buffer_size(rx_size=131072, tx_size=8192)
        except (AttributeError, OSError, ValueError):
            pass
    except serial.SerialException as exc:
        print(f"[ERROR] Gagal membuka {port}: {exc}")
        return None

    time.sleep(ESP32_BOOT_WAIT_S)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print("[INIT] Mencari ESP32...")
    deadline = time.monotonic() + READY_TIMEOUT_S
    next_ping = 0.0

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_ping:
            ser.write(b"?")
            ser.flush()
            next_ping = now + 0.5

        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line.startswith("#SYSTEM_READY"):
            print("[OK] ESP32 terhubung.")
            return ser

    print(
        "[ERROR] ESP32 tidak merespons. Periksa COM, baud, kabel, firmware, "
        "dan pastikan Serial Monitor tertutup."
    )
    ser.close()
    return None


def query_firmware_metadata(ser, timeout_s: float = 3.0) -> dict[str, Any]:
    """Minta metadata firmware secara eksplisit sebelum pengukuran.

    v8.5 memakai ini untuk memastikan Python dan firmware menggunakan protokol
    streaming yang sama. Dengan demikian pengguna tidak tanpa sengaja menjalankan
    processing v8.5 bersama firmware lama yang belum memiliki CRC.
    """
    try:
        ser.reset_input_buffer()
        ser.write(b"I")
        ser.flush()
    except (OSError, getattr(serial, "SerialException", OSError)) as exc:
        return {"ok": False, "error": str(exc), "lines": []}

    lines: list[str] = []
    deadline = time.monotonic() + max(0.5, float(timeout_s))
    while time.monotonic() < deadline:
        try:
            line = ser.readline().decode("ascii", errors="ignore").strip()
        except (OSError, getattr(serial, "SerialException", OSError)) as exc:
            return {"ok": False, "error": str(exc), "lines": lines}
        if not line:
            continue
        lines.append(line)
        if line.startswith("#SYSTEM_READY"):
            break

    fw_version = next(
        (line for line in lines if line.startswith("#FW_VERSION,")),
        "",
    )
    stream_protocol = next(
        (line for line in lines if line.startswith("#STREAM_PROTOCOL,")),
        "",
    )
    adc_config = next(
        (line for line in lines if line.startswith("#ADC_CONFIG,")),
        "",
    )
    return {
        "ok": bool(fw_version),
        "error": "" if fw_version else "Metadata firmware tidak lengkap",
        "lines": lines,
        "fw_version": fw_version,
        "stream_protocol": stream_protocol,
        "adc_config": adc_config,
    }


def _parse_zero_result(line: str, config: dict[str, Any]):
    """Parse #ZERO_RESULT firmware dan konversi residual counts menjadi kg."""
    if not line.startswith("#ZERO_RESULT,"):
        return None

    parts = line.split(",")
    if len(parts) < 12:
        return None

    try:
        mean_counts = np.array([float(value) for value in parts[1:9]], dtype=float)
    except ValueError:
        return None

    metadata: dict[str, str] = {}
    for item in parts[9:]:
        if "=" in item:
            key, value = item.split("=", 1)
            metadata[key.strip().upper()] = value.strip()

    try:
        valid_frames = int(metadata.get("VALID", "0"))
        timeout_mask = int(metadata.get("TIMEOUT_MASK", "0"), 0)
        saturation_mask = int(metadata.get("SAT_MASK", "0"), 0)
    except ValueError:
        return None

    factors = np.asarray(config["counts_per_kg"], dtype=float)
    signs = np.asarray(config["channel_sign"], dtype=float)
    residual_kg = mean_counts * signs / factors

    left_kg = float(np.sum(residual_kg[LEFT_INDICES]))
    right_kg = float(np.sum(residual_kg[RIGHT_INDICES]))
    total_kg = left_kg + right_kg

    total_limit = float(config.get("zero_total_limit_kg", 0.10))
    side_limit = float(config.get("zero_side_limit_kg", 0.10))

    passed = bool(
        valid_frames > 0
        and timeout_mask == 0
        and saturation_mask == 0
        and abs(total_kg) <= total_limit
        and abs(left_kg) <= side_limit
        and abs(right_kg) <= side_limit
    )

    return {
        "passed": passed,
        "valid_frames": valid_frames,
        "timeout_mask": timeout_mask,
        "saturation_mask": saturation_mask,
        "mean_counts": mean_counts.tolist(),
        "residual_channel_kg": residual_kg.tolist(),
        "left_kg": left_kg,
        "right_kg": right_kg,
        "total_kg": total_kg,
        "total_limit_kg": total_limit,
        "side_limit_kg": side_limit,
    }


def request_zero_check(ser, config: dict[str, Any]):
    """Verifikasi zero-return tanpa mengubah tare offset.

    Temuan validasi 12 kondisi menunjukkan zero-return kadang belum memenuhi
    ±0.10 kg pada pemeriksaan pertama tetapi kembali mendekati nol setelah
    settling tambahan. Karena itu verifikasi dapat diulang sebelum tare diulang.
    """
    attempts = max(1, int(config.get("zero_check_max_attempts", 2)))
    retry_delay_s = max(0.0, float(config.get("zero_check_retry_delay_s", 5.0)))

    last_result = None
    for attempt in range(1, attempts + 1):
        ser.reset_input_buffer()
        ser.write(b"Z")
        ser.flush()

        deadline = time.monotonic() + TARE_TIMEOUT_S
        failure_message = "Timeout"

        while time.monotonic() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            if line.startswith("#ZERO_START"):
                continue

            if line.startswith("#ZERO_RESULT"):
                result = _parse_zero_result(line, config)
                if result is None:
                    failure_message = "Format #ZERO_RESULT tidak valid"
                    break

                last_result = result
                print(
                    "[ZERO] "
                    f"L={result['left_kg']:+.4f} kg | "
                    f"R={result['right_kg']:+.4f} kg | "
                    f"Total={result['total_kg']:+.4f} kg | "
                    f"valid={result['valid_frames']}"
                )

                if result["passed"]:
                    result["attempt"] = attempt
                    result["status"] = (
                        "OK_FIRST_CHECK"
                        if attempt == 1
                        else "RECOVERED_AFTER_SETTLING"
                    )
                    print(f"[OK] Zero gate: {result['status']}.")
                    return True, result

                failure_message = (
                    "Residual zero melebihi batas "
                    f"(total ±{result['total_limit_kg']:.3f} kg, "
                    f"sisi ±{result['side_limit_kg']:.3f} kg)"
                )
                break

            if line.startswith("#ZERO_FAILED") or line.startswith("#ZERO_DENIED"):
                failure_message = line
                break

        print(f"[WARNING] Zero check {attempt}/{attempts} belum lolos: {failure_message}")

        if attempt < attempts:
            print(
                f"[RECOVERY] Diamkan kedua plate {retry_delay_s:.1f} detik "
                "tanpa beban sebelum zero check berikutnya."
            )
            time.sleep(retry_delay_s)

    if last_result is not None:
        last_result["attempt"] = attempts
        last_result["status"] = "FAILED_AFTER_SETTLING"
    return False, last_result


def request_tare(ser, config: dict[str, Any]):
    """Tare robust + settling + zero-return gate.

    Setelah operator memastikan plate kosong, program memberi waktu recovery
    sebelum tare. Tare dianggap siap untuk pengukuran hanya jika firmware
    berhasil dan residual zero setelah settling juga memenuhi batas.
    """
    pre_settle_s = max(0.0, float(config.get("pre_tare_settle_s", 5.0)))
    post_settle_s = max(0.0, float(config.get("post_tare_zero_settle_s", 2.0)))

    for attempt in range(1, MAX_TARE_ATTEMPTS + 1):
        prompt = (
            "Pastikan kedua plate KOSONG dan tidak disentuh. Tekan ENTER untuk memulai recovery + TARE..."
            if attempt == 1
            else f"Tare/zero gate percobaan {attempt - 1} belum lolos. "
                 "Pastikan plate kosong lalu tekan ENTER..."
        )
        input(prompt)

        if pre_settle_s > 0.0:
            print(
                f"[RECOVERY] Menunggu {pre_settle_s:.1f} detik agar plate/load cell "
                "kembali stabil sebelum tare."
            )
            time.sleep(pre_settle_s)

        ser.reset_input_buffer()
        ser.write(b"T")
        ser.flush()

        deadline = time.monotonic() + TARE_TIMEOUT_S
        failure_message = "Timeout"
        tare_ok = False

        while time.monotonic() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("#TARE_CH"):
                print(line)
                continue
            if line.startswith("#TARE_WARNING"):
                print(f"[WARNING] {line}")
                continue
            if line.startswith("#TARE_OK"):
                tare_ok = True
                print("[OK] Tare firmware berhasil.")
                break
            if line.startswith("#TARE_FAILED"):
                failure_message = line
                break

        if not tare_ok:
            print(f"[ERROR] Tare gagal: {failure_message}")
            continue

        if post_settle_s > 0.0:
            print(
                f"[SETTLING] Menunggu {post_settle_s:.1f} detik sebelum "
                "verifikasi zero-return."
            )
            time.sleep(post_settle_s)

        zero_ok, zero_info = request_zero_check(ser, config)
        if zero_ok:
            return True, zero_info

        print(
            "[WARNING] Tare selesai tetapi zero-return belum stabil. "
            "Program akan melakukan tare ulang, bukan menyembunyikan residual."
        )

    return False, None


def wait_for_response(ser, success_prefix, failure_prefixes, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if line.startswith(success_prefix):
            return True, line
        if any(line.startswith(prefix) for prefix in failure_prefixes):
            return False, line
    return False, "Timeout"


def select_firmware_profile(ser, mode: str):
    """Pilih data-rate firmware sebelum tare.

    Standing memakai 600 SPS NORMAL untuk mempertahankan karakteristik statis.
    CMJ memakai DR_1000SPS + MODE_TURBO (= nominal 2000 SPS) untuk memperpendek
    raw separation AIN01<->AIN23. Pergantian profil selalu membatalkan tare,
    sehingga fungsi ini wajib dipanggil sebelum request_tare().
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"standing", "cmj"}:
        return False, "Mode profile tidak dikenal"

    command = b"C" if normalized_mode == "cmj" else b"G"
    ser.reset_input_buffer()
    ser.write(command)
    ser.flush()
    ok, message = wait_for_response(
        ser,
        "#PROFILE_SET",
        ("#PROFILE_DENIED", "#ERROR"),
        STREAM_TIMEOUT_S,
    )
    if not ok:
        return False, message

    expected_sps = 2000 if normalized_mode == "cmj" else 600
    expected_profile = "CMJ_2000_SPS_TURBO" if normalized_mode == "cmj" else "STANDING_600_SPS_NORMAL"
    if expected_profile not in message or f"NOMINAL_SPS={expected_sps}" not in message:
        return False, (
            "Firmware merespons profil yang tidak sesuai: " + message
        )
    return True, message


class _BufferedSerialLineReader:
    """Pembaca serial berbasis byte-buffer untuk fase streaming.

    v8.5 melakukan tiga hal:
    1) hanya mengeluarkan baris yang sudah memiliki newline lengkap;
    2) mengabaikan burst byte NUL/kontrol yang tidak membawa frame;
    3) melakukan resinkronisasi ke prefix ``@FP1,`` bila firmware CRC v8.5 dipakai.

    Byte yang berada *di dalam* frame CRC tidak dihapus secara diam-diam. Jika
    frame benar-benar korup, checksum akan gagal sehingga korupsi tetap terlihat.
    """

    def __init__(self, ser, max_partial_bytes: int = 8192):
        self.ser = ser
        self.max_partial_bytes = max(1024, int(max_partial_bytes))
        self._byte_buffer = bytearray()
        self._complete_lines: deque[bytes] = deque()
        self.bytes_received = 0
        self.read_calls = 0
        self.partial_waits = 0
        self.decode_errors = 0
        self.resync_events = 0
        self.max_buffer_bytes = 0
        self.max_pending_lines = 0
        self.noise_only_lines = 0
        self.nul_bytes_seen = 0
        self.recovered_prefixed_lines = 0
        self.recovered_legacy_nul_lines = 0

    @staticmethod
    def _legacy_ascii_candidate(raw_line: bytes) -> bytes | None:
        """Pulihkan format lama hanya bila penghapusan NUL menghasilkan CSV jelas.

        Ini hanya kompatibilitas firmware lama. Pada protokol @FP1 ber-CRC,
        byte NUL di dalam frame tidak pernah dihapus karena checksum harus menjadi
        sumber kebenaran integritas data.
        """
        if b"\x00" not in raw_line:
            return None
        candidate = raw_line.replace(b"\x00", b"").strip()
        if not candidate or candidate.count(b",") != 11:
            return None
        allowed = set(b"0123456789,+- \t")
        if any(byte not in allowed for byte in candidate):
            return None
        return candidate

    def _queue_candidate(self, raw_line: bytes) -> None:
        raw_line = raw_line.rstrip(b"\r")
        if not raw_line:
            return

        self.nul_bytes_seen += raw_line.count(b"\x00")

        stripped = raw_line.strip(b"\x00\t \r")
        if not stripped:
            self.noise_only_lines += 1
            return

        # Firmware v8.5 memakai prefix @FP1,. Bila ada noise sebelum prefix,
        # buang hanya prefix-noise tersebut lalu serahkan frame utuh ke CRC.
        prefix_index = raw_line.find(b"@FP1,")
        if prefix_index > 0:
            raw_line = raw_line[prefix_index:]
            self.recovered_prefixed_lines += 1
            self.resync_events += 1
        elif prefix_index < 0:
            legacy = self._legacy_ascii_candidate(raw_line)
            if legacy is not None:
                raw_line = legacy
                self.recovered_legacy_nul_lines += 1

        self._complete_lines.append(raw_line)

    def _extract_complete_lines(self) -> None:
        while True:
            newline_index = self._byte_buffer.find(b"\n")
            if newline_index < 0:
                break
            raw_line = bytes(self._byte_buffer[:newline_index])
            del self._byte_buffer[: newline_index + 1]
            self._queue_candidate(raw_line)

        self.max_buffer_bytes = max(self.max_buffer_bytes, len(self._byte_buffer))
        self.max_pending_lines = max(self.max_pending_lines, len(self._complete_lines))

        if len(self._byte_buffer) > self.max_partial_bytes:
            # Bila ada prefix frame baru di dalam buffer yang terlalu panjang,
            # pertahankan bagian mulai prefix terakhir. Jika tidak ada prefix,
            # kosongkan buffer sampai newline berikutnya.
            last_prefix = self._byte_buffer.rfind(b"@FP1,")
            if last_prefix >= 0:
                del self._byte_buffer[:last_prefix]
            else:
                self._byte_buffer.clear()
            self.resync_events += 1

    def read_line(self, timeout_s: float = 0.25) -> str | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))

        while True:
            if self._complete_lines:
                raw_line = self._complete_lines.popleft()
                try:
                    return raw_line.decode("ascii", errors="strict").strip()
                except UnicodeDecodeError:
                    self.decode_errors += 1
                    return raw_line.decode("ascii", errors="replace").strip()

            if time.monotonic() >= deadline:
                if self._byte_buffer:
                    self.partial_waits += 1
                return None

            try:
                waiting = int(getattr(self.ser, "in_waiting", 0) or 0)
            except (OSError, ValueError):
                waiting = 0

            request_size = min(max(waiting, 1), 65536)
            chunk = self.ser.read(request_size)
            self.read_calls += 1
            if not chunk:
                continue

            self.bytes_received += len(chunk)
            self._byte_buffer.extend(chunk)
            self._extract_complete_lines()

    def diagnostics(self) -> dict[str, int]:
        return {
            "bytes_received": int(self.bytes_received),
            "read_calls": int(self.read_calls),
            "partial_waits": int(self.partial_waits),
            "decode_errors": int(self.decode_errors),
            "resync_events": int(self.resync_events),
            "max_partial_buffer_bytes": int(self.max_buffer_bytes),
            "max_pending_lines": int(self.max_pending_lines),
            "pending_partial_bytes": int(len(self._byte_buffer)),
            "pending_complete_lines": int(len(self._complete_lines)),
            "noise_only_lines": int(self.noise_only_lines),
            "nul_bytes_seen": int(self.nul_bytes_seen),
            "recovered_prefixed_lines": int(self.recovered_prefixed_lines),
            "recovered_legacy_nul_lines": int(self.recovered_legacy_nul_lines),
        }


def _crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF."""
    crc = int(initial) & 0xFFFF
    for byte in data:
        crc ^= int(byte) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def _parse_data_line_detailed(line: str):
    """Parse frame streaming lama atau frame v8.5 @FP1 ber-CRC.

    Return ``(parsed, reason)``. ``reason`` dipakai hanya untuk diagnostik;
    tidak ada data numerik yang ditebak bila checksum gagal.
    """
    text = str(line).strip()
    if not text:
        return None, "empty"

    payload = text
    protocol = "LEGACY"
    if text.startswith("@FP1,"):
        protocol = "FP1_CRC16"
        if "*" not in text:
            return None, "framing"
        body, crc_text = text[1:].rsplit("*", 1)  # body dimulai FP1,
        if len(crc_text) != 4:
            return None, "checksum_format"
        try:
            received_crc = int(crc_text, 16)
        except ValueError:
            return None, "checksum_format"
        calculated_crc = _crc16_ccitt(body.encode("ascii", errors="strict"))
        if received_crc != calculated_crc:
            return None, "checksum"
        if not body.startswith("FP1,"):
            return None, "protocol"
        payload = body[4:]

    parts = payload.split(",")
    if len(parts) != 12:
        return None, "field_count"

    try:
        frame = int(parts[0])
        t_mux1 = int(parts[1])
        t_mux2 = int(parts[2])
        raw_counts = np.array([int(value) for value in parts[3:11]], dtype=float)
        status_mask = int(parts[11])
    except ValueError:
        return None, "numeric"

    return (frame, t_mux1, t_mux2, raw_counts, status_mask), protocol


def _parse_data_line(line: str):
    """Kompatibilitas internal: kembalikan tuple data atau None."""
    parsed, _ = _parse_data_line_detailed(line)
    return parsed


def _mode_parameters(
    mode: str,
    config: dict[str, Any],
    acquisition_profile: str = "human",
):
    """Parameter trigger dan lama perekaman.

    Faktor kalibrasi tidak berubah. Profil hanya mengatur kapan rekaman dimulai,
    berapa lama data direkam, dan cutoff gaya untuk tujuan analisis.
    """
    if mode == "standing":
        if acquisition_profile == "human":
            return {
                "record_duration": float(
                    config["human_standing_record_duration_s"]
                ),
                "stable_confirm_s": 4.0,
                "settle_hold_s": 3.0,
                "stable_total_cv_limit": 0.015,
                "stable_side_cv_limit": 0.035,
                "stable_total_slope_limit_n_s": 6.0,
                "stable_side_slope_limit_n_s": 6.0,
                "stable_cop_range_limit_m": 0.025,
                "stable_cop_slope_limit_m_s": 0.0015,
                "force_filter_hz": 5.0,
            }

        if acquisition_profile == "static_rigid":
            return {
                "record_duration": float(
                    config["static_rigid_record_duration_s"]
                ),
                "stable_confirm_s": 5.0,
                "settle_hold_s": 5.0,
                "stable_total_cv_limit": 0.008,
                "stable_side_cv_limit": 0.018,
                "stable_total_slope_limit_n_s": 2.5,
                "stable_side_slope_limit_n_s": 3.0,
                "stable_cop_range_limit_m": 0.008,
                "stable_cop_slope_limit_m_s": 0.00050,
                "force_filter_hz": 5.0,
            }

        if acquisition_profile == "static_liquid":
            # Cairan dapat tetap bergerak setelah wadah sudah diletakkan.
            # Rekaman sengaja dibuat panjang; standing.py memilih window 30 s
            # paling stabil tanpa menyembunyikan gerakan cairan dengan notch.
            return {
                "record_duration": float(
                    config["static_liquid_record_duration_s"]
                ),
                "stable_confirm_s": 3.0,
                "settle_hold_s": 2.0,
                "stable_total_cv_limit": 0.020,
                "stable_side_cv_limit": 0.045,
                "stable_total_slope_limit_n_s": 8.0,
                "stable_side_slope_limit_n_s": 10.0,
                "stable_cop_range_limit_m": 0.035,
                "stable_cop_slope_limit_m_s": 0.0030,
                "force_filter_hz": 3.0,
            }

        raise ValueError(
            "acquisition_profile harus 'human', 'static_rigid', "
            "atau 'static_liquid'."
        )

    if mode == "cmj":
        # record_duration adalah batas keselamatan maksimum. Rekaman normal
        # berhenti setelah landing DAN kondisi post-landing bilateral/CoP stabil,
        # atau setelah batas post-landing maksimum tercapai.
        return {
            "record_duration": float(config["cmj_record_duration_s"]),
            "stable_confirm_s": 1.5,
            "settle_hold_s": 1.0,
            "stable_total_cv_limit": 0.025,
            "stable_side_cv_limit": 0.060,
            "stable_total_slope_limit_n_s": 15.0,
            "stable_side_slope_limit_n_s": 18.0,
            "stable_cop_range_limit_m": 0.040,
            "stable_cop_slope_limit_m_s": 0.0040,
            "force_filter_hz": float(config["cmj_force_filter_hz"]),
            "cmj_min_prejump_s": float(config["cmj_min_prejump_s"]),
            "cmj_flight_threshold_n": float(config["cmj_flight_threshold_n"]),
            "cmj_landing_threshold_n": float(config["cmj_landing_threshold_n"]),
            "cmj_event_persistence_s": float(config["cmj_event_persistence_s"]),
            "cmj_post_search_delay_s": float(config["cmj_post_search_delay_s"]),
            "cmj_post_landing_min_s": float(config["cmj_post_landing_min_s"]),
            "cmj_post_landing_max_s": float(config["cmj_post_landing_max_s"]),
            "cmj_post_stable_window_s": float(config["cmj_post_stable_window_s"]),
            "cmj_post_total_cv_limit": float(config["cmj_post_total_cv_limit"]),
            "cmj_post_side_cv_limit": float(config["cmj_post_side_cv_limit"]),
            "cmj_post_total_slope_limit_n_s": float(config["cmj_post_total_slope_limit_n_s"]),
            "cmj_post_side_slope_limit_n_s": float(config["cmj_post_side_slope_limit_n_s"]),
            "cmj_post_cop_range_limit_m": float(config["cmj_post_cop_range_limit_m"]),
            "cmj_post_cop_slope_limit_m_s": float(config["cmj_post_cop_slope_limit_m_s"]),
            "cmj_post_stable_hold_s": float(config["cmj_post_stable_hold_s"]),
            "cmj_post_capture_tail_s": float(config["cmj_post_capture_tail_s"]),
            "cmj_primary_method": str(config["cmj_primary_method"]),
            "cmj_baseline_guard_s": float(config["cmj_baseline_guard_s"]),
            "cmj_primary_baseline_cv_limit": float(config["cmj_primary_baseline_cv_limit"]),
            "cmj_primary_baseline_slope_limit_n_s": float(config["cmj_primary_baseline_slope_limit_n_s"]),
            "cmj_impulse_height_threshold_caution_cm": float(config["cmj_impulse_height_threshold_caution_cm"]),
            "cmj_impulse_height_threshold_review_cm": float(config["cmj_impulse_height_threshold_review_cm"]),
        }

    raise ValueError("mode harus 'standing' atau 'cmj'.")

def _positive_cv(values: np.ndarray) -> float:
    """Coefficient of variation untuk gaya yang seharusnya positif."""
    array = np.asarray(values, dtype=float)
    if array.size < 2 or not np.all(np.isfinite(array)):
        return float("inf")
    mean_value = float(np.mean(array))
    if mean_value <= 0.0:
        return float("inf")
    return float(np.std(array, ddof=0) / mean_value)


def _update_stability_window(
    stable_buffer: deque[tuple[int, float, float, float, float, float]],
    timestamp_us: int,
    total_force: float,
    left_force: float,
    right_force: float,
    cop_ap: float,
    cop_ml: float,
    params: dict[str, float],
):
    """Evaluasi kestabilan gaya dan CoP pada sliding window.

    Pemeriksaan CoP mencegah proses memindahkan beban di atas plate dianggap
    stabil hanya karena total gaya kiri/kanan sudah konstan.
    """
    sample = (
        int(timestamp_us),
        float(total_force),
        float(left_force),
        float(right_force),
        float(cop_ap),
        float(cop_ml),
    )
    stable_buffer.append(sample)

    required_us = max(
        1,
        int(round(float(params["stable_confirm_s"]) * 1_000_000.0)),
    )
    if len(stable_buffer) < 3:
        return False, None

    newest_us = stable_buffer[-1][0]
    cutoff_us = newest_us - required_us
    while len(stable_buffer) > 2 and stable_buffer[1][0] <= cutoff_us:
        stable_buffer.popleft()

    if stable_buffer[-1][0] - stable_buffer[0][0] < required_us:
        return False, None

    values = np.asarray(stable_buffer, dtype=float)
    timestamps_s = (values[:, 0] - values[0, 0]) / 1_000_000.0
    total_window = values[:, 1]
    left_window = values[:, 2]
    right_window = values[:, 3]
    cop_ap_window = values[:, 4]
    cop_ml_window = values[:, 5]

    if (
        len(timestamps_s) < 3
        or timestamps_s[-1] <= 0.0
        or not np.all(np.isfinite(values))
    ):
        return False, None

    def slope(signal: np.ndarray) -> float:
        return float(np.polyfit(timestamps_s, signal, 1)[0])

    def robust_range(signal: np.ndarray) -> float:
        return float(np.percentile(signal, 95) - np.percentile(signal, 5))

    metrics = {
        "total_cv": _positive_cv(total_window),
        "left_cv": _positive_cv(left_window),
        "right_cv": _positive_cv(right_window),
        "total_slope": slope(total_window),
        "left_slope": slope(left_window),
        "right_slope": slope(right_window),
        "cop_ap_slope": slope(cop_ap_window),
        "cop_ml_slope": slope(cop_ml_window),
        "cop_ap_range": robust_range(cop_ap_window),
        "cop_ml_range": robust_range(cop_ml_window),
        "duration_s": float(timestamps_s[-1]),
        "sample_count": int(len(timestamps_s)),
        "mean_total": float(np.mean(total_window)),
        "mean_left": float(np.mean(left_window)),
        "mean_right": float(np.mean(right_window)),
        "start_timestamp_us": int(stable_buffer[0][0]),
        "end_timestamp_us": int(stable_buffer[-1][0]),
    }

    reference_force = params.get("reference_force_n")
    reference_tolerance = float(params.get("reference_force_tolerance_fraction", 0.08))
    reference_ok = bool(
        reference_force is None
        or (
            np.isfinite(float(reference_force))
            and float(reference_force) > 0.0
            and abs(metrics["mean_total"] - float(reference_force))
            <= reference_tolerance * float(reference_force)
        )
    )

    stable_ok = bool(
        reference_ok
        and metrics["total_cv"] <= float(params["stable_total_cv_limit"])
        and metrics["left_cv"] <= float(params["stable_side_cv_limit"])
        and metrics["right_cv"] <= float(params["stable_side_cv_limit"])
        and abs(metrics["total_slope"])
        <= float(params["stable_total_slope_limit_n_s"])
        and abs(metrics["left_slope"])
        <= float(params["stable_side_slope_limit_n_s"])
        and abs(metrics["right_slope"])
        <= float(params["stable_side_slope_limit_n_s"])
        and metrics["cop_ap_range"]
        <= float(params["stable_cop_range_limit_m"])
        and metrics["cop_ml_range"]
        <= float(params["stable_cop_range_limit_m"])
        and abs(metrics["cop_ap_slope"])
        <= float(params["stable_cop_slope_limit_m_s"])
        and abs(metrics["cop_ml_slope"])
        <= float(params["stable_cop_slope_limit_m_s"])
    )
    return stable_ok, metrics

def _write_measurement_csv(filename, data_store, mode):
    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["# Date", datetime.now().isoformat()])
        writer.writerow(["# Mode", mode])
        writer.writerow(["# Acquisition Profile", data_store.get("acquisition_profile", "-")])
        writer.writerow(["# Calibration counts_per_kg", *data_store["config"]["counts_per_kg"]])
        writer.writerow(["# Plate Center Distance m", data_store["config"]["plate_center_distance_m"]])
        writer.writerow(["# Force filter Hz", data_store["force_filter_hz"]])
        writer.writerow(["# MUX Alignment", "Linear interpolation to midpoint time"])
        writer.writerow(["# Firmware Profile", data_store.get("firmware_profile_message", "-")])
        writer.writerow(["# ADC Nominal SPS", data_store.get("adc_nominal_sps", "-")])
        if mode == "cmj":
            writer.writerow(["# CMJ Capture", data_store.get("cmj_capture_message", "-")])
            writer.writerow(["# CMJ Post Stable Live", data_store.get("cmj_post_stable", False)])
            writer.writerow([
                "# CMJ Live Event Times s",
                data_store.get("cmj_takeoff_live_s"),
                data_store.get("cmj_landing_confirm_live_s", data_store.get("cmj_landing_live_s")),
            ])
            writer.writerow([
                "# CMJ Live Landing Timestamp Meaning",
                "landing confirmation threshold crossing, not first contact",
            ])
            writer.writerow([
                "# CMJ Live Stable Window s",
                data_store.get("cmj_post_live_window_start_s"),
                data_store.get("cmj_post_live_window_end_s"),
            ])

        headers = [
            "Frame",
            "Time_s",
            "t_mux1_us",
            "t_mux2_us",
            *[f"{name}_raw_count" for name in SENSOR_NAMES],
            *[f"{name}_aligned_count" for name in SENSOR_NAMES],
            *[f"{name}_force_raw_N" for name in SENSOR_NAMES],
            *[f"{name}_force_filtered_N" for name in SENSOR_NAMES],
            "Fz_L_N",
            "Fz_R_N",
            "Total_GRF_N",
            "CoP_L_AP_m",
            "CoP_L_ML_m",
            "CoP_R_AP_m",
            "CoP_R_ML_m",
            "CoP_AP_m",
            "CoP_ML_m",
        ]
        writer.writerow(headers)

        for i in range(len(data_store["time"])):
            writer.writerow(
                [
                    data_store["frame"][i],
                    data_store["time"][i],
                    data_store["t_mux1_us"][i],
                    data_store["t_mux2_us"][i],
                    *data_store["raw_counts"][i],
                    *data_store["aligned_counts"][i],
                    *data_store["force_channels_raw_n"][i],
                    *data_store["force_channels_filtered_n"][i],
                    data_store["grf_l"][i],
                    data_store["grf_r"][i],
                    data_store["grf"][i],
                    data_store["cop_l_ap"][i],
                    data_store["cop_l_ml"][i],
                    data_store["cop_r_ap"][i],
                    data_store["cop_r_ml"][i],
                    data_store["cop_ap"][i],
                    data_store["cop_ml"][i],
                ]
            )


def acquire_data(
    filename_prefix,
    calculate_cop_flag=True,
    mode="standing",
    config: dict[str, Any] | None = None,
    acquisition_profile: str = "human",
):
    cfg = load_config() if config is None else config
    params = _mode_parameters(mode, cfg, acquisition_profile)
    ser = open_forceplate_serial(cfg)
    if ser is None:
        return None

    filename = (
        f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    all_received_frames: list[int] = []
    all_received_t_us: list[int] = []
    record_frames: list[int] = []
    record_t_mux1: list[int] = []
    record_t_mux2: list[int] = []
    record_raw_counts: list[np.ndarray] = []

    adc_rejected_frames = 0
    malformed_rows = 0
    adc_rejected_pre_record = 0
    adc_rejected_record = 0
    malformed_rows_pre_record = 0
    malformed_rows_record = 0
    malformed_examples: list[str] = []
    protocol_error_counts: dict[str, int] = {}
    checksum_error_rows = 0
    status_counts: dict[int, int] = {}

    baseline_values: list[float] = []
    baseline_start_us: int | None = None
    baseline_mean = np.nan
    baseline_sd = np.nan
    dynamic_start = np.nan

    contact_start_us: int | None = None
    stable_buffer: deque[tuple[int, float, float, float, float, float]] = deque()
    start_timestamp_us: int | None = None
    stability_hold_start_us: int | None = None
    trigger_stability: dict[str, float] | None = None
    last_wait_report_s = 0.0
    analysis_start_announced = False
    analysis_complete_announced = False

    # State machine khusus CMJ. Standing tidak menggunakan variabel ini,
    # sehingga antarmuka dan perilaku standing.py tetap kompatibel.
    cmj_live_state = "WAIT_FLIGHT"
    cmj_low_start_us: int | None = None
    cmj_high_start_us: int | None = None
    cmj_takeoff_live_us: int | None = None
    cmj_landing_live_us: int | None = None
    cmj_capture_complete = False
    cmj_capture_message = "-"
    cmj_post_stable = False
    cmj_post_stability: dict[str, float] | None = None
    cmj_post_buffer: deque[tuple[int, float, float, float, float, float]] = deque()
    cmj_post_stable_since_us: int | None = None
    cmj_post_tail_start_us: int | None = None
    cmj_post_live_window_start_us: int | None = None
    cmj_post_live_window_end_us: int | None = None
    cmj_event_force_buffer: deque[float] = deque(
        maxlen=max(1, int(cfg.get("cmj_live_event_median_samples", 3)))
    )
    firmware_profile_message = "-"
    firmware_metadata: dict[str, Any] = {}
    zero_gate_info: dict[str, Any] | None = None
    adc_nominal_sps = int(
        cfg["cmj_adc_nominal_sps"] if mode == "cmj"
        else cfg["standing_adc_nominal_sps"]
    )
    stream_reader: _BufferedSerialLineReader | None = None

    try:
        print("=" * 70)
        print(f"Mode: {mode.upper()} | Profile: {acquisition_profile.upper()}")

        firmware_metadata = query_firmware_metadata(ser)
        if not firmware_metadata.get("ok", False):
            print(
                "[ERROR] Metadata firmware tidak dapat diverifikasi: "
                f"{firmware_metadata.get('error', '-')}."
            )
            return None
        print(f"[FW] {firmware_metadata.get('fw_version', '-')}")
        print(f"[FW] {firmware_metadata.get('stream_protocol', 'STREAM_PROTOCOL_NOT_REPORTED')}")

        required_protocol = "#STREAM_PROTOCOL,FP1_CSV_CRC16_CCITT_FALSE"
        if bool(cfg.get("require_crc_stream_protocol", True)) and (
            required_protocol not in str(firmware_metadata.get("stream_protocol", ""))
        ):
            print(
                "[ERROR] Firmware tidak memakai protokol CRC v8.5. "
                "Flash firmware.cpp v8.5 terlebih dahulu agar Python dan ESP32 sinkron."
            )
            return None

        profile_ok, firmware_profile_message = select_firmware_profile(ser, mode)
        if not profile_ok:
            print(f"[ERROR] Gagal memilih profil ADC: {firmware_profile_message}")
            return None
        print(f"[OK] {firmware_profile_message}")

        tare_ok, zero_gate_info = request_tare(ser, cfg)
        if not tare_ok:
            return None

        ser.reset_input_buffer()
        ser.write(b"S")
        ser.flush()

        stream_ok, stream_message = wait_for_response(
            ser,
            "#STREAM_STARTED",
            ("#STREAM_DENIED", "#ERROR"),
            STREAM_TIMEOUT_S,
        )
        if not stream_ok:
            print(f"[ERROR] Streaming gagal: {stream_message}")
            return None

        # Mulai byte-buffer reader hanya setelah handshake #STREAM_STARTED.
        # Byte yang mungkin sudah menunggu di buffer driver akan langsung
        # didrain tanpa memotong baris berdasarkan timeout.
        stream_reader = _BufferedSerialLineReader(ser)

        print("[OK] Streaming dimulai. Sistem mengukur baseline kosong selama 1 detik.")
        last_rx_time = time.monotonic()
        last_valid_frame_time = time.monotonic()
        subject_wait_start = time.monotonic()
        contact_confirm_us = int(0.20 * 1_000_000)

        while True:
            now = time.monotonic()
            if now - last_rx_time > SERIAL_SILENCE_TIMEOUT_S:
                print("[ERROR] Tidak ada data serial selama 5 detik.")
                return None
            if now - last_valid_frame_time > VALID_FRAME_TIMEOUT_S:
                print("[ERROR] Tidak ada frame ADC valid selama 5 detik.")
                return None
            if (
                start_timestamp_us is None
                and now - subject_wait_start > SUBJECT_WAIT_TIMEOUT_S
            ):
                print("[ERROR] Beban/subjek tidak stabil dalam 120 detik.")
                return None

            if stream_reader is None:
                print("[ERROR INTERNAL] Buffered serial reader belum dibuat.")
                return None

            line = stream_reader.read_line(timeout_s=0.25)
            if line is None or not line:
                continue
            last_rx_time = time.monotonic()
            if line.startswith("#"):
                continue

            parsed, parse_reason = _parse_data_line_detailed(line)
            if parsed is None:
                malformed_rows += 1
                protocol_error_counts[parse_reason] = protocol_error_counts.get(parse_reason, 0) + 1
                if parse_reason == "checksum":
                    checksum_error_rows += 1
                if len(malformed_examples) < 5:
                    malformed_examples.append(f"{parse_reason}: {repr(line[:200])}")
                if start_timestamp_us is None:
                    malformed_rows_pre_record += 1
                else:
                    malformed_rows_record += 1
                continue

            frame, t_mux1, t_mux2, raw_counts, status_mask = parsed
            frame_timestamp_us = (t_mux1 + t_mux2) // 2
            all_received_frames.append(frame)
            all_received_t_us.append(frame_timestamp_us)

            if status_mask != 0:
                adc_rejected_frames += 1
                if start_timestamp_us is None:
                    adc_rejected_pre_record += 1
                else:
                    adc_rejected_record += 1
                status_counts[status_mask] = status_counts.get(status_mask, 0) + 1
                continue

            last_valid_frame_time = time.monotonic()
            preview = calculate_forces_and_cop(raw_counts, cfg)
            total_force = preview["grf"]
            left_force = preview["fz_l"]
            right_force = preview["fz_r"]

            if baseline_start_us is None:
                baseline_start_us = frame_timestamp_us

            if not np.isfinite(baseline_mean):
                baseline_values.append(total_force)
                if frame_timestamp_us - baseline_start_us < 1_000_000:
                    continue

                baseline_array = np.asarray(baseline_values, dtype=float)
                baseline_mean = float(np.median(baseline_array))
                baseline_mad = float(
                    np.median(np.abs(baseline_array - baseline_mean))
                )
                baseline_sd = max(1.4826 * baseline_mad, 1e-9)
                dynamic_start = max(50.0, baseline_mean + 8.0 * baseline_sd)
                print(
                    f"[ARMED] Naikkan subjek atau letakkan kedua beban. "
                    f"Threshold={dynamic_start:.1f} N."
                )
                continue

            both_plates_loaded = (
                total_force > dynamic_start
                and left_force > float(cfg["min_plate_force_n"])
                and right_force > float(cfg["min_plate_force_n"])
            )

            if start_timestamp_us is None:
                if not both_plates_loaded:
                    contact_start_us = None
                    stable_buffer.clear()
                    stability_hold_start_us = None
                    continue

                if contact_start_us is None:
                    contact_start_us = frame_timestamp_us
                    stable_buffer.clear()
                    stability_hold_start_us = None

                if frame_timestamp_us - contact_start_us < contact_confirm_us:
                    continue

                stable_ok, stability = _update_stability_window(
                    stable_buffer,
                    frame_timestamp_us,
                    total_force,
                    left_force,
                    right_force,
                    preview["cop_ap"],
                    preview["cop_ml"],
                    params,
                )

                if not stable_ok:
                    stability_hold_start_us = None
                    if stability is not None:
                        elapsed_wait_s = now - subject_wait_start
                        if elapsed_wait_s - last_wait_report_s >= 5.0:
                            last_wait_report_s = elapsed_wait_s
                            print(
                                "[WAIT] Kedua plate terdeteksi, tetapi gaya/CoP "
                                "belum stabil. "
                                f"CV total/L/R={stability['total_cv'] * 100:.2f}/"
                                f"{stability['left_cv'] * 100:.2f}/"
                                f"{stability['right_cv'] * 100:.2f}% | "
                                f"CoP range AP/ML={stability['cop_ap_range'] * 100:.2f}/"
                                f"{stability['cop_ml_range'] * 100:.2f} cm."
                            )
                    continue

                if stability_hold_start_us is None:
                    stability_hold_start_us = frame_timestamp_us
                    print(
                        "[SETTLING] Kondisi awal stabil. Jangan sentuh plate; "
                        f"menunggu {params['settle_hold_s']:.1f} detik."
                    )
                    continue

                hold_elapsed_s = (
                    frame_timestamp_us - stability_hold_start_us
                ) / 1_000_000.0
                if hold_elapsed_s < float(params["settle_hold_s"]):
                    continue

                trigger_stability = dict(stability)
                start_timestamp_us = frame_timestamp_us
                print(
                    "[STABLE] Gaya dan CoP stabil. "
                    f"CV total/L/R={stability['total_cv'] * 100:.3f}/"
                    f"{stability['left_cv'] * 100:.3f}/"
                    f"{stability['right_cv'] * 100:.3f}% | "
                    f"CoP range AP/ML={stability['cop_ap_range'] * 100:.3f}/"
                    f"{stability['cop_ml_range'] * 100:.3f} cm."
                )
                print("[RECORD] Perekaman dimulai.")
                if mode == "cmj":
                    print(
                        "[ACTION] Berdiri tenang minimal "
                        f"{params['cmj_min_prejump_s']:.1f} detik, lalu lakukan "
                        "SATU CMJ. Setelah landing, tetap diam sampai bunyi akhir."
                    )

            # Defensive guard. Secara logika bagian ini hanya tercapai setelah
            # start_timestamp_us terisi, tetapi jangan gunakan assert pada runtime.
            if start_timestamp_us is None:
                print(
                    "[ERROR INTERNAL] Perekaman belum memiliki timestamp awal. "
                    "Akuisisi dibatalkan dengan aman."
                )
                return None

            elapsed = (frame_timestamp_us - start_timestamp_us) / 1_000_000.0
            record_frames.append(frame)
            record_t_mux1.append(t_mux1)
            record_t_mux2.append(t_mux2)
            record_raw_counts.append(raw_counts)

            if mode == "standing" and acquisition_profile == "human":
                analysis_start_s = float(cfg["human_analysis_start_s"])
                analysis_end_s = (
                    analysis_start_s
                    + float(cfg["human_analysis_duration_s"])
                )
                if not analysis_start_announced and elapsed >= analysis_start_s:
                    analysis_start_announced = True
                    print(
                        "[ANALYSIS] Window quiet standing 30 detik dimulai. "
                        "Subjek tetap melihat target dan tidak berbicara."
                    )
                if (
                    not analysis_complete_announced
                    and elapsed >= analysis_end_s
                ):
                    analysis_complete_announced = True
                    print(
                        "[POST-MONITOR] Window utama selesai. Tetap diam "
                        f"{float(cfg['human_post_monitor_s']):.0f} detik sampai "
                        "bunyi akhir agar gerakan pasca-window dapat dideteksi."
                    )

            if mode == "cmj":
                cmj_event_force_buffer.append(float(total_force))
                event_force_live = float(np.median(np.asarray(cmj_event_force_buffer)))
                persistence_us = max(1, int(round(
                    float(params["cmj_event_persistence_s"]) * 1_000_000.0
                )))
                min_prejump_us = max(0, int(round(
                    float(params["cmj_min_prejump_s"]) * 1_000_000.0
                )))

                if cmj_live_state == "WAIT_FLIGHT":
                    armed_for_jump = (
                        frame_timestamp_us - start_timestamp_us >= min_prejump_us
                    )
                    if (
                        armed_for_jump
                        and event_force_live < float(params["cmj_flight_threshold_n"])
                    ):
                        if cmj_low_start_us is None:
                            cmj_low_start_us = frame_timestamp_us
                        elif frame_timestamp_us - cmj_low_start_us >= persistence_us:
                            cmj_takeoff_live_us = cmj_low_start_us
                            cmj_live_state = "IN_FLIGHT"
                            cmj_high_start_us = None
                            print("[EVENT] Flight terdeteksi. Menunggu landing...")
                    else:
                        cmj_low_start_us = None

                elif cmj_live_state == "IN_FLIGHT":
                    if event_force_live > float(params["cmj_landing_threshold_n"]):
                        if cmj_high_start_us is None:
                            cmj_high_start_us = frame_timestamp_us
                        elif frame_timestamp_us - cmj_high_start_us >= persistence_us:
                            cmj_landing_live_us = cmj_high_start_us
                            cmj_live_state = "POST_LANDING"
                            cmj_post_buffer.clear()
                            cmj_post_stable = False
                            cmj_post_stability = None
                            cmj_post_stable_since_us = None
                            cmj_post_tail_start_us = None
                            cmj_post_live_window_start_us = None
                            cmj_post_live_window_end_us = None
                            print(
                                "[EVENT] Landing terdeteksi. Tetap diam hingga "
                                "gaya kiri-kanan dan CoP kembali stabil "
                                f"(minimal {params['cmj_post_landing_min_s']:.1f} detik)."
                            )
                    else:
                        cmj_high_start_us = None

                elif cmj_live_state == "POST_LANDING":
                    if cmj_landing_live_us is None:
                        cmj_capture_complete = False
                        cmj_capture_message = "internal_missing_landing_timestamp"
                        print("[ERROR INTERNAL] Timestamp landing hilang.")
                        break

                    post_elapsed_s = (
                        frame_timestamp_us - cmj_landing_live_us
                    ) / 1_000_000.0
                    search_delay_s = float(params["cmj_post_search_delay_s"])

                    if post_elapsed_s >= search_delay_s:
                        post_params = {
                            "stable_confirm_s": float(params["cmj_post_stable_window_s"]),
                            "stable_total_cv_limit": float(params["cmj_post_total_cv_limit"]),
                            "stable_side_cv_limit": float(params["cmj_post_side_cv_limit"]),
                            "stable_total_slope_limit_n_s": float(params["cmj_post_total_slope_limit_n_s"]),
                            "stable_side_slope_limit_n_s": float(params["cmj_post_side_slope_limit_n_s"]),
                            "stable_cop_range_limit_m": float(params["cmj_post_cop_range_limit_m"]),
                            "stable_cop_slope_limit_m_s": float(params["cmj_post_cop_slope_limit_m_s"]),
                            "reference_force_n": (
                                float(trigger_stability["mean_total"])
                                if trigger_stability is not None
                                else None
                            ),
                            "reference_force_tolerance_fraction": 0.08,
                        }
                        stable_now, stability_now = _update_stability_window(
                            cmj_post_buffer,
                            frame_timestamp_us,
                            total_force,
                            left_force,
                            right_force,
                            preview["cop_ap"],
                            preview["cop_ml"],
                            post_params,
                        )
                    else:
                        cmj_post_buffer.clear()
                        stable_now = False
                        stability_now = None

                    if stable_now and stability_now is not None:
                        cmj_post_stability = dict(stability_now)
                        if cmj_post_stable_since_us is None:
                            cmj_post_stable_since_us = frame_timestamp_us
                            print(
                                "[POST-STABLE] Kandidat window stabil ditemukan. "
                                "Menunggu konfirmasi kontinu dan tail capture."
                            )

                        stable_hold_s = (
                            frame_timestamp_us - cmj_post_stable_since_us
                        ) / 1_000_000.0
                        if (
                            post_elapsed_s >= float(params["cmj_post_landing_min_s"])
                            and stable_hold_s >= float(params["cmj_post_stable_hold_s"])
                            and cmj_post_tail_start_us is None
                        ):
                            cmj_post_stable = True
                            cmj_post_tail_start_us = frame_timestamp_us
                            cmj_post_live_window_start_us = int(
                                stability_now["start_timestamp_us"]
                            )
                            cmj_post_live_window_end_us = int(
                                stability_now["end_timestamp_us"]
                            )
                            print(
                                "[POST-CONFIRMED] Stabilitas bilateral/CoP terkonfirmasi. "
                                f"Merekam tail {params['cmj_post_capture_tail_s']:.2f} detik "
                                "agar analisis offline tidak terpotong di tepi data."
                            )
                    else:
                        cmj_post_stable_since_us = None
                        if cmj_post_tail_start_us is None:
                            cmj_post_stable = False

                    if cmj_post_tail_start_us is not None:
                        tail_elapsed_s = (
                            frame_timestamp_us - cmj_post_tail_start_us
                        ) / 1_000_000.0
                        if tail_elapsed_s >= float(params["cmj_post_capture_tail_s"]):
                            cmj_capture_complete = True
                            cmj_capture_message = (
                                "landing_and_bilateral_stability_confirmed_with_tail"
                            )
                            print("\a", end="", flush=True)
                            print(
                                "[SELESAI] Post-landing stabil telah terkonfirmasi "
                                "dan tail capture selesai. Subjek boleh turun dari plate."
                            )
                            break

                    if post_elapsed_s >= float(params["cmj_post_landing_max_s"]):
                        cmj_capture_complete = True
                        cmj_capture_message = "landing_complete_but_post_stability_not_reached"
                        print("\a", end="", flush=True)
                        print(
                            "[SELESAI/REVIEW] Landing terekam, tetapi kestabilan "
                            "post-landing belum tercapai sampai batas maksimum."
                        )
                        break

                if elapsed >= params["record_duration"]:
                    cmj_capture_complete = False
                    cmj_capture_message = (
                        "maximum_duration_reached_before_complete_landing_capture"
                    )
                    print("\a", end="", flush=True)
                    print(
                        "[SELESAI/REVIEW] Batas maksimum CMJ "
                        f"{params['record_duration']:.0f} detik tercapai. "
                        "Data tetap disimpan, tetapi trial harus direview."
                    )
                    break
            elif elapsed >= params["record_duration"]:
                print("\a", end="", flush=True)
                print(
                    f"[SELESAI] Durasi {mode.upper()} "
                    f"{params['record_duration']:.0f} detik selesai. "
                    "Subjek boleh bergerak atau turun dari plate."
                )
                break

    except KeyboardInterrupt:
        print("[SELESAI] Dihentikan oleh pengguna.")
    except Exception as exc:
        serial_exception = getattr(serial, "SerialException", ())
        if serial_exception and isinstance(exc, serial_exception):
            print(f"[ERROR] Gangguan serial: {exc}")
        else:
            print(
                f"[ERROR INTERNAL] Akuisisi dihentikan dengan aman: "
                f"{type(exc).__name__}: {exc}"
            )
        return None
    finally:
        try:
            if ser.is_open:
                ser.write(b"X")
                ser.flush()
                time.sleep(0.05)
        except serial.SerialException:
            pass
        if ser.is_open:
            ser.close()

    if len(record_frames) < 3:
        print("[ERROR] Data valid tidak mencukupi.")
        return None

    record_frames_arr = np.asarray(record_frames, dtype=np.int64)
    record_t_mux1_arr = np.asarray(record_t_mux1, dtype=np.int64)
    record_t_mux2_arr = np.asarray(record_t_mux2, dtype=np.int64)
    record_raw_counts_arr = np.asarray(record_raw_counts, dtype=float)

    common_time_us, aligned_counts, valid_alignment = align_mux_channels(
        record_t_mux1_arr,
        record_t_mux2_arr,
        record_raw_counts_arr,
    )

    alignment_coverage = (
        float(np.mean(valid_alignment))
        if len(valid_alignment) > 0
        else 0.0
    )
    minimum_alignment_coverage = float(
        cfg.get("mux_alignment_min_coverage", 0.98)
    )
    if alignment_coverage < minimum_alignment_coverage:
        print(
            "[WARNING] Cakupan MUX alignment hanya "
            f"{alignment_coverage * 100:.2f}% "
            f"(< {minimum_alignment_coverage * 100:.2f}%)."
        )

    frames_aligned = record_frames_arr[valid_alignment]
    t1_aligned = record_t_mux1_arr[valid_alignment]
    t2_aligned = record_t_mux2_arr[valid_alignment]
    raw_aligned_rows = record_raw_counts_arr[valid_alignment]
    relative_time = (common_time_us - common_time_us[0]) / 1_000_000.0
    common_time_start_us = float(common_time_us[0])
    cmj_post_live_window_start_s = (
        (float(cmj_post_live_window_start_us) - common_time_start_us) / 1_000_000.0
        if cmj_post_live_window_start_us is not None
        else None
    )
    cmj_post_live_window_end_s = (
        (float(cmj_post_live_window_end_us) - common_time_start_us) / 1_000_000.0
        if cmj_post_live_window_end_us is not None
        else None
    )
    cmj_takeoff_live_s = (
        (float(cmj_takeoff_live_us) - common_time_start_us) / 1_000_000.0
        if cmj_takeoff_live_us is not None
        else None
    )
    cmj_landing_live_s = (
        (float(cmj_landing_live_us) - common_time_start_us) / 1_000_000.0
        if cmj_landing_live_us is not None
        else None
    )

    sampling = get_sampling_diagnostics(frames_aligned, relative_time)
    fs = float(sampling["fs"])
    if not np.isfinite(fs) or fs <= 0:
        print("[ERROR] Sampling rate hasil akuisisi tidak valid.")
        return None

    signals = calculate_forces_and_cop_batch(
        aligned_counts,
        cfg,
        fs=fs,
        force_cutoff_hz=float(params["force_filter_hz"]),
    )

    serial_transport_diag = (
        stream_reader.diagnostics()
        if stream_reader is not None
        else {
            "bytes_received": 0,
            "read_calls": 0,
            "partial_waits": 0,
            "decode_errors": 0,
            "resync_events": 0,
            "max_partial_buffer_bytes": 0,
            "max_pending_lines": 0,
            "pending_partial_bytes": 0,
            "pending_complete_lines": 0,
            "noise_only_lines": 0,
            "nul_bytes_seen": 0,
            "recovered_prefixed_lines": 0,
            "recovered_legacy_nul_lines": 0,
        }
    )

    data_store = {
        "frame": frames_aligned.tolist(),
        "time": relative_time.tolist(),
        "t_mux1_us": t1_aligned.tolist(),
        "t_mux2_us": t2_aligned.tolist(),
        "raw_counts": raw_aligned_rows.tolist(),
        "aligned_counts": aligned_counts.tolist(),
        "force_channels_raw_n": signals["force_channels_raw_n"].tolist(),
        "force_channels_filtered_n": signals[
            "force_channels_filtered_n"
        ].tolist(),
        "grf_l": signals["fz_l"].tolist(),
        "grf_r": signals["fz_r"].tolist(),
        "grf": signals["grf"].tolist(),
        "cop_l_ap": signals["cop_l_ap"].tolist(),
        "cop_l_ml": signals["cop_l_ml"].tolist(),
        "cop_r_ap": signals["cop_r_ap"].tolist(),
        "cop_r_ml": signals["cop_r_ml"].tolist(),
        "cop_ap": signals["cop_ap"].tolist(),
        "cop_ml": signals["cop_ml"].tolist(),
        "mux_skew": (t2_aligned - t1_aligned).astype(float).tolist(),
        "mux_alignment_applied": True,
        "mux_alignment_method": str(cfg.get("mux_alignment_method", "linear_to_group_midpoint")),
        "mux_alignment_coverage": alignment_coverage,
        "zero_gate_info": zero_gate_info,
        "calibration_version": str(cfg.get("calibration_version", CALIBRATION_VERSION)),
        "all_received_frames": all_received_frames,
        "all_received_t_us": all_received_t_us,
        "baseline_mean": baseline_mean,
        "baseline_sd": baseline_sd,
        "dynamic_start": dynamic_start,
        "adc_rejected_frames": adc_rejected_frames,
        "malformed_rows": malformed_rows,
        "adc_rejected_pre_record": adc_rejected_pre_record,
        "adc_rejected_record": adc_rejected_record,
        "malformed_rows_pre_record": malformed_rows_pre_record,
        "malformed_rows_record": malformed_rows_record,
        "malformed_examples": malformed_examples,
        "protocol_error_counts": protocol_error_counts,
        "checksum_error_rows": checksum_error_rows,
        "serial_transport_diag": serial_transport_diag,
        "status_counts": status_counts,
        "force_filter_hz": float(params["force_filter_hz"]),
        "trigger_stability": trigger_stability,
        "settle_hold_s": float(params["settle_hold_s"]),
        "acquisition_profile": acquisition_profile,
        "config": cfg,
        "filename": filename,
        "software_version": SOFTWARE_VERSION,
        "serial_baud": int(cfg["baud_rate"]),
        "firmware_profile_message": firmware_profile_message,
        "firmware_metadata": firmware_metadata,
        "firmware_version_message": firmware_metadata.get("fw_version", "-"),
        "stream_protocol_message": firmware_metadata.get("stream_protocol", "-"),
        "adc_nominal_sps": adc_nominal_sps,
        "record_duration_s": float(params["record_duration"]),
        "cmj_live_state": cmj_live_state,
        "cmj_takeoff_live_us": cmj_takeoff_live_us,
        "cmj_landing_live_us": cmj_landing_live_us,
        "cmj_takeoff_live_s": cmj_takeoff_live_s,
        # Timestamp live landing berasal dari awal persistence di atas
        # landing-confirm threshold. Alias eksplisit mencegah perbandingan
        # keliru terhadap first-contact threshold pada analisis offline.
        "cmj_landing_live_s": cmj_landing_live_s,
        "cmj_landing_confirm_live_s": cmj_landing_live_s,
        "cmj_capture_complete": cmj_capture_complete,
        "cmj_capture_message": cmj_capture_message,
        "cmj_post_stable": cmj_post_stable,
        "cmj_post_stability": cmj_post_stability,
        "cmj_post_live_window_start_s": cmj_post_live_window_start_s,
        "cmj_post_live_window_end_s": cmj_post_live_window_end_s,
        "cmj_post_stable_hold_s": float(params.get("cmj_post_stable_hold_s", 0.0)),
        "cmj_post_capture_tail_s": float(params.get("cmj_post_capture_tail_s", 0.0)),
        "cmj_post_landing_min_s": float(params.get("cmj_post_landing_min_s", 0.0)),
        "cmj_post_landing_max_s": float(params.get("cmj_post_landing_max_s", 0.0)),
        # Alias dipertahankan untuk kompatibilitas pembaca file lama.
        "cmj_post_landing_s": float(params.get("cmj_post_landing_min_s", 0.0)),
        "human_analysis_start_s": float(cfg["human_analysis_start_s"]),
        "human_analysis_duration_s": float(cfg["human_analysis_duration_s"]),
        "human_analysis_end_s": (
            float(cfg["human_analysis_start_s"])
            + float(cfg["human_analysis_duration_s"])
        ),
        "human_post_monitor_s": float(cfg["human_post_monitor_s"]),
    }

    if not calculate_cop_flag:
        # Tetap disimpan di CSV untuk audit, tetapi caller boleh mengabaikan.
        pass

    _write_measurement_csv(filename, data_store, mode)
    print(f"[OK] Data tersimpan: {filename}")
    return data_store


