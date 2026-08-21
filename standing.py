from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import welch
from scipy.stats import gaussian_kde

import processing as prc


SOFTWARE_VERSION = "8.5_STANDING_CRC_FRAMED_GAP_ROBUST"
UI_VERSION = "7.0_RESPONSIVE_CORE_NOTE_POLICY"
ANALYSIS_DURATION_S = 30.0
MAX_SHORT_NAN_GAP_S = 0.05
MAX_HAMPEL_REPLACEMENT_RATIO = 0.01
ENDPOINT_MEDIAN_WINDOW_S = 1.0
ROLLING_STEP_WINDOW_S = 0.50
NEAR_LIMIT_FRACTION = 0.80


@dataclass(frozen=True)
class Profile:
    name: str
    label: str
    acquisition_profile: str
    use_best_window: bool
    minimum_window_start_s: float
    analysis_cutoff_hz: float
    centering_tolerance_m: float
    thresholds: dict[str, float]


def profile_definition(name: str) -> Profile:
    if name == "human":
        return Profile(
            name="human",
            label="QUIET STANDING MANUSIA",
            acquisition_profile="human",
            use_best_window=False,
            minimum_window_start_s=7.0,
            analysis_cutoff_hz=5.0,
            centering_tolerance_m=np.nan,
            thresholds={
                "total_cv": 0.020,
                "side_cv": 0.050,
                "drift_m_s": 0.00050,
                "endpoint_m": 0.0200,
                "sd_m": 0.0100,
                "robust_range_m": 0.0500,
                "mean_velocity_m_s": 0.0200,
                "block_step_m": 0.0120,
                "hard_block_step_m": 0.0250,
                "hard_range_m": 0.0800,
                "mass_error_percent": 3.0,
                "narrowband_ratio": 0.55,
                "narrowband_min_hz": 1.5,
                "narrowband_min_sd_m": 0.0010,
                "lowfreq_ratio": 0.75,
                "lowfreq_max_hz": 1.0,
            },
        )

    if name == "static_rigid":
        return Profile(
            name="static_rigid",
            label="VALIDASI BEBAN STATIS KAKU",
            acquisition_profile="static_rigid",
            use_best_window=True,
            minimum_window_start_s=5.0,
            analysis_cutoff_hz=5.0,
            centering_tolerance_m=0.0050,
            thresholds={
                "total_cv": 0.0050,
                "side_cv": 0.0150,
                "drift_m_s": 0.00010,
                "endpoint_m": 0.0015,
                "sd_m": 0.0008,
                "robust_range_m": 0.0050,
                "mean_velocity_m_s": 0.0030,
                "block_step_m": 0.0020,
                "mass_error_percent": 2.0,
                "side_mass_error_percent": 3.0,
                "cop_error_m": 0.0050,
                "narrowband_ratio": 0.25,
                "narrowband_min_hz": 1.0,
                "narrowband_min_sd_m": 0.00030,
            },
        )

    if name == "static_liquid":
        return Profile(
            name="static_liquid",
            label="VALIDASI BEBAN STATIS BERISI CAIRAN",
            acquisition_profile="static_liquid",
            use_best_window=True,
            minimum_window_start_s=12.0,
            # 3 Hz tetap mempertahankan gerakan cairan 2.1 Hz, tetapi
            # mengurangi kontribusi noise frekuensi tinggi pada path length.
            analysis_cutoff_hz=3.0,
            centering_tolerance_m=0.0100,
            thresholds={
                "total_cv": 0.0080,
                "side_cv": 0.0200,
                "drift_m_s": 0.00020,
                "endpoint_m": 0.0030,
                "sd_m": 0.0015,
                "robust_range_m": 0.0080,
                "mean_velocity_m_s": 0.0050,
                "block_step_m": 0.0035,
                "mass_error_percent": 2.0,
                "side_mass_error_percent": 3.0,
                "narrowband_ratio": 0.35,
                "narrowband_min_hz": 0.8,
                "narrowband_min_sd_m": 0.00040,
            },
        )

    raise ValueError(f"Profil tidak dikenal: {name}")


def ask_optional_positive_float(prompt: str) -> float | None:
    text = input(prompt).strip().replace(",", ".")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError("Nilai massa harus berupa angka positif.") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Nilai massa harus berupa angka positif.")
    return value


def ask_profile() -> tuple[Profile, dict[str, float | str | None]]:
    print("Pilih profil pengukuran:")
    print("  1 = Quiet standing manusia")
    print("  2 = Validasi beban statis kaku/padat")
    print("  3 = Validasi beban statis berisi cairan (galon)")
    selection = input("Pilihan [Enter = 1]: ").strip()

    metadata: dict[str, float | str | None] = {
        "expected_human_mass_kg": None,
        "expected_left_mass_kg": None,
        "expected_right_mass_kg": None,
        "vision_condition": None,
        "stance_width_cm": None,
        "mass_reference_quality": "unknown",
    }

    if selection in ("", "1"):
        profile = profile_definition("human")
        metadata["expected_human_mass_kg"] = ask_optional_positive_float(
            "Massa referensi subjek (kg) [Enter jika tidak diketahui]: "
        )
        if metadata["expected_human_mass_kg"] is not None:
            source = input(
                "Sumber massa [1=diukur langsung, 2=perkiraan, Enter=2]: "
            ).strip().lower()
            if source in {"1", "ukur", "diukur", "measured"}:
                metadata["mass_reference_quality"] = "measured"
            elif source in {"", "2", "kira", "perkiraan", "estimated"}:
                metadata["mass_reference_quality"] = "estimated"
            else:
                raise ValueError("Sumber massa hanya 1 atau 2.")
        vision = input(
            "Kondisi penglihatan [Enter=eyes open / ketik closed]: "
        ).strip().lower()
        metadata["vision_condition"] = (
            "eyes_closed" if vision in {"closed", "close", "tutup", "tertutup"}
            else "eyes_open"
        )
        metadata["stance_width_cm"] = ask_optional_positive_float(
            "Jarak antar pusat tumit (cm) [Enter jika tidak diukur]: "
        )
        return profile, metadata

    if selection == "2":
        profile = profile_definition("static_rigid")
    elif selection == "3":
        profile = profile_definition("static_liquid")
    else:
        raise ValueError("Pilihan hanya 1, 2, atau 3.")

    left_mass = ask_optional_positive_float(
        "Massa referensi plate kiri (kg) [Enter jika tidak diketahui]: "
    )
    right_mass = ask_optional_positive_float(
        "Massa referensi plate kanan (kg) [Enter jika tidak diketahui]: "
    )
    if (left_mass is None) != (right_mass is None):
        raise ValueError("Masukkan kedua massa referensi atau kosongkan keduanya.")
    metadata["expected_left_mass_kg"] = left_mass
    metadata["expected_right_mass_kg"] = right_mass
    return profile, metadata


def interpolate_short_nan_gaps(data, time_arr, max_gap_s=MAX_SHORT_NAN_GAP_S):
    values = np.asarray(data, dtype=float).copy()
    times = np.asarray(time_arr, dtype=float)
    if values.shape != times.shape:
        raise ValueError("Data dan waktu tidak sejajar.")

    valid = np.isfinite(values)
    if np.sum(valid) < 2:
        raise ValueError("Data valid terlalu sedikit.")

    invalid_indices = np.where(~valid)[0]
    if len(invalid_indices) == 0:
        return values

    groups = np.split(
        invalid_indices,
        np.where(np.diff(invalid_indices) > 1)[0] + 1,
    )
    for group in groups:
        if len(group) == 0:
            continue
        left = int(group[0] - 1)
        right = int(group[-1] + 1)
        if left < 0 or right >= len(values):
            raise ValueError("NaN berada di ujung rekaman.")
        gap_duration = float(times[right] - times[left])
        if gap_duration > max_gap_s:
            raise ValueError(f"Gap NaN terlalu panjang: {gap_duration:.3f} s.")
        values[group] = np.interp(
            times[group],
            [times[left], times[right]],
            [values[left], values[right]],
        )
    return values


def hampel_filter(data, half_window=5, n_sigmas=3.0):
    values = np.asarray(data, dtype=float)
    filtered = values.copy()
    replaced = 0
    if half_window < 1:
        return filtered, replaced

    for index in range(half_window, len(values) - half_window):
        window = values[index - half_window : index + half_window + 1]
        median = float(np.nanmedian(window))
        mad = float(np.nanmedian(np.abs(window - median)))
        if not np.isfinite(mad) or mad < 1e-12:
            continue
        threshold = n_sigmas * 1.4826 * mad
        if abs(values[index] - median) > threshold:
            filtered[index] = median
            replaced += 1
    return filtered, replaced


def coefficient_of_variation(values) -> float:
    array = np.asarray(values, dtype=float)
    mean_value = float(np.mean(array))
    if not np.isfinite(mean_value) or abs(mean_value) < 1e-12:
        return np.inf
    return float(np.std(array, ddof=0) / abs(mean_value))


def median_endpoint_shift(values, fs, window_s=ENDPOINT_MEDIAN_WINDOW_S):
    array = np.asarray(values, dtype=float)
    count = max(3, int(round(fs * window_s)))
    count = min(count, max(1, len(array) // 3))
    first = float(np.median(array[:count]))
    last = float(np.median(array[-count:]))
    return last - first, first, last


def robust_range(values, lower=1.0, upper=99.0) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.percentile(array, upper) - np.percentile(array, lower))


def block_median_series(values, time_arr, block_s=ROLLING_STEP_WINDOW_S):
    array = np.asarray(values, dtype=float)
    times = np.asarray(time_arr, dtype=float)
    if len(array) != len(times) or len(array) < 4:
        return np.array([], dtype=float), np.array([], dtype=float)
    start_time = float(times[0])
    end_time = float(times[-1])
    edges = np.arange(start_time, end_time + block_s, block_s)
    centers: list[float] = []
    medians: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (times >= start) & (times < end)
        if np.any(mask):
            centers.append((start + end) / 2.0)
            medians.append(float(np.median(array[mask])))
    return np.asarray(centers, dtype=float), np.asarray(medians, dtype=float)


def maximum_block_step_details(values, time_arr, block_s=ROLLING_STEP_WINDOW_S):
    centers, medians = block_median_series(values, time_arr, block_s)
    if len(medians) < 2:
        return {
            "magnitude": np.nan, "time_s": np.nan,
            "before": np.nan, "after": np.nan,
        }
    differences = np.diff(medians)
    index = int(np.argmax(np.abs(differences)))
    return {
        "magnitude": float(abs(differences[index])),
        "time_s": float((centers[index] + centers[index + 1]) / 2.0),
        "before": float(medians[index]),
        "after": float(medians[index + 1]),
    }


def maximum_block_step(values, time_arr, block_s=ROLLING_STEP_WINDOW_S):
    return float(maximum_block_step_details(values, time_arr, block_s)["magnitude"])


def summarize_time_blocks(time_arr, cop_ap, cop_ml, block_s=5.0):
    """Ringkasan mean dan SD CoP per blok waktu berurutan."""
    times = np.asarray(time_arr, dtype=float)
    ap = np.asarray(cop_ap, dtype=float)
    ml = np.asarray(cop_ml, dtype=float)
    if not (len(times) == len(ap) == len(ml)) or len(times) < 3:
        return []

    relative = times - times[0]
    duration = float(relative[-1])
    blocks = []
    start = 0.0
    while start < duration - 1e-9:
        end = min(start + block_s, duration + 1e-9)
        mask = (relative >= start) & (relative < end)
        if np.count_nonzero(mask) >= 3:
            blocks.append({
                "start_s": float(start),
                "end_s": float(min(start + block_s, duration)),
                "mean_ap": float(np.mean(ap[mask])),
                "mean_ml": float(np.mean(ml[mask])),
                "sd_ap": float(np.std(ap[mask], ddof=0)),
                "sd_ml": float(np.std(ml[mask], ddof=0)),
            })
        start += block_s
    return blocks


def detect_end_anticipation(blocks):
    """Deteksi informasional gerakan yang membesar pada 5 detik terakhir."""
    if len(blocks) < 3:
        return {"detected": False, "axis": "-", "reason": "insufficient"}

    last = blocks[-1]
    previous = blocks[-2]
    earlier = blocks[:-1]
    typical_sd_ap = float(np.median([item["sd_ap"] for item in earlier]))
    typical_sd_ml = float(np.median([item["sd_ml"] for item in earlier]))
    mean_jump_ap = abs(last["mean_ap"] - previous["mean_ap"])
    mean_jump_ml = abs(last["mean_ml"] - previous["mean_ml"])
    sd_ratio_ap = last["sd_ap"] / max(typical_sd_ap, 1e-9)
    sd_ratio_ml = last["sd_ml"] / max(typical_sd_ml, 1e-9)

    ap_detected = mean_jump_ap >= 0.008 or (
        last["sd_ap"] >= 0.004 and sd_ratio_ap >= 1.75
    )
    ml_detected = mean_jump_ml >= 0.006 or (
        last["sd_ml"] >= 0.003 and sd_ratio_ml >= 1.75
    )
    if not (ap_detected or ml_detected):
        return {
            "detected": False,
            "axis": "-",
            "mean_jump_ap": mean_jump_ap,
            "mean_jump_ml": mean_jump_ml,
            "sd_ratio_ap": sd_ratio_ap,
            "sd_ratio_ml": sd_ratio_ml,
        }

    axis = "AP" if (mean_jump_ap + last["sd_ap"]) >= (mean_jump_ml + last["sd_ml"]) else "ML"
    return {
        "detected": True,
        "axis": axis,
        "mean_jump_ap": mean_jump_ap,
        "mean_jump_ml": mean_jump_ml,
        "sd_ratio_ap": sd_ratio_ap,
        "sd_ratio_ml": sd_ratio_ml,
    }


def detect_post_window_motion(
    time_arr, cop_ap, cop_ml, end_idx, fs,
    threshold_ap_m=0.010, threshold_ml_m=0.006,
):
    """Deteksi gerakan setelah window utama; tidak mengubah metrik utama."""
    times = np.asarray(time_arr, dtype=float)
    ap = np.asarray(cop_ap, dtype=float)
    ml = np.asarray(cop_ml, dtype=float)
    if not (len(times) == len(ap) == len(ml)) or end_idx >= len(times) - 3:
        return {
            "available": False, "detected": False, "time_s": np.nan,
            "shift_ap": np.nan, "shift_ml": np.nan,
        }

    baseline_count = max(3, int(round(fs * 1.0)))
    baseline_start = max(0, end_idx - baseline_count)
    baseline_ap = float(np.median(ap[baseline_start:end_idx]))
    baseline_ml = float(np.median(ml[baseline_start:end_idx]))

    post_time = times[end_idx:]
    post_ap = ap[end_idx:]
    post_ml = ml[end_idx:]
    centers_ap, medians_ap = block_median_series(post_ap, post_time, 0.25)
    centers_ml, medians_ml = block_median_series(post_ml, post_time, 0.25)
    count = min(len(medians_ap), len(medians_ml))
    if count == 0:
        return {
            "available": False, "detected": False, "time_s": np.nan,
            "shift_ap": np.nan, "shift_ml": np.nan,
        }

    shifts_ap = medians_ap[:count] - baseline_ap
    shifts_ml = medians_ml[:count] - baseline_ml
    normalized = np.maximum(
        np.abs(shifts_ap) / max(threshold_ap_m, 1e-9),
        np.abs(shifts_ml) / max(threshold_ml_m, 1e-9),
    )
    index = int(np.argmax(normalized))
    detected = bool(
        abs(shifts_ap[index]) >= threshold_ap_m
        or abs(shifts_ml[index]) >= threshold_ml_m
    )
    return {
        "available": True,
        "detected": detected,
        "time_s": float(centers_ap[index] - times[0]),
        "shift_ap": float(shifts_ap[index]),
        "shift_ml": float(shifts_ml[index]),
        "baseline_ap": baseline_ap,
        "baseline_ml": baseline_ml,
    }


def robust_block_slope(values, time_arr, block_s=1.0) -> float:
    centers, medians = block_median_series(values, time_arr, block_s)
    if len(medians) < 3 or centers[-1] <= centers[0]:
        return np.nan
    return float(np.polyfit(centers - centers[0], medians, 1)[0])


def signal_slope(values, time_arr) -> float:
    # Slope dihitung dari median per 1 detik agar tidak didominasi satu gerakan
    # mendadak atau satu sampel ekstrem.
    return robust_block_slope(values, time_arr, block_s=1.0)


def detrend_with_block_slope(values, time_arr):
    array = np.asarray(values, dtype=float)
    times = np.asarray(time_arr, dtype=float)
    slope = robust_block_slope(array, times, block_s=1.0)
    if not np.isfinite(slope):
        return array.copy(), np.nan
    centered_time = times - float(np.mean(times))
    return array - slope * centered_time, float(slope)


def window_metrics(
    time_arr,
    grf,
    grf_left,
    grf_right,
    cop_ap,
    cop_ml,
    fs,
):
    t = np.asarray(time_arr, dtype=float)
    ap = np.asarray(cop_ap, dtype=float)
    ml = np.asarray(cop_ml, dtype=float)
    diff_ap = np.diff(ap)
    diff_ml = np.diff(ml)
    duration = float(t[-1] - t[0])
    path_length = float(np.sum(np.sqrt(diff_ap**2 + diff_ml**2)))
    endpoint_ap, _, _ = median_endpoint_shift(ap, fs)
    endpoint_ml, _, _ = median_endpoint_shift(ml, fs)
    step_ap = maximum_block_step_details(ap, t - t[0])
    step_ml = maximum_block_step_details(ml, t - t[0])
    sd_ap = float(np.std(ap, ddof=0))
    sd_ml = float(np.std(ml, ddof=0))
    smaller_sd = max(min(sd_ap, sd_ml), 1e-12)
    anisotropy = max(sd_ap, sd_ml) / smaller_sd
    return {
        "total_cv": coefficient_of_variation(grf),
        "left_cv": coefficient_of_variation(grf_left),
        "right_cv": coefficient_of_variation(grf_right),
        "ap_slope": signal_slope(ap, t),
        "ml_slope": signal_slope(ml, t),
        "endpoint_ap": endpoint_ap,
        "endpoint_ml": endpoint_ml,
        "sd_ap": sd_ap,
        "sd_ml": sd_ml,
        "anisotropy_ratio": float(anisotropy),
        "range_ap": robust_range(ap),
        "range_ml": robust_range(ml),
        "block_step_ap": step_ap["magnitude"],
        "block_step_ml": step_ml["magnitude"],
        "block_step_ap_time_s": step_ap["time_s"],
        "block_step_ml_time_s": step_ml["time_s"],
        "block_step_ap_before": step_ap["before"],
        "block_step_ap_after": step_ap["after"],
        "block_step_ml_before": step_ml["before"],
        "block_step_ml_after": step_ml["after"],
        "path_length": path_length,
        "mean_velocity": path_length / duration if duration > 0 else np.inf,
        "duration": duration,
    }


def stability_score(metrics: dict[str, float], thresholds: dict[str, float]) -> float:
    def ratio(value: float, limit: float) -> float:
        if not np.isfinite(value) or limit <= 0:
            return 1e6
        return min(abs(value) / limit, 100.0)

    return float(
        ratio(metrics["total_cv"], thresholds["total_cv"])
        + 0.5 * ratio(metrics["left_cv"], thresholds["side_cv"])
        + 0.5 * ratio(metrics["right_cv"], thresholds["side_cv"])
        + ratio(metrics["ap_slope"], thresholds["drift_m_s"])
        + ratio(metrics["ml_slope"], thresholds["drift_m_s"])
        + ratio(metrics["endpoint_ap"], thresholds["endpoint_m"])
        + ratio(metrics["endpoint_ml"], thresholds["endpoint_m"])
        + 0.5 * ratio(metrics["sd_ap"], thresholds["sd_m"])
        + 0.5 * ratio(metrics["sd_ml"], thresholds["sd_m"])
        + ratio(metrics["mean_velocity"], thresholds["mean_velocity_m_s"])
    )


def select_analysis_window(
    profile: Profile,
    time_arr,
    grf,
    grf_left,
    grf_right,
    cop_ap,
    cop_ml,
    fs,
):
    times = np.asarray(time_arr, dtype=float)
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("Timestamp tidak valid atau tidak monotonik.")

    if not profile.use_best_window:
        start = int(
            np.searchsorted(
                times,
                times[0] + profile.minimum_window_start_s,
                side="left",
            )
        )
        end = int(
            np.searchsorted(
                times,
                times[start] + ANALYSIS_DURATION_S,
                side="right",
            )
        )
        end = min(end, len(times))
        if end - start < 3 or times[end - 1] - times[start] < 29.95:
            raise ValueError("Durasi rekaman tidak cukup untuk analisis 30 detik.")
        metrics = window_metrics(
            times[start:end], grf[start:end], grf_left[start:end],
            grf_right[start:end], cop_ap[start:end], cop_ml[start:end], fs,
        )
        return start, end, metrics, stability_score(metrics, profile.thresholds)

    search_start_time = times[0] + profile.minimum_window_start_s
    last_start_time = times[-1] - ANALYSIS_DURATION_S
    if last_start_time < search_start_time:
        raise ValueError(
            "Rekaman terlalu pendek untuk memilih window stabil 30 detik."
        )

    step_s = 0.50
    candidate_starts = np.arange(search_start_time, last_start_time + 1e-9, step_s)
    candidates = []
    min_plate_force = 20.0

    for candidate_time in candidate_starts:
        start = int(np.searchsorted(times, candidate_time, side="left"))
        end = int(
            np.searchsorted(
                times,
                times[start] + ANALYSIS_DURATION_S,
                side="right",
            )
        )
        end = min(end, len(times))
        if end - start < 3 or times[end - 1] - times[start] < 29.95:
            continue

        finite = (
            np.isfinite(cop_ap[start:end])
            & np.isfinite(cop_ml[start:end])
            & np.isfinite(grf[start:end])
        )
        if np.mean(finite) < 0.995:
            continue
        if (
            np.min(grf_left[start:end]) <= min_plate_force
            or np.min(grf_right[start:end]) <= min_plate_force
        ):
            continue

        metrics = window_metrics(
            times[start:end],
            grf[start:end],
            grf_left[start:end],
            grf_right[start:end],
            cop_ap[start:end],
            cop_ml[start:end],
            fs,
        )
        score = stability_score(metrics, profile.thresholds)
        candidates.append((score, start, end, metrics))

    if not candidates:
        raise ValueError("Tidak ditemukan window analisis 30 detik yang valid.")

    score, start, end, metrics = min(candidates, key=lambda item: item[0])
    return start, end, metrics, float(score)


def calculate_confidence_ellipse(cop_ap, cop_ml):
    return prc.calculate_confidence_ellipse(cop_ap, cop_ml)


def spectral_metrics(frequency, power, minimum_hz=0.05, maximum_hz=5.0):
    f = np.asarray(frequency, dtype=float)
    p = np.asarray(power, dtype=float)
    valid = (
        np.isfinite(f)
        & np.isfinite(p)
        & (f >= minimum_hz)
        & (f <= maximum_hz)
        & (p >= 0.0)
    )
    f = f[valid]
    p = p[valid]
    if len(f) < 3 or np.sum(p) <= 0:
        return np.nan, np.nan
    peak_index = int(np.argmax(p))
    peak_frequency = float(f[peak_index])
    narrow_mask = np.abs(f - peak_frequency) <= 0.15
    narrow_ratio = float(np.sum(p[narrow_mask]) / np.sum(p))
    return peak_frequency, narrow_ratio


def physical_bounds_cm(config):
    half_distance = float(config["plate_center_distance_m"]) / 2.0
    half_width = float(config["plate_width_m"]) / 2.0
    half_length = float(config["plate_length_m"]) / 2.0
    return (
        -(half_distance + half_width) * 100.0,
        +(half_distance + half_width) * 100.0,
        -half_length * 100.0,
        +half_length * 100.0,
    )


def sensor_support_bounds_cm(config):
    half_distance = float(config["plate_center_distance_m"]) / 2.0
    x_local = np.asarray(config["sensor_x_local_m"], dtype=float)
    y_local = np.asarray(config["sensor_y_local_m"], dtype=float)
    return (
        (-half_distance + np.min(x_local[:4])) * 100.0,
        (+half_distance + np.max(x_local[4:])) * 100.0,
        float(np.min(y_local) * 100.0),
        float(np.max(y_local) * 100.0),
    )


def add_plate_outlines(axis, config):
    center_distance_cm = float(config["plate_center_distance_m"]) * 100.0
    width_cm = float(config["plate_width_m"]) * 100.0
    length_cm = float(config["plate_length_m"]) * 100.0
    for center_x in (-center_distance_cm / 2.0, center_distance_cm / 2.0):
        axis.add_patch(
            Rectangle(
                (center_x - width_cm / 2.0, -length_cm / 2.0),
                width_cm,
                length_cm,
                fill=False,
                linewidth=1.0,
                linestyle=":",
            )
        )


def make_zoom_bounds(ml_cm, ap_cm, physical_bounds, extra_points=None):
    ml = np.asarray(ml_cm, dtype=float)
    ap = np.asarray(ap_cm, dtype=float)
    x_min_phys, x_max_phys, y_min_phys, y_max_phys = physical_bounds
    x_values = [float(np.percentile(ml, 0.5)), float(np.percentile(ml, 99.5))]
    y_values = [float(np.percentile(ap, 0.5)), float(np.percentile(ap, 99.5))]
    if extra_points:
        for x_value, y_value in extra_points:
            if np.isfinite(x_value):
                x_values.append(float(x_value))
            if np.isfinite(y_value):
                y_values.append(float(y_value))
    x_low, x_high = min(x_values), max(x_values)
    y_low, y_high = min(y_values), max(y_values)
    x_span = max(4.0, x_high - x_low + 1.0)
    y_span = max(4.0, y_high - y_low + 1.0)
    x_center = 0.5 * (x_low + x_high)
    y_center = 0.5 * (y_low + y_high)
    x_min = max(x_min_phys, x_center - x_span / 2.0)
    x_max = min(x_max_phys, x_center + x_span / 2.0)
    y_min = max(y_min_phys, y_center - y_span / 2.0)
    y_max = min(y_max_phys, y_center + y_span / 2.0)
    return x_min, x_max, y_min, y_max


def compute_relative_heatmap(ml_cm, ap_cm, bounds_cm, grid_size=180):
    ml = np.asarray(ml_cm, dtype=float)
    ap = np.asarray(ap_cm, dtype=float)
    valid = np.isfinite(ml) & np.isfinite(ap)
    ml = ml[valid]
    ap = ap[valid]
    if len(ml) < 20:
        raise ValueError("Sampel CoP terlalu sedikit untuk heatmap.")

    if len(ml) > 2500:
        index = np.linspace(0, len(ml) - 1, 2500, dtype=int)
        ml_kde, ap_kde = ml[index], ap[index]
    else:
        ml_kde, ap_kde = ml, ap

    x_min, x_max, y_min, y_max = bounds_cm
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    try:
        kde = gaussian_kde(np.vstack([ml_kde, ap_kde]))
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    except (np.linalg.LinAlgError, ValueError):
        histogram, x_edges, y_edges = np.histogram2d(
            ml,
            ap,
            bins=100,
            range=[[x_min, x_max], [y_min, y_max]],
            density=True,
        )
        Z = gaussian_filter(histogram.T, sigma=1.5)
        X, Y = np.meshgrid(
            (x_edges[:-1] + x_edges[1:]) / 2.0,
            (y_edges[:-1] + y_edges[1:]) / 2.0,
        )
    maximum = float(np.nanmax(Z))
    if maximum > 0:
        Z = Z / maximum * 100.0
    return X, Y, Z



def write_standing_summary_csv(
    result: dict[str, object],
    data: dict[str, object],
) -> Path:
    """Simpan metrik utama standing sebagai CSV pendamping."""
    output_csv = prc.derived_output_path(
        str(data["filename"]), "_standing_summary_v85_dashboard", ".csv"
    )
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    rows = [
        ("software_version", result["software_version"], "-"),
        ("profile", result["profile_label"], "-"),
        ("trial_status", result["overall_status"], "-"),
        ("signal_quality", result["signal_status"], "-"),
        ("postural_stability", result["postural_status"], "-"),
        ("mass_status", result["mass_status"], "-"),
        ("effective_frame_rate", result["fs"], "Hz"),
        ("analysis_duration", metrics["duration"], "s"),
        ("measured_mass", result["measured_mass"], "kg"),
        ("measured_left_mass", result["measured_left_mass"], "kg"),
        ("measured_right_mass", result["measured_right_mass"], "kg"),
        ("left_load_share", result["left_share"], "%"),
        ("right_load_share", result["right_share"], "%"),
        ("mean_cop_ap", result["mean_ap"] * 100.0, "cm"),
        ("mean_cop_ml", result["mean_ml"] * 100.0, "cm"),
        ("cop_sd_ap", metrics["sd_ap"] * 100.0, "cm"),
        ("cop_sd_ml", metrics["sd_ml"] * 100.0, "cm"),
        ("cop_robust_range_ap", metrics["range_ap"] * 100.0, "cm"),
        ("cop_robust_range_ml", metrics["range_ml"] * 100.0, "cm"),
        ("max_block_step_ap", metrics["block_step_ap"] * 100.0, "cm"),
        ("max_block_step_ml", metrics["block_step_ml"] * 100.0, "cm"),
        ("max_block_step_ap_time", metrics["block_step_ap_time_s"], "s"),
        ("max_block_step_ml_time", metrics["block_step_ml_time_s"], "s"),
        ("block_step_limit", result.get("block_step_limit_m", np.nan) * 100.0, "cm"),
        ("near_limit_fraction", result.get("near_limit_fraction", np.nan), "fraction"),
        ("path_length", metrics["path_length"], "m"),
        ("mean_velocity", metrics["mean_velocity"] * 100.0, "cm/s"),
        ("ellipse_area_95", result["ellipse_area"] * 10000.0, "cm2"),
        ("ellipse_angle", result["ellipse_angle"], "degree"),
        ("grf_total_cv", metrics["total_cv"] * 100.0, "%"),
        ("grf_left_cv", metrics["left_cv"] * 100.0, "%"),
        ("grf_right_cv", metrics["right_cv"] * 100.0, "%"),
        ("psd_peak_ap", result["peak_ap_hz"], "Hz"),
        ("psd_peak_ml", result["peak_ml_hz"], "Hz"),
        ("serial_lost_frames", result["serial_lost_frames"], "count"),
        ("adc_rejected_record", result["record_adc_rejected"], "count"),
        ("malformed_record", result["record_malformed"], "count"),
        ("analysis_frame_gaps", result["analysis_frame_gaps"], "count"),
    ]
    prc.write_key_value_csv(output_csv, rows)
    print(f"[OK] Ringkasan standing tersimpan: {output_csv}")
    return output_csv


def _standing_detail_text(result: dict[str, object], data: dict[str, object]) -> str:
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    values = [
        ("Profile", result["profile_label"]),
        ("Trial status", result["overall_status"]),
        ("Signal quality", result["signal_status"]),
        ("Postural stability", result["postural_status"]),
        ("Mass status", result["mass_status"]),
        ("Measured mass", prc.format_metric(result["measured_mass"], "kg", 3)),
        ("Load share L/R", (
            f"{prc.format_metric(result['left_share'], '%', 2)} / "
            f"{prc.format_metric(result['right_share'], '%', 2)}"
        )),
        ("Mean global AP/ML", (
            f"{prc.format_metric(float(result['mean_ap']) * 100.0, 'cm', 3)} / "
            f"{prc.format_metric(float(result['mean_ml']) * 100.0, 'cm', 3)}"
        )),
        ("Sway SD AP/ML", (
            f"{prc.format_metric(metrics['sd_ap'] * 100.0, 'cm', 3)} / "
            f"{prc.format_metric(metrics['sd_ml'] * 100.0, 'cm', 3)}"
        )),
        ("Robust range AP/ML", (
            f"{prc.format_metric(metrics['range_ap'] * 100.0, 'cm', 3)} / "
            f"{prc.format_metric(metrics['range_ml'] * 100.0, 'cm', 3)}"
        )),
        ("Max 0.5-s shift AP/ML", (
            f"{prc.format_metric(metrics['block_step_ap'] * 100.0, 'cm', 3)} / "
            f"{prc.format_metric(metrics['block_step_ml'] * 100.0, 'cm', 3)}"
        )),
        ("Largest AP shift time", prc.format_metric(metrics["block_step_ap_time_s"], "s", 2)),
        ("Largest ML shift time", prc.format_metric(metrics["block_step_ml_time_s"], "s", 2)),
        ("0.5-s shift limit", prc.format_metric(float(result.get("block_step_limit_m", np.nan)) * 100.0, "cm", 3)),
        ("Path length", prc.format_metric(metrics["path_length"], "m", 4)),
        ("Mean velocity", prc.format_metric(metrics["mean_velocity"] * 100.0, "cm/s", 3)),
        ("95% ellipse area", prc.format_metric(float(result["ellipse_area"]) * 10000.0, "cm²", 3)),
        ("Ellipse direction", prc.format_metric(result["ellipse_angle"], "°", 1)),
        ("CoP drift AP/ML", (
            f"{prc.format_metric(metrics['ap_slope'] * 100.0, 'cm/s', 4)} / "
            f"{prc.format_metric(metrics['ml_slope'] * 100.0, 'cm/s', 4)}"
        )),
        ("GRF CV total/L/R", (
            f"{metrics['total_cv'] * 100.0:.3f}% / "
            f"{metrics['left_cv'] * 100.0:.3f}% / "
            f"{metrics['right_cv'] * 100.0:.3f}%"
        )),
        ("PSD peak AP/ML", (
            f"{prc.format_metric(result['peak_ap_hz'], 'Hz', 2)} / "
            f"{prc.format_metric(result['peak_ml_hz'], 'Hz', 2)}"
        )),
        ("Raw data", str(data["filename"])),
    ]
    return "\n".join(f"{name:<32}: {value}" for name, value in values)


def _standing_status_message(result: dict[str, object]) -> str:
    """Interpretasi ringkas dengan pemisahan core failure dan catatan postural."""
    status = str(result.get("overall_status", "-"))
    if status == prc.TRIAL_STATUS_PASS:
        return "Trial PASS. Data utama quiet standing layak digunakan."

    reasons: list[object] = []
    for key in ("failed_data", "failed_stability", "near_limit", "failed_placement",
                "record_gap_reasons", "selected_gap_reasons"):
        value = result.get(key, [])
        if isinstance(value, (list, tuple)):
            reasons.extend(value)
    if str(result.get("mass_status", "")).upper() == "REVIEW":
        reasons.append("akurasi massa referensi")

    if status == prc.TRIAL_STATUS_REPEAT:
        return (
            "Trial perlu diulang setelah penyebab core diperiksa. Perhatian utama: "
            f"{prc.compact_reason_text(reasons, limit=3)}."
        )
    return (
        "Trial tetap dapat digunakan tanpa wajib diulang. Catatan utama: "
        f"{prc.compact_reason_text(reasons, limit=3)}."
    )


def _configure_tree_status_tags(tree: object) -> None:
    """Warna baris tabel mengikuti level status."""
    for level, status in (
        ("success", "PASS"),
        ("warning", "CAUTION"),
        ("danger", "REPEAT_REQUIRED"),
        ("neutral", "UNKNOWN"),
    ):
        palette = prc.status_palette(status)
        tree.tag_configure(level, background=palette["background"], foreground=palette["foreground"])


def show_standing_result_window(
    result: dict[str, object],
    data: dict[str, object],
    figures: dict[str, plt.Figure],
    summary_csv: Path,
    output_paths: dict[str, Path],
) -> bool:
    """Tampilkan dashboard standing responsif. Return True bila trial diulang."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from tkinter.scrolledtext import ScrolledText
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    except ImportError as exc:
        print(f"[WARNING] GUI Tkinter tidak tersedia: {exc}")
        for fig in figures.values():
            fig.show()
        plt.show()
        return False

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"[WARNING] Window GUI tidak dapat dibuka: {exc}")
        plt.show()
        return False

    repeat_state = {"requested": False}
    root.title("Force Plate — Standing Result")
    root.geometry("1640x960")
    root.minsize(1120, 740)
    try:
        if root.tk.call("tk", "windowingsystem") == "win32":
            root.state("zoomed")
    except tk.TclError:
        pass

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Dashboard.TFrame", background="#F3F4F6")
    style.configure("Card.TFrame", background="#FFFFFF", relief="solid", borderwidth=1)
    style.configure("CardTitle.TLabel", background="#FFFFFF", foreground="#6B7280", font=("Segoe UI", 9))
    style.configure("CardValue.TLabel", background="#FFFFFF", foreground="#111827", font=("Segoe UI", 14, "bold"))
    style.configure("HeaderMeta.TLabel", background="#111827", foreground="#D1D5DB", font=("Segoe UI", 9))
    style.configure("TNotebook.Tab", padding=(12, 7), font=("Segoe UI", 9))
    style.configure("Action.TButton", padding=(10, 6))

    main = ttk.Frame(root, style="Dashboard.TFrame", padding=10)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.rowconfigure(4, weight=1)

    status = str(result["overall_status"])
    display_status = prc.humanize_status(status)
    badge_status = (
        "WITH NOTE" if status == prc.TRIAL_STATUS_NOTE
        else "REPEAT REQUIRED" if status == prc.TRIAL_STATUS_REPEAT
        else "PASS"
    )
    palette = prc.status_palette(status)

    header = tk.Frame(main, bg="#111827", padx=16, pady=11)
    header.grid(row=0, column=0, sticky="ew")
    tk.Label(
        header, text=str(result["profile_label"]), bg="#111827", fg="#FFFFFF",
        font=("Segoe UI", 18, "bold"),
    ).pack(side="left")
    tk.Label(
        header, text=badge_status, bg=palette["background"], fg=palette["foreground"],
        padx=12, pady=6, font=("Segoe UI", 11, "bold"),
    ).pack(side="right")

    metadata_text = (
        f"Massa {float(result['measured_mass']):.2f} kg  |  "
        f"Sampling efektif {float(result['fs']):.2f} Hz  |  "
        f"Window analisis {float(result['analysis_duration']):.2f} s  |  "
        f"Data {Path(str(data['filename'])).name}"
    )
    ttk.Label(
        main, text=metadata_text, style="HeaderMeta.TLabel", padding=(16, 6)
    ).grid(row=1, column=0, sticky="ew")

    banner = tk.Frame(main, bg=palette["background"], highlightbackground=palette["border"], highlightthickness=1)
    banner.grid(row=2, column=0, sticky="ew", pady=(8, 2))
    banner_label = tk.Label(
        banner, text=_standing_status_message(result), bg=palette["background"],
        fg=palette["foreground"], anchor="w", justify="left", padx=12, pady=7,
        font=("Segoe UI", 9, "bold"),
    )
    banner_label.pack(fill="x")
    banner.bind(
        "<Configure>",
        lambda event: banner_label.configure(
            wraplength=max(280, int(event.width) - 28)
        ),
        add="+",
    )

    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    cards = [
        ("Kelayakan Trial", display_status),
        ("Massa Terukur", prc.format_metric(result["measured_mass"], "kg", 2)),
        ("Distribusi Beban", f"Kiri {float(result['left_share']):.1f}% | Kanan {float(result['right_share']):.1f}%"),
        ("Mean CoP AP / ML", f"{float(result['mean_ap']) * 100.0:.2f} / {float(result['mean_ml']) * 100.0:.2f} cm"),
        ("Kecepatan Rata-rata", prc.format_metric(metrics["mean_velocity"] * 100.0, "cm/s", 3)),
        ("Luas Elips 95%", prc.format_metric(float(result["ellipse_area"]) * 10000.0, "cm²", 3)),
    ]
    cards_frame = ttk.Frame(main, style="Dashboard.TFrame")
    cards_frame.grid(row=3, column=0, sticky="ew", pady=(7, 7))
    card_widgets: list[ttk.Frame] = []
    for title, value in cards:
        card = ttk.Frame(cards_frame, style="Card.TFrame", padding=9)
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text=value, style="CardValue.TLabel", justify="left", wraplength=260).pack(anchor="w", pady=(3, 0))
        card_widgets.append(card)

    layout_state = {"columns": 0}
    def layout_cards(event: object | None = None) -> None:
        width = max(cards_frame.winfo_width(), root.winfo_width())
        columns = 6 if width >= 1500 else 3 if width >= 900 else 2
        if layout_state["columns"] == columns:
            return
        layout_state["columns"] = columns
        for index in range(6):
            cards_frame.columnconfigure(index, weight=1 if index < columns else 0, uniform="card" if index < columns else "")
        for index, card in enumerate(card_widgets):
            card.grid_forget()
            card.grid(row=index // columns, column=index % columns, sticky="nsew", padx=3, pady=3)
    cards_frame.bind("<Configure>", layout_cards)
    root.after_idle(layout_cards)

    notebook = ttk.Notebook(main)
    notebook.grid(row=4, column=0, sticky="nsew")
    overview_tab = ttk.Frame(notebook)
    spatial_tab = ttk.Frame(notebook)
    qc_tab = ttk.Frame(notebook, padding=10)
    detail_tab = ttk.Frame(notebook, padding=10)
    system_tab = ttk.Frame(notebook, padding=10)
    note_tab = ttk.Frame(notebook, padding=10)
    notebook.add(overview_tab, text="Ringkasan Grafik")
    notebook.add(spatial_tab, text="CoP Spatial")
    notebook.add(qc_tab, text="Quality Control")
    notebook.add(detail_tab, text="Detail")
    notebook.add(system_tab, text="System Health")
    notebook.add(note_tab, text="Catatan Operator")

    canvas_refs: list[FigureCanvasTkAgg] = []
    resize_callbacks: list[callable] = []

    def attach_figure(tab: ttk.Frame, figure: plt.Figure, note: str) -> None:
        """Embed figure responsif, termasuk figure pada tab yang awalnya tersembunyi."""
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        note_label = ttk.Label(
            tab, text=note, anchor="w", justify="left", padding=(8, 5)
        )
        note_label.grid(row=0, column=0, sticky="ew")

        plot_frame = ttk.Frame(tab)
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        canvas_widget = canvas.get_tk_widget()
        # Ukuran request kecil mencegah figure tersembunyi memaksa Notebook
        # lebih lebar daripada window. Ukuran aktual mengikuti area tab.
        canvas_widget.configure(width=100, height=100, highlightthickness=0)
        canvas_widget.grid(row=0, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        state = {"last": (0, 0), "job": None}

        def apply_resize() -> None:
            state["job"] = None
            width = max(1, int(canvas_widget.winfo_width()))
            height = max(1, int(canvas_widget.winfo_height()))
            if width < 120 or height < 120:
                return
            if state["last"] == (width, height):
                return
            state["last"] = (width, height)
            figure.set_size_inches(
                width / figure.dpi, height / figure.dpi, forward=False
            )
            canvas.draw_idle()

        def schedule_resize(event: object | None = None) -> None:
            if state["job"] is not None:
                try:
                    root.after_cancel(state["job"])
                except tk.TclError:
                    pass
            state["job"] = root.after(60, apply_resize)

        canvas_widget.bind("<Configure>", schedule_resize, add="+")
        note_label.bind(
            "<Configure>",
            lambda event: note_label.configure(
                wraplength=max(240, int(event.width) - 20)
            ),
            add="+",
        )
        canvas_refs.append(canvas)
        resize_callbacks.append(schedule_resize)

    attach_figure(
        overview_tab, figures["overview"],
        "Area berarsir adalah window analisis. Bagian post-window hanya untuk mendeteksi gerakan setelah analisis.",
    )
    attach_figure(
        spatial_tab, figures["spatial"],
        "Heatmap menunjukkan kepadatan posisi CoP, bukan tekanan telapak kaki. Konvensi: +AP = anterior dan +ML = kanan.",
    )

    def refresh_visible_figure(event: object | None = None) -> None:
        for callback in resize_callbacks:
            callback()

    notebook.bind("<<NotebookTabChanged>>", refresh_visible_figure, add="+")
    root.after_idle(refresh_visible_figure)

    qc_pane = ttk.Panedwindow(qc_tab, orient="vertical")
    qc_pane.pack(fill="both", expand=True)
    qc_table_frame = ttk.Frame(qc_pane)
    qc_detail_frame = ttk.Frame(qc_pane, padding=(0, 8, 0, 0))
    qc_pane.add(qc_table_frame, weight=3)
    qc_pane.add(qc_detail_frame, weight=2)

    qc_tree = ttk.Treeview(qc_table_frame, columns=("group", "status", "details"), show="headings", height=8)
    qc_scroll_y = ttk.Scrollbar(qc_table_frame, orient="vertical", command=qc_tree.yview)
    qc_scroll_x = ttk.Scrollbar(qc_table_frame, orient="horizontal", command=qc_tree.xview)
    qc_tree.configure(yscrollcommand=qc_scroll_y.set, xscrollcommand=qc_scroll_x.set)
    for column, title, width in (
        ("group", "Kelompok", 210),
        ("status", "Status", 160),
        ("details", "Pemeriksaan gagal / mendekati batas", 900),
    ):
        qc_tree.heading(column, text=title)
        qc_tree.column(column, width=width, minwidth=120, anchor="w", stretch=(column == "details"))
    qc_tree.grid(row=0, column=0, sticky="nsew")
    qc_scroll_y.grid(row=0, column=1, sticky="ns")
    qc_scroll_x.grid(row=1, column=0, sticky="ew")
    qc_table_frame.rowconfigure(0, weight=1)
    qc_table_frame.columnconfigure(0, weight=1)
    _configure_tree_status_tags(qc_tree)

    qc_rows = [
        ("Kualitas sinyal", result["signal_status"], result["failed_data"]),
        ("Stabilitas postural", result["postural_status"], result["failed_stability"]),
        (
            "Mendekati batas",
            "CAUTION" if result["near_limit"] else "PASS",
            [
                result.get("near_limit_details", {}).get(
                    name, prc.humanize_check_name(name)
                )
                for name in result["near_limit"]
            ],
        ),
        ("Penempatan", result["placement_status"], result["failed_placement"]),
        ("Akurasi massa", result["mass_status"], ["akurasi massa referensi"] if str(result["mass_status"]) == "REVIEW" else []),
    ]
    qc_details_by_iid: dict[str, str] = {}
    for group, qc_status, details in qc_rows:
        detail_list = list(details) if isinstance(details, (list, tuple)) else ([details] if details else [])
        full_detail = (
            "; ".join(
                str(item)
                if any(token in str(item) for token in (":", ";", " cm", " %", " t="))
                else prc.humanize_check_name(item)
                for item in detail_list
            )
            if detail_list
            else "Tidak ada masalah terdeteksi."
        )
        iid = qc_tree.insert(
            "", "end",
            values=(group, prc.humanize_status(qc_status), full_detail),
            tags=(prc.status_level(qc_status),),
        )
        qc_details_by_iid[iid] = f"{group}\nStatus: {prc.humanize_status(qc_status)}\n\n{full_detail}"

    ttk.Label(qc_detail_frame, text="Detail pemeriksaan terpilih", font=("Segoe UI", 10, "bold")).pack(anchor="w")
    qc_detail_text = ScrolledText(qc_detail_frame, wrap="word", height=6, font=("Segoe UI", 9))
    qc_detail_text.pack(fill="both", expand=True, pady=(4, 0))
    qc_detail_text.configure(state="disabled")
    def update_qc_detail(event: object | None = None) -> None:
        selection = qc_tree.selection()
        if not selection:
            return
        content = qc_details_by_iid.get(selection[0], "-")
        qc_detail_text.configure(state="normal")
        qc_detail_text.delete("1.0", "end")
        qc_detail_text.insert("1.0", content)
        qc_detail_text.configure(state="disabled")
    qc_tree.bind("<<TreeviewSelect>>", update_qc_detail)
    first_items = qc_tree.get_children()
    if first_items:
        qc_tree.selection_set(first_items[0])
        root.after_idle(update_qc_detail)

    detail_text = ScrolledText(detail_tab, wrap="word", font=("Consolas", 10))
    detail_text.insert("1.0", _standing_detail_text(result, data))
    detail_text.configure(state="disabled")
    detail_text.pack(fill="both", expand=True)

    system_frame = ttk.Frame(system_tab)
    system_frame.pack(fill="both", expand=True)
    system_tree = ttk.Treeview(system_frame, columns=("metric", "value"), show="headings")
    system_scroll = ttk.Scrollbar(system_frame, orient="vertical", command=system_tree.yview)
    system_tree.configure(yscrollcommand=system_scroll.set)
    system_tree.heading("metric", text="Parameter")
    system_tree.heading("value", text="Nilai")
    system_tree.column("metric", width=330, anchor="w")
    system_tree.column("value", width=850, anchor="w")
    system_tree.grid(row=0, column=0, sticky="nsew")
    system_scroll.grid(row=0, column=1, sticky="ns")
    system_frame.rowconfigure(0, weight=1)
    system_frame.columnconfigure(0, weight=1)
    system_rows = [
        ("Firmware version", data.get("firmware_version_message", "-")),
        ("Stream protocol", data.get("stream_protocol_message", "-")),
        ("Firmware profile", data.get("firmware_profile_message", "-")),
        ("Tare", "OK — streaming hanya dimulai setelah #TARE_OK"),
        ("Zero gate", (data.get("zero_gate_info") or {}).get("status", "NOT_AVAILABLE")),
        ("Calibration version", data.get("calibration_version", "-")),
        ("ADC nominal rate", f"{data.get('adc_nominal_sps', '-')} SPS"),
        ("Effective frame rate", f"{float(result['fs']):.3f} Hz"),
        ("Serial lost frames", result["serial_lost_frames"]),
        ("ADC rejected record", result["record_adc_rejected"]),
        ("Malformed/non-frame record", result["record_malformed"]),
        ("CRC checksum errors", int(data.get("checksum_error_rows", 0) or 0)),
        ("Serial transport decode errors", int((data.get("serial_transport_diag") or {}).get("decode_errors", 0))),
        ("Serial transport resync events", int((data.get("serial_transport_diag") or {}).get("resync_events", 0))),
        ("Serial NUL noise lines", int((data.get("serial_transport_diag") or {}).get("noise_only_lines", 0))),
        ("Serial NUL bytes seen", int((data.get("serial_transport_diag") or {}).get("nul_bytes_seen", 0))),
        ("Recovered prefixed lines", int((data.get("serial_transport_diag") or {}).get("recovered_prefixed_lines", 0))),
        ("Serial max pending lines", int((data.get("serial_transport_diag") or {}).get("max_pending_lines", 0))),
        ("Protocol error counts", str(data.get("protocol_error_counts", {}))),
        ("Malformed examples", " | ".join(data.get("malformed_examples", [])) if data.get("malformed_examples") else "-"),
        ("Frame gaps full", result["analysis_frame_gaps"]),
        ("Frame gaps analysis", result["selected_frame_gaps"]),
        ("Gap status full", result.get("record_gap_status", "-")),
        ("Gap status analysis", result.get("selected_gap_status", "-")),
        ("MUX skew mean (raw)", prc.format_metric(result["mux_skew_mean_us"], "µs", 1)),
        ("MUX skew P95 (raw)", prc.format_metric(result["mux_skew_p95_us"], "µs", 1)),
        ("MUX alignment", "APPLIED" if data.get("mux_alignment_applied", False) else "NOT_APPLIED"),
        ("MUX alignment coverage", prc.format_metric(float(data.get("mux_alignment_coverage", float("nan"))) * 100.0, "%", 2)),
        ("Force filter", prc.format_metric(data.get("force_filter_hz"), "Hz", 1)),
        ("Software version", result["software_version"]),
        ("Serial baud", data.get("serial_baud", "-")),
    ]
    for metric, value in system_rows:
        system_tree.insert("", "end", values=(metric, value))

    ttk.Label(note_tab, text="Catatan disimpan sebagai file teks pendamping dan tidak mengubah CSV mentah.").pack(anchor="w", pady=(0, 8))
    note_text = ScrolledText(note_tab, wrap="word", height=16)
    note_text.pack(fill="both", expand=True)

    button_bar = ttk.Frame(main, style="Dashboard.TFrame")
    button_bar.grid(row=5, column=0, sticky="ew", pady=(7, 0))
    left_actions = ttk.Frame(button_bar, style="Dashboard.TFrame")
    right_actions = ttk.Frame(button_bar, style="Dashboard.TFrame")
    left_actions.pack(side="left")
    right_actions.pack(side="right")

    def show_open_result(path: Path) -> None:
        ok, message = prc.open_path_with_default_app(path)
        if not ok:
            messagebox.showerror("Gagal membuka file", message)

    def export_figures() -> None:
        selected = filedialog.asksaveasfilename(
            title="Ekspor grafik standing",
            defaultextension=".pdf",
            filetypes=[("PDF multi-halaman", "*.pdf"), ("PNG", "*.png")],
            initialfile=Path(str(data["filename"])).with_suffix("").name + "_standing_dashboard.pdf",
        )
        if not selected:
            return
        target = Path(selected)
        try:
            saved: list[Path] = []
            if target.suffix.lower() == ".pdf":
                with PdfPages(target) as pdf:
                    pdf.savefig(figures["overview"], bbox_inches="tight")
                    pdf.savefig(figures["spatial"], bbox_inches="tight")
                saved.append(target)
            else:
                overview_target = target.with_suffix(".png")
                spatial_target = overview_target.with_name(overview_target.stem + "_spatial.png")
                figures["overview"].savefig(overview_target, dpi=200, bbox_inches="tight")
                figures["spatial"].savefig(spatial_target, dpi=200, bbox_inches="tight")
                saved.extend([overview_target, spatial_target])
        except (OSError, ValueError) as exc:
            messagebox.showerror("Ekspor gagal", str(exc))
            return
        messagebox.showinfo("Ekspor selesai", "Grafik tersimpan:\n" + "\n".join(str(path) for path in saved))

    def save_note() -> None:
        note = note_text.get("1.0", "end").strip()
        if not note:
            messagebox.showwarning("Catatan kosong", "Masukkan catatan operator terlebih dahulu.")
            return
        try:
            note_path = prc.write_operator_note(str(data["filename"]), "STANDING", status, note)
        except OSError as exc:
            messagebox.showerror("Gagal menyimpan catatan", str(exc))
            return
        messagebox.showinfo("Catatan tersimpan", str(note_path))

    def request_repeat() -> None:
        repeat_state["requested"] = True
        root.destroy()

    ttk.Button(left_actions, text="Ekspor Grafik", style="Action.TButton", command=export_figures).pack(side="left", padx=3)
    ttk.Button(left_actions, text="Buka Ringkasan", style="Action.TButton", command=lambda: show_open_result(summary_csv)).pack(side="left", padx=3)
    ttk.Button(left_actions, text="Buka Data Mentah", style="Action.TButton", command=lambda: show_open_result(Path(str(data["filename"])))).pack(side="left", padx=3)
    ttk.Button(left_actions, text="Buka Folder Hasil", style="Action.TButton", command=lambda: show_open_result(Path(str(data["filename"])).parent)).pack(side="left", padx=3)
    ttk.Button(left_actions, text="Simpan Catatan", style="Action.TButton", command=save_note).pack(side="left", padx=3)
    ttk.Button(right_actions, text="Ulangi Pengukuran", style="Action.TButton", command=request_repeat).pack(side="left", padx=3)
    ttk.Button(right_actions, text="Tutup", style="Action.TButton", command=root.destroy).pack(side="left", padx=3)

    root.bind("<Control-e>", lambda event: export_figures())
    root.bind("<Control-r>", lambda event: request_repeat())
    root.bind("<Escape>", lambda event: root.destroy())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return bool(repeat_state["requested"])

def main() -> int:
    print("\n--- MODUL STANDING v8.5 CRC / GAP-ROBUST POLICY ---")
    print(
        "[INFO] Faktor kalibrasi hasil refinement 12 kondisi statis: "
        "L1-L4=17553.7 dan R1-R4=17591.5 counts/kg."
    )

    try:
        profile, metadata = ask_profile()
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    expected_left_mass = metadata.get("expected_left_mass_kg")
    expected_right_mass = metadata.get("expected_right_mass_kg")
    expected_human_mass = metadata.get("expected_human_mass_kg")
    mass_reference_quality = str(
        metadata.get("mass_reference_quality") or "unknown"
    )

    config = prc.load_config()
    data = prc.acquire_data(
        filename_prefix="FP_Standing",
        calculate_cop_flag=True,
        mode="standing",
        config=config,
        acquisition_profile=profile.acquisition_profile,
    )
    if data is None:
        return 1

    time_arr = np.asarray(data["time"], dtype=float)
    frame_arr = np.asarray(data["frame"], dtype=np.int64)
    grf_all = np.asarray(data["grf"], dtype=float)
    grf_left_all = np.asarray(data["grf_l"], dtype=float)
    grf_right_all = np.asarray(data["grf_r"], dtype=float)
    cop_ap_all = np.asarray(data["cop_ap"], dtype=float)
    cop_ml_all = np.asarray(data["cop_ml"], dtype=float)
    cop_l_ap_all = np.asarray(data["cop_l_ap"], dtype=float)
    cop_l_ml_all = np.asarray(data["cop_l_ml"], dtype=float)
    cop_r_ap_all = np.asarray(data["cop_r_ap"], dtype=float)
    cop_r_ml_all = np.asarray(data["cop_r_ml"], dtype=float)
    force_channels_all = np.asarray(data["force_channels_filtered_n"], dtype=float)

    lengths = {
        len(time_arr), len(frame_arr), len(grf_all), len(grf_left_all),
        len(grf_right_all), len(cop_ap_all), len(cop_ml_all),
        len(cop_l_ap_all), len(cop_l_ml_all), len(cop_r_ap_all),
        len(cop_r_ml_all), len(force_channels_all),
    }
    if (
        len(lengths) != 1
        or force_channels_all.ndim != 2
        or force_channels_all.shape[1] != 8
    ):
        print("[INVALID TRIAL] Panjang atau bentuk data tidak sejajar.")
        return 1

    sampling_diag = prc.get_sampling_diagnostics(frame_arr, time_arr)
    serial_diag = prc.get_serial_diagnostics(data["all_received_frames"])
    fs = float(sampling_diag["fs"])
    if not np.isfinite(fs) or fs <= 0:
        print("[INVALID TRIAL] Sampling rate tidak valid.")
        return 1

    try:
        cop_ap_interpolated = interpolate_short_nan_gaps(cop_ap_all, time_arr)
        cop_ml_interpolated = interpolate_short_nan_gaps(cop_ml_all, time_arr)
    except ValueError as exc:
        print(f"[INVALID TRIAL] {exc}")
        return 1

    hampel_half_window = max(3, int(round(0.10 * fs)))
    cop_ap_hampel, replaced_ap_all = hampel_filter(
        cop_ap_interpolated, half_window=hampel_half_window
    )
    cop_ml_hampel, replaced_ml_all = hampel_filter(
        cop_ml_interpolated, half_window=hampel_half_window
    )
    # CoP sudah dihitung dari delapan kanal gaya yang difilter di processing.py.
    # Jangan melakukan low-pass kedua karena double-filtering dapat mengubah
    # fase/transien dan membuat path length antarversi sulit dibandingkan.
    cop_ap_filtered_all = cop_ap_hampel.copy()
    cop_ml_filtered_all = cop_ml_hampel.copy()

    try:
        start_idx, end_idx, selection_metrics, selection_score = (
            select_analysis_window(
                profile,
                time_arr,
                grf_all,
                grf_left_all,
                grf_right_all,
                cop_ap_filtered_all,
                cop_ml_filtered_all,
                fs,
            )
        )
    except ValueError as exc:
        print(f"[INVALID TRIAL] {exc}")
        return 1

    time_window = time_arr[start_idx:end_idx]
    time_window_rel = time_window - time_window[0]
    selected_frame_arr = frame_arr[start_idx:end_idx]
    selected_sampling_diag = prc.get_sampling_diagnostics(
        selected_frame_arr, time_window
    )
    cop_ap = cop_ap_filtered_all[start_idx:end_idx]
    cop_ml = cop_ml_filtered_all[start_idx:end_idx]
    grf = grf_all[start_idx:end_idx]
    grf_left = grf_left_all[start_idx:end_idx]
    grf_right = grf_right_all[start_idx:end_idx]
    force_channels = force_channels_all[start_idx:end_idx]

    replaced_ratio = max(
        replaced_ap_all / max(1, len(cop_ap_all)),
        replaced_ml_all / max(1, len(cop_ml_all)),
    )

    metrics = window_metrics(
        time_window, grf, grf_left, grf_right, cop_ap, cop_ml, fs
    )
    cop_ap_detrended, _ = detrend_with_block_slope(cop_ap, time_window_rel)
    cop_ml_detrended, _ = detrend_with_block_slope(cop_ml, time_window_rel)
    detrended_sd_ap = float(np.std(cop_ap_detrended, ddof=0))
    detrended_sd_ml = float(np.std(cop_ml_detrended, ddof=0))
    detrended_ellipse_area, _, _, _ = calculate_confidence_ellipse(
        cop_ap_detrended, cop_ml_detrended
    )

    mean_ap = float(np.mean(cop_ap))
    mean_ml = float(np.mean(cop_ml))
    # Massa menggunakan median agar lebih robust terhadap transien singkat.
    measured_mass = float(np.median(grf) / prc.GRAVITY)
    measured_left_mass = float(np.median(grf_left) / prc.GRAVITY)
    measured_right_mass = float(np.median(grf_right) / prc.GRAVITY)

    mean_l_ap = float(np.nanmean(cop_l_ap_all[start_idx:end_idx]))
    mean_l_ml = float(np.nanmean(cop_l_ml_all[start_idx:end_idx]))
    mean_r_ap = float(np.nanmean(cop_r_ap_all[start_idx:end_idx]))
    mean_r_ml = float(np.nanmean(cop_r_ml_all[start_idx:end_idx]))

    endpoint_shift_ap, endpoint_ap_start, endpoint_ap_end = median_endpoint_shift(
        cop_ap, fs
    )
    endpoint_shift_ml, endpoint_ml_start, endpoint_ml_end = median_endpoint_shift(
        cop_ml, fs
    )

    block_summaries = summarize_time_blocks(
        time_window_rel, cop_ap, cop_ml, block_s=5.0
    )
    anticipation_info = (
        detect_end_anticipation(block_summaries)
        if profile.name == "human"
        else {"detected": False, "axis": "-"}
    )
    post_window_motion = (
        detect_post_window_motion(
            time_arr, cop_ap_filtered_all, cop_ml_filtered_all,
            end_idx, fs,
        )
        if profile.name == "human"
        else {
            "available": False, "detected": False, "time_s": np.nan,
            "shift_ap": np.nan, "shift_ml": np.nan,
        }
    )

    ellipse_area, major_axis, minor_axis, ellipse_angle = (
        calculate_confidence_ellipse(cop_ap, cop_ml)
    )

    ap_cm = cop_ap * 100.0
    ml_cm = cop_ml * 100.0
    nperseg = min(len(ap_cm), max(16, int(round(fs * 10.0))))
    noverlap = nperseg // 2
    f_ap, pxx_ap = welch(
        ap_cm,
        fs=fs,
        window="hann",
        detrend="linear",
        nperseg=nperseg,
        noverlap=noverlap,
    )
    f_ml, pxx_ml = welch(
        ml_cm,
        fs=fs,
        window="hann",
        detrend="linear",
        nperseg=nperseg,
        noverlap=noverlap,
    )
    frequency_band = (f_ap >= 0.05) & (f_ap <= 5.0)
    peak_ap_hz, narrow_ap_ratio = spectral_metrics(f_ap, pxx_ap)
    peak_ml_hz, narrow_ml_ratio = spectral_metrics(f_ml, pxx_ml)

    threshold = profile.thresholds
    narrowband_detected = bool(
        (
            np.isfinite(peak_ap_hz)
            and peak_ap_hz >= threshold["narrowband_min_hz"]
            and narrow_ap_ratio >= threshold["narrowband_ratio"]
            and metrics["sd_ap"] >= threshold["narrowband_min_sd_m"]
        )
        or (
            np.isfinite(peak_ml_hz)
            and peak_ml_hz >= threshold["narrowband_min_hz"]
            and narrow_ml_ratio >= threshold["narrowband_ratio"]
            and metrics["sd_ml"] >= threshold["narrowband_min_sd_m"]
        )
    )

    support_bounds = sensor_support_bounds_cm(config)
    x_min, x_max, y_min, y_max = support_bounds
    bounds_margin_cm = 0.5
    bounds_pass = bool(
        np.all((ml_cm >= x_min - bounds_margin_cm) & (ml_cm <= x_max + bounds_margin_cm))
        and np.all((ap_cm >= y_min - bounds_margin_cm) & (ap_cm <= y_max + bounds_margin_cm))
    )
    both_plates_loaded = bool(
        np.all(grf_left > float(config["min_plate_force_n"]))
        and np.all(grf_right > float(config["min_plate_force_n"]))
    )
    channel_mean_n = np.mean(force_channels, axis=0)
    channel_negative_ratio = np.mean(force_channels < -2.0, axis=0)
    channel_force_valid = bool(np.all(channel_negative_ratio <= 0.001))

    half_distance = float(config["plate_center_distance_m"]) / 2.0
    center_mass_contribution_ml = (
        measured_left_mass * (-half_distance)
        + measured_right_mass * (+half_distance)
    ) / (measured_left_mass + measured_right_mass)
    local_contribution_ml = (
        measured_left_mass * mean_l_ml
        + measured_right_mass * mean_r_ml
    ) / (measured_left_mass + measured_right_mass)
    local_contribution_ap = (
        measured_left_mass * mean_l_ap
        + measured_right_mass * mean_r_ap
    ) / (measured_left_mass + measured_right_mass)
    reconstructed_ml = center_mass_contribution_ml + local_contribution_ml
    reconstructed_ap = local_contribution_ap
    reconstruction_error_ml = mean_ml - reconstructed_ml
    reconstruction_error_ap = mean_ap - reconstructed_ap

    centering_pass = True
    if profile.name != "human":
        centering_pass = bool(
            max(
                abs(mean_l_ap), abs(mean_l_ml),
                abs(mean_r_ap), abs(mean_r_ml),
            ) <= profile.centering_tolerance_m
        )

    expected_total_mass = None
    total_mass_error_percent = None
    left_mass_error_percent = None
    right_mass_error_percent = None
    human_mass_error_percent = None
    reference_center_ml = None

    if expected_left_mass is not None and expected_right_mass is not None:
        expected_left_mass = float(expected_left_mass)
        expected_right_mass = float(expected_right_mass)
        expected_total_mass = expected_left_mass + expected_right_mass
        total_mass_error_percent = (
            abs(measured_mass - expected_total_mass) / expected_total_mass * 100.0
        )
        left_mass_error_percent = (
            abs(measured_left_mass - expected_left_mass) / expected_left_mass * 100.0
        )
        right_mass_error_percent = (
            abs(measured_right_mass - expected_right_mass) / expected_right_mass * 100.0
        )
        reference_center_ml = (
            expected_left_mass * (-half_distance)
            + expected_right_mass * (+half_distance)
        ) / expected_total_mass

    if profile.name == "human" and expected_human_mass is not None:
        expected_human_mass = float(expected_human_mass)
        human_mass_error_percent = (
            abs(measured_mass - expected_human_mass)
            / expected_human_mass
            * 100.0
        )

    record_adc_rejected = int(
        data.get("adc_rejected_record", data["adc_rejected_frames"])
    )
    record_malformed = int(
        data.get("malformed_rows_record", data["malformed_rows"])
    )

    record_gap_status, record_gap_reasons = prc.classify_gap_severity(
        sampling_diag,
        hard_max_consecutive_missing=int(
            config.get("standing_gap_hard_max_consecutive_missing", 8)
        ),
        hard_missing_fraction=float(
            config.get("standing_gap_hard_missing_fraction", 0.002)
        ),
        hard_duration_s=float(
            config.get("standing_gap_hard_duration_s", 0.025)
        ),
    )
    selected_gap_status, selected_gap_reasons = prc.classify_gap_severity(
        selected_sampling_diag,
        hard_max_consecutive_missing=int(
            config.get("standing_gap_hard_max_consecutive_missing", 8)
        ),
        hard_missing_fraction=float(
            config.get("standing_gap_hard_missing_fraction", 0.002)
        ),
        hard_duration_s=float(
            config.get("standing_gap_hard_duration_s", 0.025)
        ),
    )
    gap_hard_failure = bool(
        record_gap_status == prc.TRIAL_STATUS_REPEAT
        or selected_gap_status == prc.TRIAL_STATUS_REPEAT
    )
    gap_has_note = bool(
        record_gap_status == "NOTE" or selected_gap_status == "NOTE"
    )

    # Malformed/NUL tidak lagi otomatis menggagalkan standing. Integritas utama
    # ditentukan oleh kontinuitas frame yang benar-benar dipakai, ADC status,
    # dan urutan timestamp. Noise tanpa frame tetap disimpan sebagai audit.
    data_checks = {
        "record_integrity": (
            not gap_hard_failure
            and record_adc_rejected == 0
            and sampling_diag.get("out_of_order", 0) == 0
            and selected_sampling_diag.get("out_of_order", 0) == 0
        ),
        "both_plates_loaded": both_plates_loaded,
        "physical_bounds": bounds_pass,
        "channel_force_valid": channel_force_valid,
        "hampel_ratio": replaced_ratio <= MAX_HAMPEL_REPLACEMENT_RATIO,
    }
    if expected_total_mass is not None:
        data_checks["total_mass_accuracy"] = (
            total_mass_error_percent <= threshold["mass_error_percent"]
        )
        data_checks["side_mass_accuracy"] = (
            left_mass_error_percent <= threshold["side_mass_error_percent"]
            and right_mass_error_percent <= threshold["side_mass_error_percent"]
        )
    human_mass_accuracy_pass = (
        human_mass_error_percent is None
        or human_mass_error_percent <= threshold["mass_error_percent"]
    )

    stability_checks = {
        "grf_total_cv": metrics["total_cv"] <= threshold["total_cv"],
        "grf_side_cv": (
            metrics["left_cv"] <= threshold["side_cv"]
            and metrics["right_cv"] <= threshold["side_cv"]
        ),
        "cop_drift": (
            abs(metrics["ap_slope"]) <= threshold["drift_m_s"]
            and abs(metrics["ml_slope"]) <= threshold["drift_m_s"]
        ),
        "endpoint_shift": (
            abs(endpoint_shift_ap) <= threshold["endpoint_m"]
            and abs(endpoint_shift_ml) <= threshold["endpoint_m"]
        ),
        "cop_sd": (
            metrics["sd_ap"] <= threshold["sd_m"]
            and metrics["sd_ml"] <= threshold["sd_m"]
        ),
        "cop_robust_range": (
            metrics["range_ap"] <= threshold["robust_range_m"]
            and metrics["range_ml"] <= threshold["robust_range_m"]
        ),
        "mean_velocity": metrics["mean_velocity"] <= threshold["mean_velocity_m_s"],
        "block_step": (
            metrics["block_step_ap"] <= threshold["block_step_m"]
            and metrics["block_step_ml"] <= threshold["block_step_m"]
        ),
        "narrowband_motion": not narrowband_detected,
    }

    low_frequency_dominance = bool(
        profile.name == "human"
        and (
            (
                np.isfinite(peak_ap_hz)
                and peak_ap_hz <= threshold["lowfreq_max_hz"]
                and narrow_ap_ratio >= threshold["lowfreq_ratio"]
                and metrics["sd_ap"] >= 0.0020
            )
            or (
                np.isfinite(peak_ml_hz)
                and peak_ml_hz <= threshold["lowfreq_max_hz"]
                and narrow_ml_ratio >= threshold["lowfreq_ratio"]
                and metrics["sd_ml"] >= 0.0020
            )
        )
    )

    placement_checks = {}
    if profile.name != "human":
        placement_checks["load_centering"] = centering_pass
        if (
            profile.name == "static_rigid"
            and reference_center_ml is not None
            and centering_pass
        ):
            placement_checks["expected_cop_accuracy"] = (
                abs(mean_ap) <= threshold["cop_error_m"]
                and abs(mean_ml - reference_center_ml) <= threshold["cop_error_m"]
            )

    data_quality_pass = bool(all(data_checks.values()))
    load_stability_pass = bool(all(stability_checks.values()))
    placement_pass = bool(all(placement_checks.values())) if placement_checks else True

    hard_movement = bool(
        profile.name == "human"
        and (
            max(metrics["block_step_ap"], metrics["block_step_ml"])
            > threshold["hard_block_step_m"]
            or max(metrics["range_ap"], metrics["range_ml"])
            > threshold["hard_range_m"]
        )
    )

    if profile.name == "human":
        # v8.5: core signal/hard movement tetap ketat. Gap kecil terisolasi dan
        # gap kecil menjadi catatan; noise-only transport tetap audit, bukan alasan mengulang.
        signal_status = "PASS" if data_quality_pass else "REPEAT_REQUIRED"
        if hard_movement:
            postural_status = "REPEAT_REQUIRED"
        else:
            postural_status = "PASS" if load_stability_pass else "REVIEW"

        if human_mass_error_percent is None:
            mass_status = "NOT_CHECKED"
        elif mass_reference_quality == "estimated":
            mass_status = "INFORMATIONAL"
        else:
            mass_status = "PASS" if human_mass_accuracy_pass else "REVIEW"

        if not data_quality_pass or hard_movement:
            trial_usability = prc.TRIAL_STATUS_REPEAT
        elif gap_has_note or not load_stability_pass or mass_status == "REVIEW":
            trial_usability = prc.TRIAL_STATUS_NOTE
        else:
            trial_usability = prc.TRIAL_STATUS_PASS
        overall_status = trial_usability
    else:
        # Validasi statis: hard gap/data/stabilitas/placement tetap wajib ulang.
        # Noise serial tanpa kehilangan frame tidak menggagalkan validasi.
        signal_status = "PASS" if data_quality_pass else "REVIEW"
        postural_status = "PASS" if load_stability_pass else "REVIEW"
        mass_status = "NOT_APPLICABLE"
        if not data_quality_pass or not load_stability_pass or not placement_pass:
            overall_status = prc.TRIAL_STATUS_REPEAT
        elif gap_has_note:
            overall_status = prc.TRIAL_STATUS_NOTE
        else:
            overall_status = prc.TRIAL_STATUS_PASS
        trial_usability = overall_status

    failed_data = [name for name, passed in data_checks.items() if not passed]
    failed_stability = [name for name, passed in stability_checks.items() if not passed]
    failed_placement = [name for name, passed in placement_checks.items() if not passed]

    ratios = {
        "grf_total_cv": metrics["total_cv"] / threshold["total_cv"],
        "grf_side_cv": max(metrics["left_cv"], metrics["right_cv"]) / threshold["side_cv"],
        "cop_drift": max(abs(metrics["ap_slope"]), abs(metrics["ml_slope"])) / threshold["drift_m_s"],
        "endpoint_shift": max(abs(endpoint_shift_ap), abs(endpoint_shift_ml)) / threshold["endpoint_m"],
        "cop_sd": max(metrics["sd_ap"], metrics["sd_ml"]) / threshold["sd_m"],
        "cop_robust_range": max(metrics["range_ap"], metrics["range_ml"]) / threshold["robust_range_m"],
        "mean_velocity": metrics["mean_velocity"] / threshold["mean_velocity_m_s"],
        "block_step": max(metrics["block_step_ap"], metrics["block_step_ml"]) / threshold["block_step_m"],
    }
    near_limit = [
        name for name, ratio in ratios.items()
        if np.isfinite(ratio) and NEAR_LIMIT_FRACTION <= ratio <= 1.0
    ]

    near_limit_details: dict[str, str] = {}
    for name in near_limit:
        if name == "block_step":
            if metrics["block_step_ap"] >= metrics["block_step_ml"]:
                axis = "AP"
                value_m = float(metrics["block_step_ap"])
                time_s = float(metrics["block_step_ap_time_s"])
            else:
                axis = "ML"
                value_m = float(metrics["block_step_ml"])
                time_s = float(metrics["block_step_ml_time_s"])
            near_limit_details[name] = (
                f"Pergeseran CoP antarblok mendekati batas: "
                f"{value_m * 100.0:.3f} cm pada sumbu {axis}, "
                f"batas {threshold['block_step_m'] * 100.0:.3f} cm, "
                f"terjadi sekitar t={time_s:.2f} s dalam window analisis"
            )
        elif name == "endpoint_shift":
            value_m = max(abs(endpoint_shift_ap), abs(endpoint_shift_ml))
            near_limit_details[name] = (
                f"Pergeseran titik awal–akhir {value_m * 100.0:.3f} cm; "
                f"batas {threshold['endpoint_m'] * 100.0:.3f} cm"
            )
        elif name == "cop_drift":
            value_m_s = max(abs(metrics["ap_slope"]), abs(metrics["ml_slope"]))
            near_limit_details[name] = (
                f"Drift CoP {value_m_s * 100.0:.4f} cm/s; "
                f"batas {threshold['drift_m_s'] * 100.0:.4f} cm/s"
            )
        elif name == "cop_sd":
            value_m = max(metrics["sd_ap"], metrics["sd_ml"])
            near_limit_details[name] = (
                f"SD CoP maksimum {value_m * 100.0:.3f} cm; "
                f"batas {threshold['sd_m'] * 100.0:.3f} cm"
            )
        elif name == "cop_robust_range":
            value_m = max(metrics["range_ap"], metrics["range_ml"])
            near_limit_details[name] = (
                f"Robust range maksimum {value_m * 100.0:.3f} cm; "
                f"batas {threshold['robust_range_m'] * 100.0:.3f} cm"
            )
        elif name == "mean_velocity":
            near_limit_details[name] = (
                f"Kecepatan CoP rata-rata {metrics['mean_velocity'] * 100.0:.3f} cm/s; "
                f"batas {threshold['mean_velocity_m_s'] * 100.0:.3f} cm/s"
            )
        elif name == "grf_total_cv":
            near_limit_details[name] = (
                f"CV GRF total {metrics['total_cv'] * 100.0:.3f}%; "
                f"batas {threshold['total_cv'] * 100.0:.3f}%"
            )
        elif name == "grf_side_cv":
            value = max(metrics["left_cv"], metrics["right_cv"])
            near_limit_details[name] = (
                f"CV GRF sisi maksimum {value * 100.0:.3f}%; "
                f"batas {threshold['side_cv'] * 100.0:.3f}%"
            )
        else:
            near_limit_details[name] = prc.humanize_check_name(name)

    mux_skew = np.asarray(data["mux_skew"], dtype=float)
    mux_skew_fraction = (
        float(np.mean(np.abs(mux_skew)) / (1_000_000.0 / fs))
        if len(mux_skew) > 0
        else np.nan
    )
    mux_skew_p95_fraction = (
        float(np.percentile(np.abs(mux_skew), 95) / (1_000_000.0 / fs))
        if len(mux_skew) > 0
        else np.nan
    )
    mux_alignment_coverage = float(data.get("mux_alignment_coverage", np.nan))
    mux_alignment_applied = bool(data.get("mux_alignment_applied", False))

    print("\n" + "=" * 84)
    print(f"{profile.label} REPORT v8.1 — STATIC-REFINED CALIBRATION")
    print("=" * 84)
    print(f"Software Version      : {SOFTWARE_VERSION}")
    print(
        f"Calibration Version   : "
        f"{data.get('calibration_version', config.get('calibration_version', '-'))}"
    )
    zero_gate_info = data.get("zero_gate_info") or {}
    print(
        f"Pre-trial Zero Gate   : "
        f"{zero_gate_info.get('status', 'NOT_AVAILABLE')}"
    )
    print(f"Sampling Rate         : {fs:.2f} Hz")
    print(f"Recorded Duration     : {time_arr[-1] - time_arr[0]:.2f} s")
    if profile.name == "human":
        print(
            f"Fixed Analysis Window : {time_arr[start_idx] - time_arr[0]:.2f}–"
            f"{time_arr[end_idx - 1] - time_arr[0]:.2f} s"
        )
    else:
        print(
            f"Selected Window       : {time_arr[start_idx] - time_arr[0]:.2f}–"
            f"{time_arr[end_idx - 1] - time_arr[0]:.2f} s "
            f"(stability score {selection_score:.2f})"
        )
    print(f"Analysis Duration     : {metrics['duration']:.2f} s")
    print(f"Force-channel Cutoff  : {profile.analysis_cutoff_hz:.1f} Hz")
    print(f"Serial Baud           : {int(data.get('serial_baud', config['baud_rate']))}")
    print(f"Serial Lost (full)    : {serial_diag['serial_lost_frames']}")
    print(f"ADC Rejected record   : {record_adc_rejected}")
    print(f"Malformed record      : {record_malformed}")
    print(f"Frame Gaps record     : {sampling_diag['analysis_frame_gaps']}")
    print(f"Frame Gaps analysis   : {selected_sampling_diag['analysis_frame_gaps']}")
    if len(mux_skew) > 0:
        print(
            f"MUX Skew raw          : mean|skew|={np.mean(np.abs(mux_skew)):.1f} us, "
            f"P95|skew|={np.percentile(np.abs(mux_skew), 95):.1f} us, "
            f"max|skew|={np.max(np.abs(mux_skew)):.1f} us"
        )
        print(
            f"MUX Fraction          : mean={mux_skew_fraction * 100:.1f}% | "
            f"P95={mux_skew_p95_fraction * 100:.1f}% periode frame"
        )
        print(
            f"MUX Alignment         : "
            f"{'APPLIED' if mux_alignment_applied else 'NOT_APPLIED'} | "
            f"coverage={mux_alignment_coverage * 100:.2f}%"
            if np.isfinite(mux_alignment_coverage)
            else f"MUX Alignment         : {'APPLIED' if mux_alignment_applied else 'NOT_APPLIED'}"
        )
    if profile.name == "human":
        print(f"Signal Quality        : {signal_status}")
        print(f"Postural Stability    : {postural_status}")
        print(f"Mass Accuracy         : {mass_status}")
        print(f"Trial Usability       : {trial_usability}")
        if post_window_motion.get("available", False):
            post_status = (
                "MOTION_DETECTED_EXCLUDED"
                if post_window_motion.get("detected", False)
                else "NO_LARGE_MOTION"
            )
        else:
            post_status = "NOT_AVAILABLE"
        print(f"Post-window Monitor   : {post_status}")
        print(
            f"End Anticipation      : "
            f"{'POSSIBLE' if anticipation_info.get('detected', False) else 'NOT_DETECTED'}"
        )
    else:
        print(f"Data/System Quality   : {'PASS' if data_quality_pass else 'REVIEW'}")
        print(f"Motion/Load Stability : {'PASS' if load_stability_pass else 'REVIEW'}")
        print(f"Load Placement        : {'PASS' if placement_pass else 'REVIEW'}")
        print(f"Overall Trial         : {overall_status}")
    print(f"Failed Data Checks    : {', '.join(failed_data) if failed_data else '-'}")
    print(
        f"Failed Stability      : "
        f"{', '.join(failed_stability) if failed_stability else '-'}"
    )
    print(f"Near-limit Checks     : {', '.join(near_limit) if near_limit else '-'}")
    if profile.name != "human":
        print(
            f"Failed Placement      : "
            f"{', '.join(failed_placement) if failed_placement else '-'}"
        )
    print("-" * 84)
    print(f"Measured Mass Total   : {measured_mass:.3f} kg")
    print(
        f"Measured Mass L / R   : {measured_left_mass:.3f} / "
        f"{measured_right_mass:.3f} kg"
    )
    total_side_mass = measured_left_mass + measured_right_mass
    print(
        f"Load Share L / R      : "
        f"{measured_left_mass / total_side_mass * 100:.2f}% / "
        f"{measured_right_mass / total_side_mass * 100:.2f}%"
    )
    if profile.name == "human":
        print(
            f"Vision / Stance Width : {metadata.get('vision_condition') or '-'} / "
            f"{metadata.get('stance_width_cm') if metadata.get('stance_width_cm') is not None else '-'} cm"
        )
        if expected_human_mass is not None:
            print(
                f"Reference Human Mass  : {float(expected_human_mass):.3f} kg "
                f"(error {human_mass_error_percent:.3f}%, "
                f"source={mass_reference_quality})"
            )
    if expected_total_mass is not None:
        print(
            f"Reference Mass Total  : {expected_total_mass:.3f} kg "
            f"(error {total_mass_error_percent:.3f}%)"
        )
        print(
            f"Mass Error L / R      : {left_mass_error_percent:.3f}% / "
            f"{right_mass_error_percent:.3f}%"
        )
    print(
        f"GRF CV total/L/R      : {metrics['total_cv'] * 100:.3f}% / "
        f"{metrics['left_cv'] * 100:.3f}% / "
        f"{metrics['right_cv'] * 100:.3f}%"
    )
    print(f"Hampel Replaced       : {replaced_ratio * 100:.3f}%")
    print("-" * 84)
    print(f"Mean Global AP / ML   : {mean_ap * 100:.3f} / {mean_ml * 100:.3f} cm")
    print(
        f"Local CoP Left AP/ML  : {mean_l_ap * 100:.3f} / "
        f"{mean_l_ml * 100:.3f} cm"
    )
    print(
        f"Local CoP Right AP/ML : {mean_r_ap * 100:.3f} / "
        f"{mean_r_ml * 100:.3f} cm"
    )
    print(
        f"ML center-mass term   : {center_mass_contribution_ml * 100:.3f} cm"
    )
    print(
        f"ML local-offset term  : {local_contribution_ml * 100:.3f} cm"
    )
    print(
        f"ML reconstructed      : {reconstructed_ml * 100:.3f} cm "
        f"(residual {reconstruction_error_ml * 100:.5f} cm)"
    )
    print(
        f"AP reconstructed      : {reconstructed_ap * 100:.3f} cm "
        f"(residual {reconstruction_error_ap * 100:.5f} cm)"
    )
    if reference_center_ml is not None:
        print(
            f"Reference center-only : AP=0.000 cm, "
            f"ML={reference_center_ml * 100:.3f} cm"
        )
        if not centering_pass:
            print(
                "Reference CoP status   : NOT EVALUATED — beban tidak cukup "
                "terpusat pada koordinat lokal plate."
            )
        elif profile.name == "static_liquid":
            print(
                "Reference CoP status   : INFORMATIONAL — cairan dapat "
                "memindahkan pusat massa internal wadah."
            )
    print("-" * 84)
    print(
        f"Sway SD AP / ML       : {metrics['sd_ap'] * 100:.3f} / "
        f"{metrics['sd_ml'] * 100:.3f} cm"
    )
    print(
        f"Detrended SD AP / ML  : {detrended_sd_ap * 100:.3f} / "
        f"{detrended_sd_ml * 100:.3f} cm"
    )
    print(f"AP/ML SD Ratio        : {metrics['anisotropy_ratio']:.2f}x")
    print(
        f"Robust Range AP / ML  : {metrics['range_ap'] * 100:.3f} / "
        f"{metrics['range_ml'] * 100:.3f} cm"
    )
    print(
        f"Endpoint Median AP    : {endpoint_ap_start * 100:.3f} -> "
        f"{endpoint_ap_end * 100:.3f} cm"
    )
    print(
        f"Endpoint Median ML    : {endpoint_ml_start * 100:.3f} -> "
        f"{endpoint_ml_end * 100:.3f} cm"
    )
    print(
        f"Endpoint Shift AP/ML  : {endpoint_shift_ap * 100:.3f} / "
        f"{endpoint_shift_ml * 100:.3f} cm"
    )
    print(
        f"Max 0.5s Step AP/ML   : {metrics['block_step_ap'] * 100:.3f} / "
        f"{metrics['block_step_ml'] * 100:.3f} cm"
    )
    print(
        f"Largest AP Step       : t={metrics['block_step_ap_time_s']:.2f} s, "
        f"{metrics['block_step_ap_before'] * 100:.3f} -> "
        f"{metrics['block_step_ap_after'] * 100:.3f} cm"
    )
    print(
        f"Largest ML Step       : t={metrics['block_step_ml_time_s']:.2f} s, "
        f"{metrics['block_step_ml_before'] * 100:.3f} -> "
        f"{metrics['block_step_ml_after'] * 100:.3f} cm"
    )
    print(f"Path Length           : {metrics['path_length']:.4f} m")
    print(f"Mean Velocity         : {metrics['mean_velocity'] * 100:.3f} cm/s")
    print(f"95% Ellipse Area      : {ellipse_area * 10000:.3f} cm²")
    print(f"Detrended Ellipse     : {detrended_ellipse_area * 10000:.3f} cm²")
    print(f"Ellipse Direction     : {ellipse_angle:.1f}°")
    print(
        f"CoP Drift AP / ML     : {metrics['ap_slope'] * 100:.4f} / "
        f"{metrics['ml_slope'] * 100:.4f} cm/s"
    )
    print(
        f"PSD Peak AP           : {peak_ap_hz:.2f} Hz "
        f"(narrow-band {narrow_ap_ratio * 100:.1f}%)"
    )
    print(
        f"PSD Peak ML           : {peak_ml_hz:.2f} Hz "
        f"(narrow-band {narrow_ml_ratio * 100:.1f}%)"
    )
    print(f"Physical Bounds       : {'PASS' if bounds_pass else 'FAIL'}")
    print("Mean channel forces   :")
    print(
        "  "
        + ", ".join(
            f"{name}={value:.2f} N"
            for name, value in zip(prc.SENSOR_NAMES, channel_mean_n)
        )
    )
    if profile.name == "human":
        print("5-second block summary:")
        print("  Block(s)   Mean AP   Mean ML    SD AP    SD ML")
        for item in block_summaries:
            print(
                f"  {item['start_s']:>4.0f}-{item['end_s']:<4.0f} "
                f"{item['mean_ap'] * 100:>8.3f} "
                f"{item['mean_ml'] * 100:>8.3f} "
                f"{item['sd_ap'] * 100:>8.3f} "
                f"{item['sd_ml'] * 100:>8.3f} cm"
            )
        if post_window_motion.get("available", False):
            print(
                f"Post-window Max Shift : t={post_window_motion['time_s']:.2f} s, "
                f"AP={post_window_motion['shift_ap'] * 100:.3f} cm, "
                f"ML={post_window_motion['shift_ml'] * 100:.3f} cm "
                "(excluded from main analysis)"
            )
    print("=" * 84)

    if profile.name == "human":
        if low_frequency_dominance:
            print(
                "[INFO] Sway didominasi frekuensi rendah. Ini dapat berasal dari "
                "koreksi postur lambat, pernapasan, atau perpindahan tekanan "
                "berulang; bukan otomatis noise alat."
            )
        if metrics["anisotropy_ratio"] >= 8.0:
            dominant_axis = "AP" if metrics["sd_ap"] >= metrics["sd_ml"] else "ML"
            print(
                f"[INFO] Gerakan sangat dominan pada sumbu {dominant_axis} "
                f"({metrics['anisotropy_ratio']:.1f}x). Pastikan arah jari kaki "
                "sesuai FRONT, posisi kaki simetris, dan subjek tidak sengaja "
                "mengayun ke depan-belakang."
            )
        if not stability_checks["block_step"]:
            dominant_step_axis = (
                "AP" if metrics["block_step_ap"] >= metrics["block_step_ml"] else "ML"
            )
            step_time = (
                metrics["block_step_ap_time_s"]
                if dominant_step_axis == "AP"
                else metrics["block_step_ml_time_s"]
            )
            print(
                f"[WARNING] Terdeteksi koreksi postur mendadak pada sumbu "
                f"{dominant_step_axis} sekitar detik {step_time:.2f} dalam window. "
                "Trial sebaiknya diulang bila gerakan tersebut bukan bagian dari "
                "protokol."
            )
        if anticipation_info.get("detected", False):
            print(
                f"[INFO] Lima detik terakhir window menunjukkan kemungkinan "
                f"anticipatory movement dominan pada sumbu "
                f"{anticipation_info.get('axis', '-')}. Jangan beri tahu subjek "
                "detik pasti berakhirnya window; gunakan bunyi akhir otomatis."
            )
        if post_window_motion.get("detected", False):
            print(
                f"[INFO] Gerakan setelah window utama terdeteksi sekitar "
                f"detik {post_window_motion['time_s']:.2f}; perubahan ini "
                "dikeluarkan dari metrik quiet standing."
            )
        if human_mass_error_percent is not None and not human_mass_accuracy_pass:
            if mass_reference_quality == "measured":
                print(
                    "[WARNING] Massa force plate berbeda lebih dari 3% terhadap "
                    "massa yang diukur langsung. Uji benda padat sekitar massa "
                    "subjek sebelum mengubah faktor kalibrasi."
                )
            else:
                print(
                    "[INFO] Massa referensi ditandai sebagai perkiraan, sehingga "
                    "selisih massa hanya informasional dan tidak menggagalkan "
                    "usability trial."
                )
        if trial_usability == prc.TRIAL_STATUS_REPEAT:
            print(
                "[REPEAT REQUIRED] Data/sinyal atau gerakan besar tidak memenuhi "
                "batas core. Jangan gunakan trial ini sebagai data utama."
            )
        elif trial_usability == prc.TRIAL_STATUS_NOTE:
            print(
                "[USABLE WITH NOTE] Data tetap dapat digunakan; catatan postural/massa "
                "tidak mewajibkan pengulangan."
            )
    elif profile.name == "static_liquid" and narrowband_detected:
        print(
            "[INTERPRETATION] Gerakan narrow-band pada beban berisi cairan "
            "lebih tepat diklasifikasikan sebagai gerakan objek/cairan atau "
            "resonansi mekanik, bukan langsung sebagai error kalibrasi. Ulangi "
            "dengan beban padat untuk memisahkan kedua sumber tersebut."
        )
    elif narrowband_detected:
        print(
            "[WARNING] Osilasi narrow-band pada beban kaku mencurigakan. "
            "Periksa rocking plate, baut, stopper, kabel, meja/lantai, dan catu daya."
        )

    if profile.name != "human" and not centering_pass:
        print(
            "[WARNING] Perbedaan global CoP terhadap nilai center-only terutama "
            "berasal dari posisi lokal beban, sehingga tidak boleh langsung "
            "dianggap sebagai pembalikan tanda ML atau error kalibrasi."
        )

    if profile.name == "static_liquid":
        print(
            "[INFO] Window dipilih otomatis dari rekaman panjang. Waktu awal "
            "window menunjukkan lamanya program menunggu bagian rekaman yang "
            "paling stabil setelah galon diletakkan."
        )

    total_side_mass = measured_left_mass + measured_right_mass
    left_share = (
        measured_left_mass / total_side_mass * 100.0
        if total_side_mass > 0.0 else np.nan
    )
    right_share = (
        measured_right_mass / total_side_mass * 100.0
        if total_side_mass > 0.0 else np.nan
    )
    placement_status = (
        "NOT_APPLICABLE"
        if profile.name == "human"
        else "PASS" if placement_pass else "REVIEW"
    )
    mux_skew_mean_us = float(np.mean(mux_skew)) if len(mux_skew) else np.nan
    mux_skew_p95_us = (
        float(np.percentile(np.abs(mux_skew), 95))
        if len(mux_skew) else np.nan
    )

    result = {
        "software_version": SOFTWARE_VERSION,
        "calibration_version": str(
            data.get("calibration_version", config.get("calibration_version", "-"))
        ),
        "zero_gate_info": data.get("zero_gate_info"),
        "mux_alignment_applied": bool(data.get("mux_alignment_applied", False)),
        "mux_alignment_coverage": float(data.get("mux_alignment_coverage", np.nan)),
        "profile_name": profile.name,
        "profile_label": profile.label,
        "overall_status": overall_status,
        "trial_usability": trial_usability,
        "signal_status": signal_status,
        "postural_status": postural_status,
        "mass_status": mass_status,
        "placement_status": placement_status,
        "measured_mass": measured_mass,
        "measured_left_mass": measured_left_mass,
        "measured_right_mass": measured_right_mass,
        "left_share": left_share,
        "right_share": right_share,
        "mean_ap": mean_ap,
        "mean_ml": mean_ml,
        "ellipse_area": ellipse_area,
        "ellipse_angle": ellipse_angle,
        "metrics": metrics,
        "fs": fs,
        "analysis_duration": metrics["duration"],
        "analysis_start_s": float(time_arr[start_idx] - time_arr[0]),
        "analysis_end_s": float(time_arr[end_idx - 1] - time_arr[0]),
        "peak_ap_hz": peak_ap_hz,
        "peak_ml_hz": peak_ml_hz,
        "serial_lost_frames": serial_diag["serial_lost_frames"],
        "record_adc_rejected": record_adc_rejected,
        "record_malformed": record_malformed,
        "record_gap_status": record_gap_status,
        "selected_gap_status": selected_gap_status,
        "record_gap_reasons": record_gap_reasons,
        "selected_gap_reasons": selected_gap_reasons,
        "analysis_frame_gaps": sampling_diag["analysis_frame_gaps"],
        "selected_frame_gaps": selected_sampling_diag["analysis_frame_gaps"],
        "mux_skew_mean_us": mux_skew_mean_us,
        "mux_skew_p95_us": mux_skew_p95_us,
        "mux_skew_mean_fraction": mux_skew_fraction,
        "mux_skew_p95_fraction": mux_skew_p95_fraction,
        "failed_data": failed_data,
        "failed_stability": failed_stability,
        "failed_placement": failed_placement,
        "near_limit": near_limit,
        "near_limit_details": near_limit_details,
        "near_limit_fraction": NEAR_LIMIT_FRACTION,
        "block_step_limit_m": float(threshold["block_step_m"]),
        "post_window_motion": post_window_motion,
        "anticipation_info": anticipation_info,
        "channel_mean_n": channel_mean_n,
        "mass_error_percent": human_mass_error_percent,
        "mass_reference_quality": mass_reference_quality,
    }

    physical_bounds = physical_bounds_cm(config)
    extra_points = []
    if reference_center_ml is not None:
        extra_points.append((reference_center_ml * 100.0, 0.0))
    zoom_bounds = make_zoom_bounds(
        ml_cm, ap_cm, physical_bounds, extra_points=extra_points
    )
    try:
        X, Y, Z = compute_relative_heatmap(ml_cm, ap_cm, zoom_bounds)
    except ValueError as exc:
        print(f"[WARNING] Heatmap gagal: {exc}")
        X = Y = Z = None

    # Grafik dipisahkan menjadi dua figure agar setiap panel tetap besar dan
    # terbaca pada layar 1366–1920 px. Perhitungan metrik tidak berubah.
    full_time_rel = time_arr - time_arr[0]

    overview_figure, overview_axes = plt.subplots(2, 1, figsize=(12.4, 6.4), dpi=100)
    overview_figure.patch.set_facecolor("white")
    ax_time, ax_psd = overview_axes

    ax_time.plot(full_time_rel, cop_ap_filtered_all * 100.0, label="AP")
    ax_time.plot(full_time_rel, cop_ml_filtered_all * 100.0, label="ML")
    ax_time.axvspan(
        full_time_rel[start_idx], full_time_rel[end_idx - 1],
        alpha=0.15, label="Window analisis",
    )
    if profile.name == "human" and end_idx < len(full_time_rel):
        ax_time.axvspan(
            full_time_rel[end_idx], full_time_rel[-1],
            alpha=0.08, hatch="//", label="Post-window monitor",
        )
    if profile.name == "human" and np.isfinite(metrics["block_step_ap_time_s"]):
        event_time_full = full_time_rel[start_idx] + metrics["block_step_ap_time_s"]
        ax_time.axvline(event_time_full, linestyle="--", linewidth=1.0, label="Largest AP shift")
    if (
        profile.name == "human"
        and post_window_motion.get("detected", False)
        and np.isfinite(post_window_motion.get("time_s", np.nan))
    ):
        ax_time.axvline(
            float(post_window_motion["time_s"]), linestyle=":", linewidth=1.2,
            label="Post-window motion",
        )
    ax_time.set_title("CoP Rekaman Penuh dan Window Analisis")
    ax_time.set_xlabel("Waktu rekaman (s)")
    ax_time.set_ylabel("Posisi (cm)")
    ax_time.grid(True)
    ax_time.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=5, framealpha=0.92)

    ax_psd.plot(f_ap[frequency_band], pxx_ap[frequency_band], label="AP")
    ax_psd.plot(f_ml[frequency_band], pxx_ml[frequency_band], label="ML")
    if np.isfinite(peak_ap_hz):
        ax_psd.axvline(peak_ap_hz, linestyle=":", linewidth=1.0, label=f"Peak AP {peak_ap_hz:.2f} Hz")
    if np.isfinite(peak_ml_hz):
        ax_psd.axvline(peak_ml_hz, linestyle=":", linewidth=1.0, label=f"Peak ML {peak_ml_hz:.2f} Hz")
    ax_psd.set_title("PSD Welch — Window Analisis")
    ax_psd.set_xlabel("Frekuensi (Hz)")
    ax_psd.set_ylabel("Power (cm²/Hz)")
    ax_psd.grid(True)
    ax_psd.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.92)

    for axis in overview_axes:
        axis.set_title(axis.get_title(), fontsize=11, pad=9)
        axis.tick_params(axis="both", labelsize=8, pad=3)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
    overview_figure.subplots_adjust(
        left=0.075, right=0.975, bottom=0.11, top=0.91, hspace=0.48
    )

    spatial_figure, spatial_axes = plt.subplots(1, 2, figsize=(12.4, 6.4), dpi=100, gridspec_kw={"width_ratios": [1.05, 1.0]})
    spatial_figure.patch.set_facecolor("white")
    spatial_figure.suptitle("CoP Spatial — Window Analisis", fontsize=12, fontweight="bold")
    ax_heatmap, ax_path = spatial_axes

    if Z is not None:
        heatmap = ax_heatmap.contourf(X, Y, Z, levels=np.linspace(0, 100, 21))
        ax_heatmap.plot(ml_cm, ap_cm, linewidth=0.4, alpha=0.35)
        ax_heatmap.scatter([mean_ml * 100.0], [mean_ap * 100.0], marker="x", s=80, label="Mean CoP")
        colorbar = spatial_figure.colorbar(heatmap, ax=ax_heatmap, fraction=0.045, pad=0.035)
        colorbar.set_label("Kepadatan relatif (% puncak)", fontsize=9, labelpad=8)
        colorbar.ax.tick_params(labelsize=8)
    if reference_center_ml is not None:
        ax_heatmap.scatter([reference_center_ml * 100.0], [0.0], marker="+", s=100, label="Center-only reference")
    add_plate_outlines(ax_heatmap, config)
    ax_heatmap.set_title("Heatmap Kepadatan CoP")
    ax_heatmap.set_xlabel("ML (cm)")
    ax_heatmap.set_ylabel("AP (cm)")
    ax_heatmap.set_xlim(zoom_bounds[0], zoom_bounds[1])
    ax_heatmap.set_ylim(zoom_bounds[2], zoom_bounds[3])
    ax_heatmap.set_aspect("equal", adjustable="box")
    ax_heatmap.grid(True)
    ax_heatmap.legend(fontsize=8, loc="upper right", framealpha=0.92)

    ax_path.plot(ml_cm, ap_cm, linewidth=0.8, alpha=0.80, label="CoP Path")
    if np.all(np.isfinite([major_axis, minor_axis, ellipse_angle])):
        ax_path.add_patch(
            Ellipse(
                xy=(mean_ml * 100.0, mean_ap * 100.0),
                width=major_axis * 100.0,
                height=minor_axis * 100.0,
                angle=ellipse_angle,
                fill=False,
                linewidth=2,
                linestyle="--",
                label="95% Ellipse",
            )
        )
    ax_path.scatter([ml_cm[0]], [ap_cm[0]], marker="o", label="Start")
    ax_path.scatter([ml_cm[-1]], [ap_cm[-1]], marker="s", label="End")
    ax_path.scatter([mean_ml * 100.0], [mean_ap * 100.0], marker="x", s=80, label="Mean")
    if reference_center_ml is not None:
        ax_path.scatter([reference_center_ml * 100.0], [0.0], marker="+", s=100, label="Center-only reference")
    add_plate_outlines(ax_path, config)
    ax_path.set_title("Statokinesigram CoP")
    ax_path.set_xlabel("ML (cm)")
    ax_path.set_ylabel("AP (cm)")
    ax_path.set_xlim(zoom_bounds[0], zoom_bounds[1])
    ax_path.set_ylim(zoom_bounds[2], zoom_bounds[3])
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True)
    ax_path.legend(fontsize=7.5, loc="best", framealpha=0.92)

    for axis in spatial_axes:
        axis.set_title(axis.get_title(), fontsize=11, pad=10)
        axis.tick_params(axis="both", labelsize=8, pad=3)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
    spatial_figure.subplots_adjust(
        left=0.075, right=0.94, bottom=0.11, top=0.86, wspace=0.34
    )

    overview_png = Path(data["filename"]).with_suffix("").with_name(
        Path(data["filename"]).stem + "_standing_overview_v85.png"
    )
    spatial_png = Path(data["filename"]).with_suffix("").with_name(
        Path(data["filename"]).stem + "_standing_spatial_v85.png"
    )
    overview_figure.savefig(overview_png, dpi=200, bbox_inches="tight")
    spatial_figure.savefig(spatial_png, dpi=200, bbox_inches="tight")
    print(f"[OK] Grafik overview tersimpan: {overview_png}")
    print(f"[OK] Grafik spatial tersimpan: {spatial_png}")

    summary_csv = write_standing_summary_csv(result, data)
    figures = {"overview": overview_figure, "spatial": spatial_figure}
    output_paths = {"overview": overview_png, "spatial": spatial_png}
    repeat_requested = show_standing_result_window(result, data, figures, summary_csv, output_paths)
    for figure in figures.values():
        plt.close(figure)
    return 2 if repeat_requested else 0


if __name__ == "__main__":
    try:
        exit_code = main()
        while exit_code == 2:
            print("\n[REPEAT] Memulai pengukuran standing baru.")
            exit_code = main()
        raise SystemExit(exit_code)
    except KeyboardInterrupt:
        print("\n[SELESAI] Program dihentikan oleh pengguna.")
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"[ERROR INTERNAL] Modul STANDING dihentikan dengan aman: "
            f"{type(exc).__name__}: {exc}"
        )
        raise SystemExit(1)


