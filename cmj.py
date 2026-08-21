from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

import processing as prc


SOFTWARE_VERSION = "8.5_CMJ_CRC_FRAMED_GAP_ROBUST"
CMJ_FILTER_CUTOFF_HZ = 30.0
EVENT_FILTER_CUTOFF_HZ = 45.0
IMPACT_FILTER_CUTOFF_HZ = 60.0
BASELINE_WINDOW_S = 1.25
BASELINE_SEARCH_END_S = 8.0
EVENT_MEDIAN_WINDOW_S = 0.015
EVENT_PERSISTENCE_S = 0.025
UNWEIGHT_PERSISTENCE_S = 0.030
POST_LANDING_STABLE_WINDOW_S = 0.80
POST_LANDING_SEARCH_DELAY_S = 0.60
POST_STABILITY_FILTER_HZ = 10.0
TERMINAL_VELOCITY_WINDOW_S = 0.50
PRIMARY_METHOD = "IMPULSE_MOMENTUM"


def trapezoid_integral(y, x) -> float:
    values = np.asarray(y, dtype=float)
    times = np.asarray(x, dtype=float)
    if len(values) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, times))
    return float(np.trapz(values, times))


def integrate_between(time_arr, signal, start_time, end_time) -> float:
    times = np.asarray(time_arr, dtype=float)
    values = np.asarray(signal, dtype=float)

    if len(times) != len(values):
        raise ValueError("Waktu dan sinyal tidak sejajar.")
    if not start_time < end_time:
        raise ValueError("Batas integrasi tidak valid.")
    if start_time < times[0] or end_time > times[-1]:
        raise ValueError("Batas integrasi berada di luar data.")

    inside = (times > start_time) & (times < end_time)
    segment_time = np.concatenate(([start_time], times[inside], [end_time]))
    segment_signal = np.concatenate(
        (
            [np.interp(start_time, times, values)],
            values[inside],
            [np.interp(end_time, times, values)],
        )
    )
    return trapezoid_integral(segment_signal, segment_time)


def cumulative_integral_from_time(time_arr, signal, start_time) -> np.ndarray:
    """Integral kumulatif dengan batas awal yang diinterpolasi tepat."""
    times = np.asarray(time_arr, dtype=float)
    values = np.asarray(signal, dtype=float)
    output = np.full(len(times), np.nan, dtype=float)

    start_index = int(np.searchsorted(times, start_time, side="right"))
    if start_index >= len(times):
        return output

    segment_time = np.concatenate(([start_time], times[start_index:]))
    segment_values = np.concatenate(
        ([np.interp(start_time, times, values)], values[start_index:])
    )
    integrated = cumulative_trapezoid(
        segment_values,
        x=segment_time,
        initial=0.0,
    )
    output[start_index:] = integrated[1:]
    return output


def interpolate_crossing(time_arr, signal, idx, threshold) -> float:
    times = np.asarray(time_arr, dtype=float)
    values = np.asarray(signal, dtype=float)

    if idx <= 0:
        return float(times[idx])

    t0 = float(times[idx - 1])
    t1 = float(times[idx])
    y0 = float(values[idx - 1])
    y1 = float(values[idx])

    if abs(y1 - y0) < 1e-12:
        return t1

    fraction = float(np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0))
    return t0 + fraction * (t1 - t0)


def find_downward_crossing(time_arr, signal, threshold, start_idx, end_idx):
    values = np.asarray(signal, dtype=float)
    for i in range(max(1, start_idx), min(end_idx, len(values))):
        if values[i - 1] >= threshold and values[i] < threshold:
            return interpolate_crossing(time_arr, values, i, threshold)
    return None


def find_upward_crossing(time_arr, signal, threshold, start_idx, end_idx):
    values = np.asarray(signal, dtype=float)
    for i in range(max(1, start_idx), min(end_idx, len(values))):
        if values[i - 1] <= threshold and values[i] > threshold:
            return interpolate_crossing(time_arr, values, i, threshold)
    return None


def find_last_upward_crossing(time_arr, signal, threshold, start_idx, end_idx):
    """Crossing naik terakhir pada interval, untuk estimasi kontak setelah konfirmasi."""
    values = np.asarray(signal, dtype=float)
    stop = min(int(end_idx), len(values))
    for i in range(stop - 1, max(1, int(start_idx)) - 1, -1):
        if values[i - 1] <= threshold and values[i] > threshold:
            return interpolate_crossing(time_arr, values, i, threshold), i
    return None, None


def persistent_first(values, condition, start_idx, persistence_samples, end_idx=None):
    array = np.asarray(values)
    stop = len(array) if end_idx is None else min(len(array), int(end_idx))
    count = 0
    for i in range(max(0, int(start_idx)), stop):
        if condition(array[i]):
            count += 1
            if count >= persistence_samples:
                return i - persistence_samples + 1
        else:
            count = 0
    return None



def detect_side_landing_onset(
    side_force: np.ndarray,
    time_arr: np.ndarray,
    takeoff_idx: int,
    landing_confirmation_idx: int,
    fs: float,
    threshold_n: float,
    persistence_s: float,
):
    """Deteksi first-contact satu sisi pada sinyal yang sudah MUX-aligned.

    Deteksi bilateral ini tidak menggantikan landing total. Tujuannya untuk
    menjelaskan apakah dua puncak landing berasal dari kaki kiri/kanan yang
    menyentuh plate secara bertahap.
    """
    force = np.asarray(side_force, dtype=float)
    times = np.asarray(time_arr, dtype=float)
    persistence = max(2, int(round(float(fs) * float(persistence_s))))
    search_end = min(
        len(force),
        int(landing_confirmation_idx) + max(persistence * 4, int(round(0.20 * fs))),
    )

    idx = persistent_first(
        force,
        lambda value: value > float(threshold_n),
        int(takeoff_idx) + 1,
        persistence,
        end_idx=search_end,
    )
    if idx is None:
        return np.nan, None

    crossing = find_upward_crossing(
        times,
        force,
        float(threshold_n),
        int(takeoff_idx) + 1,
        min(len(force), int(idx) + persistence + 3),
    )
    if crossing is None:
        crossing = float(times[int(idx)])

    return float(crossing), int(idx)


def _window_slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 3 or times[-1] <= times[0]:
        return np.inf
    return float(np.polyfit(times - times[0], values, 1)[0])


def find_stable_baseline(
    force,
    time_arr,
    fs,
    search_end_time: float | None = None,
    prefer_latest: bool = False,
):
    """Cari window baseline stabil.

    Pass pertama memakai window stabil awal untuk mendeteksi gerakan kasar.
    Pass kedua memilih window stabil terakhir yang selesai sebelum gerakan.
    Pendekatan dua-pass mengurangi bias bila sensor/subjek mengalami settling
    lambat selama beberapa detik sebelum lompatan.
    """
    values = np.asarray(force, dtype=float)
    times = np.asarray(time_arr, dtype=float)

    window_samples = max(10, int(round(fs * BASELINE_WINDOW_S)))
    search_start = max(0, int(round(0.20 * fs)))
    if search_end_time is None:
        end_time = min(float(times[-1]), BASELINE_SEARCH_END_S)
    else:
        end_time = min(float(times[-1]), float(search_end_time))
    search_end = min(
        len(values) - window_samples,
        int(np.searchsorted(times, end_time, side="right")) - window_samples,
    )
    if search_end <= search_start:
        return None

    candidates: list[dict[str, float | int]] = []
    for start in range(search_start, search_end + 1):
        end = start + window_samples
        segment = values[start:end]
        segment_time = times[start:end]
        if not np.all(np.isfinite(segment)):
            continue

        mean_force = float(np.mean(segment))
        median_force = float(np.median(segment))
        sd_force = float(np.std(segment, ddof=0))
        mad_force = float(np.median(np.abs(segment - median_force)))
        robust_sd = max(1.4826 * mad_force, 1e-9)
        if median_force <= 100.0:
            continue

        cv = sd_force / max(mean_force, 1e-9)
        robust_cv = robust_sd / max(median_force, 1e-9)
        slope = _window_slope(segment_time, segment)
        robust_range = float(np.percentile(segment, 95) - np.percentile(segment, 5))
        if cv <= 0.015 and abs(slope) <= 8.0:
            candidates.append({
                "start": int(start),
                "end": int(end),
                "start_time": float(segment_time[0]),
                "end_time": float(segment_time[-1]),
                "mean": mean_force,
                "median": median_force,
                "sd": sd_force,
                "mad": mad_force,
                "robust_sd": robust_sd,
                "cv": cv,
                "robust_cv": robust_cv,
                "slope": slope,
                "robust_range": robust_range,
                "score": cv + abs(slope) / max(median_force, 1.0),
            })

    if not candidates:
        return None
    if prefer_latest:
        # Utamakan window paling dekat dengan movement onset; jika beberapa
        # berakhir hampir bersamaan, pilih yang skornya lebih baik.
        candidates.sort(key=lambda item: (float(item["end_time"]), -float(item["score"])))
        return candidates[-1]
    return min(candidates, key=lambda item: float(item["score"]))


def detect_unweighting_and_movement(
    grf: np.ndarray,
    time_arr: np.ndarray,
    fs: float,
    baseline: dict[str, Any],
):
    # Body weight memakai median baseline dan robust SD agar lebih tahan
    # terhadap transien singkat. Mean/SD tetap disimpan untuk audit.
    body_weight = float(baseline.get("median", baseline["mean"]))
    body_weight_sd = float(baseline.get("robust_sd", baseline["sd"]))
    baseline_end_idx = int(baseline["end"])

    unweight_drop = max(5.0 * body_weight_sd, 0.03 * body_weight, 15.0)
    unweight_threshold = body_weight - unweight_drop
    unweight_persistence = max(4, int(round(fs * UNWEIGHT_PERSISTENCE_S)))
    unweighting_idx = persistent_first(
        grf,
        lambda value: value < unweight_threshold,
        baseline_end_idx,
        unweight_persistence,
    )
    if unweighting_idx is None:
        raise ValueError("Unweighting tidak ditemukan.")

    onset_drop = max(5.0 * body_weight_sd, 0.010 * body_weight, 8.0)
    onset_threshold = body_weight - onset_drop
    onset_persistence = max(4, int(round(fs * 0.030)))
    onset_candidate_idx = persistent_first(
        grf,
        lambda value: value < onset_threshold,
        baseline_end_idx,
        onset_persistence,
        end_idx=unweighting_idx + unweight_persistence + 1,
    )
    if onset_candidate_idx is None:
        onset_candidate_idx = unweighting_idx

    movement_start_time = find_downward_crossing(
        time_arr,
        grf,
        onset_threshold,
        baseline_end_idx,
        onset_candidate_idx + onset_persistence + 2,
    )
    if movement_start_time is None:
        movement_start_time = float(
            time_arr[max(baseline_end_idx, onset_candidate_idx)]
        )
    movement_start_idx = int(
        np.searchsorted(time_arr, movement_start_time, side="left")
    )
    return {
        "body_weight": body_weight,
        "body_weight_sd": body_weight_sd,
        "unweight_threshold": unweight_threshold,
        "unweighting_idx": int(unweighting_idx),
        "unweight_persistence": int(unweight_persistence),
        "onset_threshold": onset_threshold,
        "movement_start_time": movement_start_time,
        "movement_start_idx": movement_start_idx,
    }

def _stability_metrics_for_slice(
    total: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    cop_ap: np.ndarray,
    cop_ml: np.ndarray,
    times: np.ndarray,
    start: int,
    end: int,
):
    if end - start < 10:
        return None

    t = times[start:end]
    total_seg = total[start:end]
    left_seg = left[start:end]
    right_seg = right[start:end]
    ap_seg = cop_ap[start:end]
    ml_seg = cop_ml[start:end]

    arrays = (t, total_seg, left_seg, right_seg, ap_seg, ml_seg)
    if any(not np.all(np.isfinite(array)) for array in arrays):
        return None

    total_mean = float(np.mean(total_seg))
    left_mean = float(np.mean(left_seg))
    right_mean = float(np.mean(right_seg))
    if total_mean <= 0.0 or left_mean <= 20.0 or right_mean <= 20.0:
        return None

    return {
        "start": int(start),
        "end": int(end),
        "start_time": float(t[0]),
        "end_time": float(t[-1]),
        "anchor_time": float(np.mean(t)),
        "duration_s": float(t[-1] - t[0]),
        "mean": total_mean,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "total_cv": float(np.std(total_seg, ddof=0) / total_mean),
        "left_cv": float(np.std(left_seg, ddof=0) / left_mean),
        "right_cv": float(np.std(right_seg, ddof=0) / right_mean),
        "total_slope": _window_slope(t, total_seg),
        "left_slope": _window_slope(t, left_seg),
        "right_slope": _window_slope(t, right_seg),
        "cop_ap_range": float(np.percentile(ap_seg, 95) - np.percentile(ap_seg, 5)),
        "cop_ml_range": float(np.percentile(ml_seg, 95) - np.percentile(ml_seg, 5)),
        "cop_ap_slope": _window_slope(t, ap_seg),
        "cop_ml_slope": _window_slope(t, ml_seg),
    }


def _post_window_is_stable(metrics: dict[str, float], body_weight: float, config):
    if metrics is None:
        return False

    mass_band_fraction = 0.08
    stable_ok = bool(
        abs(float(metrics["mean"]) - body_weight) <= mass_band_fraction * body_weight
        and float(metrics["total_cv"]) <= float(config["cmj_post_total_cv_limit"])
        and float(metrics["left_cv"]) <= float(config["cmj_post_side_cv_limit"])
        and float(metrics["right_cv"]) <= float(config["cmj_post_side_cv_limit"])
        and abs(float(metrics["total_slope"]))
        <= float(config["cmj_post_total_slope_limit_n_s"])
        and abs(float(metrics["left_slope"]))
        <= float(config["cmj_post_side_slope_limit_n_s"])
        and abs(float(metrics["right_slope"]))
        <= float(config["cmj_post_side_slope_limit_n_s"])
        and float(metrics["cop_ap_range"])
        <= float(config["cmj_post_cop_range_limit_m"])
        and float(metrics["cop_ml_range"])
        <= float(config["cmj_post_cop_range_limit_m"])
        and abs(float(metrics["cop_ap_slope"]))
        <= float(config["cmj_post_cop_slope_limit_m_s"])
        and abs(float(metrics["cop_ml_slope"]))
        <= float(config["cmj_post_cop_slope_limit_m_s"])
    )
    return stable_ok


def find_post_landing_stable_window(
    force,
    left_force,
    right_force,
    cop_ap,
    cop_ml,
    time_arr,
    landing_time,
    body_weight,
    fs,
    config,
    live_window_start_s=None,
    live_window_end_s=None,
):
    """Cari window post-landing yang stabil secara total, bilateral, dan CoP.

    v5.3 memakai parameter yang sama dengan akuisisi live, sinyal CoP yang
    dihitung dari kanal terfilter, dan tail capture. Kandidat live diperiksa
    terlebih dahulu agar keputusan online dan offline konsisten.
    """
    total = np.asarray(force, dtype=float)
    left = np.asarray(left_force, dtype=float)
    right = np.asarray(right_force, dtype=float)
    ap = np.asarray(cop_ap, dtype=float)
    ml = np.asarray(cop_ml, dtype=float)
    times = np.asarray(time_arr, dtype=float)

    window_s = float(config.get("cmj_post_stable_window_s", POST_LANDING_STABLE_WINDOW_S))
    window_samples = max(12, int(np.ceil(window_s * fs)))
    search_delay_s = float(
        config.get("cmj_post_search_delay_s", POST_LANDING_SEARCH_DELAY_S)
    )
    minimum_post_s = float(config.get("cmj_post_landing_min_s", 2.5))
    # Window boleh mulai sebelum batas minimum, tetapi ujungnya harus mencapai
    # sedikitnya minimum_post_s setelah landing, sama seperti keputusan live.
    search_start_offset_s = max(search_delay_s, minimum_post_s - window_s)
    search_start_time = landing_time + search_start_offset_s
    search_start_idx = int(np.searchsorted(times, search_start_time, side="left"))

    candidate_starts: list[int] = []
    if (
        (live_window_start_s is None or not np.isfinite(live_window_start_s))
        and live_window_end_s is not None
        and np.isfinite(live_window_end_s)
    ):
        live_window_start_s = float(live_window_end_s) - window_s

    if live_window_start_s is not None and np.isfinite(live_window_start_s):
        live_idx = int(np.searchsorted(times, float(live_window_start_s), side="left"))
        radius = max(1, int(round(0.20 * fs)))
        low = max(search_start_idx, live_idx - radius)
        high = min(len(total) - window_samples, live_idx + radius)
        candidate_starts.extend(range(low, high + 1))

    candidate_starts.extend(
        range(search_start_idx, max(search_start_idx, len(total) - window_samples + 1))
    )

    # Hilangkan duplikasi sambil mempertahankan urutan kandidat live lebih dulu.
    seen: set[int] = set()
    ordered_starts = []
    for value in candidate_starts:
        if value not in seen:
            seen.add(value)
            ordered_starts.append(value)

    best = None
    for start in ordered_starts:
        end = start + window_samples
        if end > len(total):
            continue
        metrics = _stability_metrics_for_slice(
            total,
            left,
            right,
            ap,
            ml,
            times,
            start,
            end,
        )
        if metrics is None or not _post_window_is_stable(metrics, body_weight, config):
            continue

        score = (
            float(metrics["total_cv"])
            + 0.5 * (float(metrics["left_cv"]) + float(metrics["right_cv"]))
            + abs(float(metrics["total_slope"])) / max(body_weight, 1.0)
            + 0.25
            * (abs(float(metrics["left_slope"])) + abs(float(metrics["right_slope"])))
            / max(body_weight, 1.0)
            + float(metrics["cop_ap_range"])
            + float(metrics["cop_ml_range"])
        )
        metrics["score"] = float(score)
        if best is None or score < float(best["score"]):
            best = metrics

    return best


def terminal_velocity_diagnostic(time_arr, velocity_raw, landing_time, window_s):
    times = np.asarray(time_arr, dtype=float)
    velocity = np.asarray(velocity_raw, dtype=float)
    start_time = max(float(landing_time), float(times[-1] - window_s))
    mask = (times >= start_time) & np.isfinite(velocity)
    if np.sum(mask) < 5:
        return {
            "median": np.nan,
            "mean": np.nan,
            "sd": np.nan,
            "window_start": start_time,
            "window_end": float(times[-1]),
            "sample_count": int(np.sum(mask)),
        }
    segment = velocity[mask]
    return {
        "median": float(np.median(segment)),
        "mean": float(np.mean(segment)),
        "sd": float(np.std(segment, ddof=0)),
        "window_start": start_time,
        "window_end": float(times[-1]),
        "sample_count": int(len(segment)),
    }


def find_velocity_zero_crossing(time_arr, velocity, start_idx, end_idx, persistence_samples):
    times = np.asarray(time_arr, dtype=float)
    values = np.asarray(velocity, dtype=float)
    stop = min(len(values) - persistence_samples, int(end_idx))
    for i in range(max(1, int(start_idx)), stop):
        test = values[i : i + persistence_samples]
        if not np.all(np.isfinite(test)):
            continue
        if values[i - 1] <= 0.0 and values[i] > 0.0 and np.all(test > 0.0):
            crossing_time = interpolate_crossing(times, values, i, 0.0)
            return i, crossing_time
    return None, None


def _smoothstep_fraction(times, start_time, end_time):
    values = np.asarray(times, dtype=float)
    denominator = float(end_time - start_time)
    if denominator <= 0.0:
        return np.zeros(len(values), dtype=float)
    x = np.clip((values - start_time) / denominator, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def apply_post_landing_drift_correction(
    time_arr,
    grf,
    body_weight,
    body_mass,
    movement_start_time,
    landing_time,
    post_window,
    config,
):
    """Audit drift dan closure tanpa mengubah hasil utama pra-take-off.

    Tiga kurva dipisahkan secara tegas:
    1. ``velocity_raw`` adalah integral gaya utama dan menjadi sumber hasil
       impulse–momentum.
    2. ``velocity_force_corrected`` hanya mengaudit kemungkinan perubahan
       offset gaya antara baseline sebelum gerakan dan window stabil sesudah
       landing.
    3. ``velocity`` adalah audit closure/ZUPT yang mulai dikoreksi tepat saat
       landing. Karena koreksi dimulai setelah landing, take-off velocity dan
       jump height utama tidak pernah berubah.

    v6.2 menolak audit ZUPT hanya karena residual velocity lebih besar dari
    0.10 m/s. Pada rekaman post-landing yang panjang, residual kecil per detik
    dapat terakumulasi menjadi nilai lebih besar meskipun gaya ekuivalen yang
    dibutuhkan untuk menutup kurva sangat kecil. v7.0 mempertahankan penilaian kelayakan dari
    gaya ekuivalen relatif terhadap body weight serta keberadaan window stabil.
    """
    times = np.asarray(time_arr, dtype=float)
    force = np.asarray(grf, dtype=float)
    raw_net_force = force - float(body_weight)
    raw_acceleration = raw_net_force / float(body_mass)
    velocity_raw = cumulative_integral_from_time(
        times, raw_acceleration, movement_start_time
    )

    result = {
        "velocity_raw": velocity_raw,
        "velocity_force_corrected": velocity_raw.copy(),
        "velocity": velocity_raw.copy(),
        "net_force_raw": raw_net_force,
        "net_force_corrected": raw_net_force.copy(),
        "dynamic_baseline_force": np.full(len(times), float(body_weight)),
        "force_bias_n": np.nan,
        "force_bias_applied": False,
        "raw_post_residual_m_s": np.nan,
        "force_corrected_post_residual_m_s": np.nan,
        "zupt_residual_m_s": np.nan,
        "zupt_applied": False,
        "correction_anchor_time": np.nan,
        "closure_start_time": np.nan,
        "closure_end_time": np.nan,
        "closure_duration_s": np.nan,
        "closure_equivalent_force_n": np.nan,
        "closure_equivalent_force_percent_bw": np.nan,
        "closure_post_residual_m_s": np.nan,
        "closure_source": "RAW_PRIMARY",
    }
    if post_window is None:
        return result

    start = int(post_window["start"])
    end = int(post_window["end"])
    raw_residual = float(np.nanmean(velocity_raw[start:end]))
    result["raw_post_residual_m_s"] = raw_residual
    result["zupt_residual_m_s"] = raw_residual

    # Audit perubahan offset gaya pre→post. Kurva ini tidak dipakai sebagai
    # hasil utama dan tidak menjadi dasar wajib bagi audit closure.
    post_force_mean = float(post_window["mean"])
    force_bias = post_force_mean - float(body_weight)
    result["force_bias_n"] = force_bias
    max_fraction = float(config.get("cmj_force_drift_max_fraction_bw", 0.03))
    max_force_bias = max_fraction * float(body_weight)
    anchor_time = float(post_window["anchor_time"])

    if (
        np.isfinite(force_bias)
        and np.isfinite(anchor_time)
        and anchor_time > movement_start_time
        and abs(force_bias) <= max_force_bias
    ):
        fraction = _smoothstep_fraction(times, movement_start_time, anchor_time)
        dynamic_baseline = float(body_weight) + force_bias * fraction
        force_corrected_net = force - dynamic_baseline
        force_corrected_velocity = cumulative_integral_from_time(
            times, force_corrected_net / float(body_mass), movement_start_time
        )
        force_corrected_residual = float(
            np.nanmean(force_corrected_velocity[start:end])
        )
        result["velocity_force_corrected"] = force_corrected_velocity
        result["net_force_corrected"] = force_corrected_net
        result["dynamic_baseline_force"] = dynamic_baseline
        result["force_corrected_post_residual_m_s"] = force_corrected_residual
        result["force_bias_applied"] = bool(
            np.isfinite(force_corrected_residual)
            and abs(force_corrected_residual) < abs(raw_residual)
        )

    # Audit closure dimulai pada landing dan selesai sebelum window stabil.
    # Ini mempertahankan seluruh velocity pra-landing, termasuk take-off.
    closure_start_time = float(landing_time)
    closure_end_time = float(post_window["start_time"])
    closure_duration_s = closure_end_time - closure_start_time
    result["closure_start_time"] = closure_start_time
    result["closure_end_time"] = closure_end_time
    result["closure_duration_s"] = closure_duration_s
    result["correction_anchor_time"] = closure_end_time

    if not np.isfinite(raw_residual) or closure_duration_s <= 0.0:
        return result

    equivalent_force_n = (
        raw_residual * float(body_mass) / closure_duration_s
    )
    equivalent_force_percent_bw = (
        abs(equivalent_force_n) / float(body_weight) * 100.0
        if body_weight > 0.0 else np.nan
    )
    result["closure_equivalent_force_n"] = equivalent_force_n
    result["closure_equivalent_force_percent_bw"] = (
        equivalent_force_percent_bw
    )

    max_equivalent_fraction = float(
        config.get("cmj_post_closure_max_equiv_force_fraction_bw", 0.03)
    )
    min_duration_s = float(
        config.get("cmj_post_closure_min_duration_s", 0.35)
    )
    closure_allowed = bool(
        closure_duration_s >= min_duration_s
        and np.isfinite(equivalent_force_percent_bw)
        and equivalent_force_percent_bw
        <= max_equivalent_fraction * 100.0
    )
    if not closure_allowed:
        return result

    closure_fraction = _smoothstep_fraction(
        times, closure_start_time, closure_end_time
    )
    velocity_closure = velocity_raw.copy()
    valid = np.isfinite(velocity_closure) & (times >= closure_start_time)
    velocity_closure[valid] = (
        velocity_closure[valid] - raw_residual * closure_fraction[valid]
    )
    closure_post_residual = float(
        np.nanmean(velocity_closure[start:end])
    )
    result["velocity"] = velocity_closure
    result["closure_post_residual_m_s"] = closure_post_residual
    result["zupt_residual_m_s"] = closure_post_residual
    result["closure_source"] = "POST_LANDING_SMOOTHSTEP"
    result["zupt_applied"] = True
    return result


def ask_expected_mass():
    mass_text = input(
        "Masukkan berat badan referensi (kg) "
        "[Enter untuk melewati]: "
    )
    if not mass_text.strip():
        return None
    try:
        mass = float(mass_text.strip().replace(",", "."))
    except ValueError as exc:
        raise ValueError("Berat badan harus berupa angka positif.") from exc
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Berat badan harus berupa angka positif.")
    return mass


def _percentage_difference(a: float, b: float) -> float:
    denominator = (abs(a) + abs(b)) / 2.0
    if denominator <= 1e-12:
        return np.nan
    return abs(a - b) / denominator * 100.0



def _adaptive_impact_cutoff(config: dict[str, Any], fs: float) -> float:
    requested = float(config.get("cmj_impact_filter_hz", IMPACT_FILTER_CUTOFF_HZ))
    max_fraction = float(config.get("cmj_impact_filter_max_fraction_fs", 0.28))
    return float(min(requested, max_fraction * fs, 0.45 * fs))


def _relative_range_percent(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size < 2:
        return np.nan
    mean_value = float(np.mean(np.abs(finite)))
    if mean_value <= 1e-12:
        return np.nan
    return float((np.max(finite) - np.min(finite)) / mean_value * 100.0)


def _exact_signal_window(
    time_arr: np.ndarray,
    signal: np.ndarray,
    start_time: float,
    end_time: float,
):
    """Ambil window dengan titik batas yang diinterpolasi tepat."""
    times = np.asarray(time_arr, dtype=float)
    values = np.asarray(signal, dtype=float)
    if len(times) != len(values) or len(times) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    start = max(float(start_time), float(times[0]))
    end = min(float(end_time), float(times[-1]))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return np.array([], dtype=float), np.array([], dtype=float)

    inside = (times > start) & (times < end)
    window_time = np.concatenate(([start], times[inside], [end]))
    window_signal = np.concatenate(
        (
            [np.interp(start, times, values)],
            values[inside],
            [np.interp(end, times, values)],
        )
    )
    valid = np.isfinite(window_time) & np.isfinite(window_signal)
    window_time = window_time[valid]
    window_signal = window_signal[valid]
    if len(window_time) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    keep = np.concatenate(([True], np.diff(window_time) > 0.0))
    return window_time[keep], window_signal[keep]


def _time_weighted_window_stats(
    time_arr: np.ndarray,
    signal: np.ndarray,
    start_time: float,
    end_time: float,
    fs: float,
):
    window_time, window_signal = _exact_signal_window(
        time_arr, signal, start_time, end_time
    )
    duration = float(end_time - start_time)
    if len(window_time) < 2 or duration <= 0.0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "duration_s": max(0.0, duration),
            "sample_count": 0,
            "equivalent_samples": np.nan,
        }
    mean_value = trapezoid_integral(window_signal, window_time) / duration
    variance = trapezoid_integral(
        (window_signal - mean_value) ** 2,
        window_time,
    ) / duration
    return {
        "mean": float(mean_value),
        "sd": float(np.sqrt(max(0.0, variance))),
        "duration_s": duration,
        "sample_count": int(len(window_time)),
        "equivalent_samples": float(duration * fs),
    }


def _linear_loading_window(
    signal: np.ndarray,
    time_arr: np.ndarray,
    landing_time: float,
    peak_time: float,
    requested_window_s: float,
    body_weight: float,
    fs: float,
):
    """Kemiringan gaya awal memakai regresi linear pada window tetap.

    Metrik ini tidak menggantikan RFD 20–80%. Tujuannya adalah menyediakan
    estimasi loading rate yang lebih stabil ketika fase 20–80% hanya terdiri
    dari sedikit sampel pada frame rate efektif sekitar 220 Hz.
    """
    requested = float(requested_window_s)
    end_time = min(float(landing_time) + requested, float(peak_time))
    window_time, window_signal = _exact_signal_window(
        time_arr, signal, landing_time, end_time
    )
    if len(window_time) < 3 or end_time <= landing_time:
        return {
            "requested_window_s": requested,
            "actual_window_s": max(0.0, end_time - float(landing_time)),
            "slope_n_s": np.nan,
            "r2": np.nan,
            "sample_count": int(len(window_time)),
            "equivalent_samples": np.nan,
            "net_impulse_n_s": np.nan,
        }

    x = window_time - float(landing_time)
    slope, intercept = np.polyfit(x, window_signal, 1)
    fitted = slope * x + intercept
    residual_ss = float(np.sum((window_signal - fitted) ** 2))
    total_ss = float(np.sum((window_signal - np.mean(window_signal)) ** 2))
    r2 = 1.0 - residual_ss / total_ss if total_ss > 1e-12 else np.nan
    actual_duration = float(end_time - landing_time)
    net_impulse = trapezoid_integral(
        window_signal - float(body_weight), window_time
    )
    return {
        "requested_window_s": requested,
        "actual_window_s": actual_duration,
        "slope_n_s": float(slope),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "sample_count": int(len(window_time)),
        "equivalent_samples": float(actual_duration * fs),
        "net_impulse_n_s": float(net_impulse),
    }


def _compute_flight_zero_metrics(
    time_arr: np.ndarray,
    total_force: np.ndarray,
    left_force: np.ndarray,
    right_force: np.ndarray,
    takeoff_time: float,
    landing_time: float,
    fs: float,
    config: dict[str, Any],
):
    """Estimasi zero-force dari bagian tengah flight.

    Sampel tepat di sekitar take-off dan landing mengandung transisi threshold,
    filter edge, dan efek ketidaksimultanan MUX. QC zero-force karena itu memakai
    flight yang dipangkas pada kedua ujung, sedangkan statistik full-flight tetap
    disimpan untuk audit.
    """
    flight_duration = float(landing_time - takeoff_time)
    requested_guard = float(config.get("cmj_flight_zero_guard_s", 0.020))
    min_samples = int(config.get("cmj_flight_zero_min_samples", 12))
    guard_s = max(requested_guard, 2.0 / fs)
    required_central = max(min_samples / fs, 0.040)
    if flight_duration - 2.0 * guard_s < required_central:
        guard_s = max(1.0 / fs, 0.10 * flight_duration)

    central_start = float(takeoff_time + guard_s)
    central_end = float(landing_time - guard_s)
    if central_end <= central_start:
        central_start = float(takeoff_time)
        central_end = float(landing_time)
        guard_s = 0.0

    full_total = _time_weighted_window_stats(
        time_arr, total_force, takeoff_time, landing_time, fs
    )
    central_total = _time_weighted_window_stats(
        time_arr, total_force, central_start, central_end, fs
    )
    central_left = _time_weighted_window_stats(
        time_arr, left_force, central_start, central_end, fs
    )
    central_right = _time_weighted_window_stats(
        time_arr, right_force, central_start, central_end, fs
    )

    return {
        "guard_s": float(guard_s),
        "central_start_time": central_start,
        "central_end_time": central_end,
        "central_mean": float(central_total["mean"]),
        "central_sd": float(central_total["sd"]),
        "central_sample_count": int(central_total["sample_count"]),
        "central_equivalent_samples": float(
            central_total["equivalent_samples"]
        ),
        "left_mean": float(central_left["mean"]),
        "right_mean": float(central_right["mean"]),
        "full_mean": float(full_total["mean"]),
        "full_sd": float(full_total["sd"]),
        "full_sample_count": int(full_total["sample_count"]),
    }


def _compute_landing_impact_metrics(
    signal: np.ndarray,
    time_arr: np.ndarray,
    landing_idx: int,
    landing_time: float,
    body_weight: float,
    fs: float,
    config: dict[str, Any] | None = None,
):
    values = np.asarray(signal, dtype=float)
    times = np.asarray(time_arr, dtype=float)
    cfg = {} if config is None else config
    end = min(len(values), landing_idx + max(2, int(round(0.50 * fs))))
    segment = values[landing_idx:end]
    segment_time = times[landing_idx:end]
    if len(segment) < 3:
        raise ValueError("Window impact landing terlalu pendek.")

    peak_relative_idx = int(np.nanargmax(segment))
    peak_idx = int(landing_idx + peak_relative_idx)
    peak_force = float(values[peak_idx])
    peak_time = float(times[peak_idx])
    time_to_peak = float(peak_time - landing_time)
    average_loading_rate = (
        (peak_force - body_weight) / time_to_peak
        if time_to_peak > 1.0 / fs
        else np.nan
    )

    amplitude = peak_force - body_weight
    level_20 = body_weight + 0.20 * amplitude
    level_80 = body_weight + 0.80 * amplitude
    # Gunakan crossing naik terakhir sebelum peak utama. Pada landing bimodal,
    # crossing pertama dapat berasal dari peak awal yang lebih kecil dan tidak
    # merepresentasikan rise menuju peak utama.
    time_20, _ = find_last_upward_crossing(
        times, values, level_20, landing_idx, peak_idx + 1
    )
    time_80, _ = find_last_upward_crossing(
        times, values, level_80, landing_idx, peak_idx + 1
    )
    duration_20_80 = (
        float(time_80 - time_20)
        if time_20 is not None and time_80 is not None and time_80 > time_20
        else np.nan
    )
    rate_20_80 = (
        (level_80 - level_20) / duration_20_80
        if np.isfinite(duration_20_80) and duration_20_80 > 0.0
        else np.nan
    )

    peak_indices, properties = find_peaks(
        segment,
        height=max(1.20 * body_weight, body_weight + 100.0),
        prominence=max(0.10 * body_weight, 50.0),
        distance=max(1, int(round(0.040 * fs))),
    )
    if len(peak_indices) == 0:
        peak_indices = np.array([peak_relative_idx], dtype=int)
        properties = {"prominences": np.array([np.nan], dtype=float)}

    peak_forces = [float(segment[idx]) for idx in peak_indices]
    peak_times = [float(segment_time[idx]) for idx in peak_indices]
    prominences = [
        float(value)
        for value in properties.get("prominences", np.array([], dtype=float))
    ]
    sorted_forces = sorted(peak_forces, reverse=True)
    second_peak_ratio = (
        float(sorted_forces[1] / sorted_forces[0])
        if len(sorted_forces) >= 2 and sorted_forces[0] > 0.0
        else np.nan
    )
    first_peak_force = peak_forces[0] if peak_forces else np.nan
    first_peak_time = peak_times[0] if peak_times else np.nan
    second_peak_force = peak_forces[1] if len(peak_forces) >= 2 else np.nan
    second_peak_time = peak_times[1] if len(peak_times) >= 2 else np.nan
    interpeak_interval_s = (
        float(second_peak_time - first_peak_time)
        if np.isfinite(first_peak_time) and np.isfinite(second_peak_time)
        else np.nan
    )
    dominant_peak_sequence = (
        int(np.argmax(np.asarray(peak_forces, dtype=float))) + 1
        if peak_forces else 1
    )

    # Landing impulse v8.1.
    # Selain peak force, laporkan impulse 0-50 ms dan 0-100 ms.
    # Impulse dihitung sebagai gross GRF integral dan net integral di atas BW.
    requested_impulse_windows = sorted(
        {
            float(value)
            for value in cfg.get("cmj_landing_impulse_windows_s", [0.050, 0.100])
            if np.isfinite(float(value)) and float(value) > 0.0
        }
    )
    landing_impulses: list[dict[str, float]] = []
    for requested_window_s in requested_impulse_windows:
        requested_end_time = float(landing_time + requested_window_s)
        actual_end_time = min(requested_end_time, float(times[-1]))
        actual_window_s = max(0.0, actual_end_time - float(landing_time))
        if actual_window_s <= 0.0:
            gross_impulse_n_s = np.nan
            net_impulse_n_s = np.nan
        else:
            gross_impulse_n_s = integrate_between(
                times,
                values,
                float(landing_time),
                actual_end_time,
            )
            net_impulse_n_s = float(
                gross_impulse_n_s - body_weight * actual_window_s
            )
        landing_impulses.append(
            {
                "requested_window_s": float(requested_window_s),
                "actual_window_s": float(actual_window_s),
                "gross_impulse_n_s": float(gross_impulse_n_s),
                "net_impulse_n_s": float(net_impulse_n_s),
            }
        )

    requested_windows = sorted(
        {
            float(value)
            for value in cfg.get("cmj_early_loading_windows_s", [0.030, 0.050])
            if np.isfinite(float(value)) and float(value) > 0.0
        }
    )
    primary_requested = float(
        cfg.get("cmj_early_loading_primary_window_s", 0.050)
    )
    if not requested_windows:
        requested_windows = [primary_requested]
    early_loading_metrics = [
        _linear_loading_window(
            values,
            times,
            landing_time,
            peak_time,
            requested_window,
            body_weight,
            fs,
        )
        for requested_window in requested_windows
    ]
    primary_early = min(
        early_loading_metrics,
        key=lambda item: abs(
            float(item["requested_window_s"]) - primary_requested
        ),
    )

    return {
        "peak_idx": peak_idx,
        "peak_force": peak_force,
        "peak_time": peak_time,
        "time_to_peak": time_to_peak,
        "average_loading_rate": average_loading_rate,
        "level_20": level_20,
        "level_80": level_80,
        "time_20": time_20,
        "time_80": time_80,
        "duration_20_80": duration_20_80,
        "rate_20_80": rate_20_80,
        "time_to_peak_samples": time_to_peak * fs,
        "samples_20_80": (
            duration_20_80 * fs if np.isfinite(duration_20_80) else np.nan
        ),
        "peak_count": int(len(peak_indices)),
        "peak_forces": peak_forces,
        "peak_times": peak_times,
        "peak_prominences": prominences,
        "second_peak_ratio": second_peak_ratio,
        "first_peak_force": float(first_peak_force),
        "first_peak_time": float(first_peak_time),
        "second_peak_force": float(second_peak_force),
        "second_peak_time": float(second_peak_time),
        "interpeak_interval_s": float(interpeak_interval_s),
        "dominant_peak_sequence": int(dominant_peak_sequence),
        "landing_impulses": landing_impulses,
        "early_loading_metrics": early_loading_metrics,
        "early_loading_primary_window_s": float(
            primary_early["requested_window_s"]
        ),
        "early_loading_primary_actual_window_s": float(
            primary_early["actual_window_s"]
        ),
        "early_loading_primary_rate_n_s": float(
            primary_early["slope_n_s"]
        ),
        "early_loading_primary_r2": float(primary_early["r2"]),
        "early_loading_primary_sample_count": int(
            primary_early["sample_count"]
        ),
        "early_loading_primary_equivalent_samples": float(
            primary_early["equivalent_samples"]
        ),
        "early_loading_primary_net_impulse_n_s": float(
            primary_early["net_impulse_n_s"]
        ),
    }


def _mux_order_diagnostics(mux_skew: np.ndarray, fs: float, config: dict[str, Any]):
    skew = np.asarray(mux_skew, dtype=float)
    skew = skew[np.isfinite(skew)]
    if skew.size == 0:
        return {
            "signed_mean_us": np.nan,
            "mean_abs_us": np.nan,
            "p95_abs_us": np.nan,
            "p95_fraction": np.nan,
            "frame_period_us": np.nan,
            "temporal_resolution_ms": np.nan,
            "alternation_fraction": np.nan,
            "signed_bias_fraction": np.nan,
            "policy": "UNKNOWN",
        }

    frame_period_us = 1_000_000.0 / fs
    mean_abs = float(np.mean(np.abs(skew)))
    signs = np.sign(skew[np.abs(skew) > 1.0])
    alternation_fraction = (
        float(np.mean(signs[1:] != signs[:-1])) if signs.size >= 2 else np.nan
    )
    signed_bias_fraction = (
        abs(float(np.mean(skew))) / mean_abs if mean_abs > 1e-12 else np.nan
    )
    min_alt = float(config.get("cmj_mux_alternation_min_fraction", 0.80))
    max_bias = float(config.get("cmj_mux_signed_bias_max_fraction", 0.20))
    if (
        np.isfinite(alternation_fraction)
        and alternation_fraction >= min_alt
        and np.isfinite(signed_bias_fraction)
        and signed_bias_fraction <= max_bias
    ):
        policy = "BALANCED_ALTERNATING"
    elif signs.size >= 2 and np.all(signs == signs[0]):
        policy = "FIXED_ORDER"
    else:
        policy = "UNBALANCED_OR_MIXED"

    return {
        "signed_mean_us": float(np.mean(skew)),
        "mean_abs_us": mean_abs,
        "p95_abs_us": float(np.percentile(np.abs(skew), 95)),
        "p95_fraction": float(np.percentile(np.abs(skew), 95) / frame_period_us),
        "frame_period_us": float(frame_period_us),
        "temporal_resolution_ms": float(frame_period_us / 1000.0),
        "alternation_fraction": alternation_fraction,
        "signed_bias_fraction": signed_bias_fraction,
        "policy": policy,
    }


def analyze_cmj(data: dict[str, Any], expected_mass: float | None = None):
    time_arr = np.asarray(data["time"], dtype=float)
    frame_arr = np.asarray(data["frame"], dtype=np.int64)
    raw_grf = np.asarray(data["grf"], dtype=float)
    raw_left = np.asarray(data["grf_l"], dtype=float)
    raw_right = np.asarray(data["grf_r"], dtype=float)
    raw_cop_ap = np.asarray(data["cop_ap"], dtype=float)
    raw_cop_ml = np.asarray(data["cop_ml"], dtype=float)

    lengths = {
        len(time_arr), len(frame_arr), len(raw_grf), len(raw_left),
        len(raw_right), len(raw_cop_ap), len(raw_cop_ml)
    }
    if len(lengths) != 1 or len(time_arr) < 50:
        raise ValueError("Panjang data CMJ tidak sejajar atau terlalu pendek.")
    if not np.all(np.diff(time_arr) > 0):
        raise ValueError("Timestamp CMJ tidak monoton naik.")

    sampling_diag = prc.get_sampling_diagnostics(frame_arr, time_arr)
    serial_diag = prc.get_serial_diagnostics(data["all_received_frames"])
    fs = float(sampling_diag["fs"])
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate tidak valid.")
    sampling_jitter_fraction = (
        float(sampling_diag["jitter_sd"]) / float(sampling_diag["mean_dt"])
        if np.isfinite(sampling_diag["jitter_sd"])
        and np.isfinite(sampling_diag["mean_dt"])
        and float(sampling_diag["mean_dt"]) > 0.0
        else np.nan
    )

    config = data.get("config", prc.load_config())
    aligned_counts = np.asarray(data.get("aligned_counts", []), dtype=float)
    if aligned_counts.ndim == 2 and aligned_counts.shape == (len(time_arr), 8):
        dynamic_signals = prc.calculate_forces_and_cop_batch(
            aligned_counts,
            config,
            fs=fs,
            force_cutoff_hz=CMJ_FILTER_CUTOFF_HZ,
        )
        event_signals = prc.calculate_forces_and_cop_batch(
            aligned_counts,
            config,
            fs=fs,
            force_cutoff_hz=min(EVENT_FILTER_CUTOFF_HZ, 0.40 * fs),
        )
        post_signals = prc.calculate_forces_and_cop_batch(
            aligned_counts,
            config,
            fs=fs,
            force_cutoff_hz=POST_STABILITY_FILTER_HZ,
        )
        impact_filter_cutoff_hz = _adaptive_impact_cutoff(config, fs)
        impact_signals = prc.calculate_forces_and_cop_batch(
            aligned_counts,
            config,
            fs=fs,
            force_cutoff_hz=impact_filter_cutoff_hz,
        )
        grf = np.asarray(dynamic_signals["grf"], dtype=float)
        event_grf = np.asarray(event_signals["grf"], dtype=float)
        event_left = np.asarray(event_signals["fz_l"], dtype=float)
        event_right = np.asarray(event_signals["fz_r"], dtype=float)
        left_force = np.asarray(dynamic_signals["fz_l"], dtype=float)
        right_force = np.asarray(dynamic_signals["fz_r"], dtype=float)
        post_grf = np.asarray(post_signals["grf"], dtype=float)
        post_left = np.asarray(post_signals["fz_l"], dtype=float)
        post_right = np.asarray(post_signals["fz_r"], dtype=float)
        post_cop_ap = np.asarray(post_signals["cop_ap"], dtype=float)
        post_cop_ml = np.asarray(post_signals["cop_ml"], dtype=float)
        impact_grf = np.asarray(impact_signals["grf"], dtype=float)
    else:
        # Kompatibilitas untuk data lama yang belum menyimpan aligned_counts.
        grf = prc.butter_lowpass_filter(raw_grf, CMJ_FILTER_CUTOFF_HZ, fs, order=4)
        event_cutoff = min(EVENT_FILTER_CUTOFF_HZ, 0.40 * fs)
        event_grf = prc.butter_lowpass_filter(
            raw_grf, event_cutoff, fs, order=4
        )
        event_left = prc.butter_lowpass_filter(
            raw_left, event_cutoff, fs, order=4
        )
        event_right = prc.butter_lowpass_filter(
            raw_right, event_cutoff, fs, order=4
        )
        left_force = prc.butter_lowpass_filter(raw_left, CMJ_FILTER_CUTOFF_HZ, fs, order=4)
        right_force = prc.butter_lowpass_filter(raw_right, CMJ_FILTER_CUTOFF_HZ, fs, order=4)
        post_grf = prc.butter_lowpass_filter(raw_grf, POST_STABILITY_FILTER_HZ, fs, order=4)
        impact_filter_cutoff_hz = _adaptive_impact_cutoff(config, fs)
        impact_grf = prc.butter_lowpass_filter(
            raw_grf, impact_filter_cutoff_hz, fs, order=4
        )
        post_left = prc.butter_lowpass_filter(raw_left, POST_STABILITY_FILTER_HZ, fs, order=4)
        post_right = prc.butter_lowpass_filter(raw_right, POST_STABILITY_FILTER_HZ, fs, order=4)
        post_cop_ap = raw_cop_ap.copy()
        post_cop_ml = raw_cop_ml.copy()

    event_kernel = max(3, int(round(fs * EVENT_MEDIAN_WINDOW_S)))
    if event_kernel % 2 == 0:
        event_kernel += 1
    grf_event = median_filter(event_grf, size=event_kernel, mode="nearest")
    left_event = median_filter(event_left, size=event_kernel, mode="nearest")
    right_event = median_filter(event_right, size=event_kernel, mode="nearest")

    initial_baseline = find_stable_baseline(grf, time_arr, fs)
    if initial_baseline is None:
        raise ValueError(
            "Fase berdiri stabil tidak ditemukan sebelum CMJ. "
            "Setelah RECORD, berdiri tenang minimal dua detik sebelum melompat."
        )

    rough_events = detect_unweighting_and_movement(
        grf, time_arr, fs, initial_baseline
    )
    baseline_guard_s = float(config.get("cmj_baseline_guard_s", 0.15))
    refined_baseline = find_stable_baseline(
        grf,
        time_arr,
        fs,
        search_end_time=float(rough_events["movement_start_time"]) - baseline_guard_s,
        prefer_latest=True,
    )
    baseline = refined_baseline if refined_baseline is not None else initial_baseline
    baseline_refined = refined_baseline is not None

    events = detect_unweighting_and_movement(grf, time_arr, fs, baseline)
    baseline_start_idx = int(baseline["start"])
    baseline_end_idx = int(baseline["end"])
    body_weight = float(events["body_weight"])
    body_weight_sd = float(events["body_weight_sd"])
    body_mass = body_weight / prc.GRAVITY
    unweight_threshold = float(events["unweight_threshold"])
    unweighting_idx = int(events["unweighting_idx"])
    unweight_persistence = int(events["unweight_persistence"])
    onset_threshold = float(events["onset_threshold"])
    movement_start_time = float(events["movement_start_time"])
    movement_start_idx = int(events["movement_start_idx"])
    baseline_gap_to_movement_s = float(
        movement_start_time - float(time_arr[baseline_end_idx - 1])
    )

    baseline_left = float(np.median(left_force[baseline_start_idx:baseline_end_idx]))
    baseline_right = float(np.median(right_force[baseline_start_idx:baseline_end_idx]))

    empty_plate_mean = float(data.get("baseline_mean", 0.0))
    empty_plate_sd = float(data.get("baseline_sd", 0.0))
    configured_flight_threshold = float(
        data.get("config", {}).get("cmj_flight_threshold_n", 25.0)
    )
    flight_threshold = max(
        configured_flight_threshold,
        empty_plate_mean + 6.0 * max(empty_plate_sd, 0.0),
    )
    flight_persistence = max(3, int(round(fs * EVENT_PERSISTENCE_S)))

    takeoff_idx = persistent_first(
        grf_event,
        lambda value: value < flight_threshold,
        unweighting_idx,
        flight_persistence,
    )
    if takeoff_idx is None:
        raise ValueError("Take-off tidak ditemukan.")

    landing_confirmation_threshold = max(
        float(config.get("cmj_landing_threshold_n", 50.0)),
        flight_threshold + 10.0,
    )
    landing_confirmation_idx = persistent_first(
        grf_event,
        lambda value: value > landing_confirmation_threshold,
        takeoff_idx + flight_persistence,
        flight_persistence,
    )
    if landing_confirmation_idx is None:
        raise ValueError("Landing tidak ditemukan atau tidak terkonfirmasi.")

    takeoff_time = find_downward_crossing(
        time_arr,
        grf_event,
        flight_threshold,
        unweighting_idx,
        takeoff_idx + flight_persistence + 3,
    )
    landing_confirmation_time = find_upward_crossing(
        time_arr,
        grf_event,
        landing_confirmation_threshold,
        takeoff_idx + 1,
        landing_confirmation_idx + flight_persistence + 3,
    )
    backtrack_samples = max(3, int(round(0.15 * fs)))
    landing_search_start = max(takeoff_idx + 1, landing_confirmation_idx - backtrack_samples)
    landing_time, landing_cross_idx = find_last_upward_crossing(
        time_arr,
        grf_event,
        flight_threshold,
        landing_search_start,
        landing_confirmation_idx + 1,
    )
    if landing_time is None:
        landing_time, landing_cross_idx = find_last_upward_crossing(
            time_arr,
            grf_event,
            flight_threshold,
            takeoff_idx + 1,
            landing_confirmation_idx + 1,
        )
    if (
        takeoff_time is None
        or landing_time is None
        or landing_cross_idx is None
        or landing_confirmation_time is None
    ):
        raise ValueError("Crossing threshold take-off/landing tidak ditemukan.")
    landing_idx = int(landing_cross_idx)

    side_landing_threshold = max(
        float(config.get("cmj_side_landing_threshold_n", 25.0)),
        0.03 * body_weight,
    )
    side_landing_persistence_s = float(
        config.get("cmj_side_landing_persistence_s", 0.015)
    )
    left_landing_time, left_landing_idx = detect_side_landing_onset(
        left_event,
        time_arr,
        takeoff_idx,
        landing_confirmation_idx,
        fs,
        side_landing_threshold,
        side_landing_persistence_s,
    )
    right_landing_time, right_landing_idx = detect_side_landing_onset(
        right_event,
        time_arr,
        takeoff_idx,
        landing_confirmation_idx,
        fs,
        side_landing_threshold,
        side_landing_persistence_s,
    )

    landing_lr_delay_s = (
        float(right_landing_time - left_landing_time)
        if np.isfinite(left_landing_time) and np.isfinite(right_landing_time)
        else np.nan
    )
    landing_lr_abs_delay_s = (
        abs(landing_lr_delay_s)
        if np.isfinite(landing_lr_delay_s)
        else np.nan
    )
    staggered_threshold_s = float(
        config.get("cmj_staggered_landing_threshold_s", 0.015)
    )
    if np.isfinite(landing_lr_delay_s):
        if landing_lr_delay_s > staggered_threshold_s:
            bilateral_landing_order = "LEFT_FIRST"
        elif landing_lr_delay_s < -staggered_threshold_s:
            bilateral_landing_order = "RIGHT_FIRST"
        else:
            bilateral_landing_order = "SIMULTANEOUS"
    else:
        bilateral_landing_order = "UNRESOLVED"

    flight_time = float(landing_time - takeoff_time)
    if not 0.10 <= flight_time <= 0.80:
        raise ValueError(f"Flight time {flight_time:.3f} s tidak realistis.")

    threshold_flight_times: list[float] = []
    threshold_takeoff_times: list[float] = []
    threshold_values = sorted(
        {max(10.0, flight_threshold - 5.0), flight_threshold, flight_threshold + 5.0}
    )
    for threshold_value in threshold_values:
        threshold_takeoff = find_downward_crossing(
            time_arr,
            grf_event,
            threshold_value,
            unweighting_idx,
            takeoff_idx + flight_persistence + 3,
        )
        threshold_landing, _ = find_last_upward_crossing(
            time_arr,
            grf_event,
            threshold_value,
            landing_search_start,
            landing_confirmation_idx + 1,
        )
        if (
            threshold_takeoff is not None
            and threshold_landing is not None
            and threshold_landing > threshold_takeoff
        ):
            threshold_flight_times.append(
                float(threshold_landing - threshold_takeoff)
            )
            threshold_takeoff_times.append(float(threshold_takeoff))
    flight_time_threshold_range_s = (
        float(max(threshold_flight_times) - min(threshold_flight_times))
        if len(threshold_flight_times) >= 2
        else np.nan
    )

    net_force = grf - body_weight
    post_window = find_post_landing_stable_window(
        post_grf,
        post_left,
        post_right,
        post_cop_ap,
        post_cop_ml,
        time_arr,
        landing_time,
        body_weight,
        fs,
        config,
        live_window_start_s=data.get("cmj_post_live_window_start_s"),
        live_window_end_s=data.get("cmj_post_live_window_end_s"),
    )
    correction = apply_post_landing_drift_correction(
        time_arr,
        grf,
        body_weight,
        body_mass,
        movement_start_time,
        landing_time,
        post_window,
        config,
    )
    # Metode utama v5.3 adalah impulse–momentum pra-take-off. Koreksi
    # post-landing/ZUPT tetap dihitung sebagai audit closure, tetapi tidak
    # mengubah hasil utama take-off velocity atau jump height.
    velocity_primary = np.asarray(correction["velocity_raw"], dtype=float)
    velocity_raw = velocity_primary.copy()
    velocity_force_corrected = np.asarray(
        correction["velocity_force_corrected"], dtype=float
    )
    velocity_audit = np.asarray(correction["velocity"], dtype=float)
    velocity = velocity_primary
    net_force_corrected = np.asarray(correction["net_force_corrected"], dtype=float)
    dynamic_baseline_force = np.asarray(
        correction["dynamic_baseline_force"], dtype=float
    )
    force_bias_n = float(correction["force_bias_n"])
    force_bias_applied = bool(correction["force_bias_applied"])
    raw_post_residual = float(correction["raw_post_residual_m_s"])
    force_corrected_post_residual = float(
        correction["force_corrected_post_residual_m_s"]
    )
    zupt_residual = float(correction["zupt_residual_m_s"])
    zupt_applied = bool(correction["zupt_applied"])
    closure_start_time = float(correction["closure_start_time"])
    closure_end_time = float(correction["closure_end_time"])
    closure_duration_s = float(correction["closure_duration_s"])
    closure_equivalent_force_n = float(
        correction["closure_equivalent_force_n"]
    )
    closure_equivalent_force_percent_bw = float(
        correction["closure_equivalent_force_percent_bw"]
    )
    closure_post_residual = float(
        correction["closure_post_residual_m_s"]
    )
    closure_source = str(correction["closure_source"])

    terminal_velocity = terminal_velocity_diagnostic(
        time_arr,
        velocity_primary,
        landing_time,
        TERMINAL_VELOCITY_WINDOW_S,
    )
    terminal_velocity_corrected = terminal_velocity_diagnostic(
        time_arr,
        velocity_audit,
        landing_time,
        TERMINAL_VELOCITY_WINDOW_S,
    )

    positive_persistence = max(3, int(round(fs * EVENT_PERSISTENCE_S)))
    propulsive_idx, propulsive_time = find_velocity_zero_crossing(
        time_arr,
        velocity_primary,
        unweighting_idx + 1,
        takeoff_idx,
        positive_persistence,
    )
    if propulsive_idx is None or propulsive_time is None:
        raise ValueError("Awal fase propulsive tidak ditemukan.")

    eccentric_velocity_segment = velocity_primary[
        movement_start_idx : propulsive_idx + 1
    ]
    minimum_velocity_relative_idx = int(np.nanargmin(eccentric_velocity_segment))
    minimum_velocity_idx = movement_start_idx + minimum_velocity_relative_idx
    minimum_velocity_before_propulsion = float(velocity_primary[minimum_velocity_idx])
    minimum_velocity_time = float(time_arr[minimum_velocity_idx])
    if minimum_velocity_before_propulsion > -0.05:
        raise ValueError(
            "Countermovement tidak cukup jelas: velocity tidak mencapai -0.05 m/s."
        )

    if not (
        movement_start_time
        < float(time_arr[unweighting_idx])
        < propulsive_time
        < takeoff_time
        < landing_time
    ):
        raise ValueError("Urutan event CMJ tidak logis.")

    valid_velocity_primary = np.isfinite(velocity_primary)
    valid_velocity_force_corrected = np.isfinite(velocity_force_corrected)
    valid_velocity_audit = np.isfinite(velocity_audit)
    takeoff_velocity_curve = float(
        np.interp(
            takeoff_time,
            time_arr[valid_velocity_primary],
            velocity_primary[valid_velocity_primary],
        )
    )
    takeoff_velocity_force_corrected = float(
        np.interp(
            takeoff_time,
            time_arr[valid_velocity_force_corrected],
            velocity_force_corrected[valid_velocity_force_corrected],
        )
    )
    takeoff_velocity_audit = float(
        np.interp(
            takeoff_time,
            time_arr[valid_velocity_audit],
            velocity_audit[valid_velocity_audit],
        )
    )
    landing_velocity = float(
        np.interp(
            landing_time,
            time_arr[valid_velocity_primary],
            velocity_primary[valid_velocity_primary],
        )
    )

    total_net_impulse = integrate_between(
        time_arr, net_force, movement_start_time, takeoff_time
    )
    total_net_impulse_corrected = integrate_between(
        time_arr, net_force_corrected, movement_start_time, takeoff_time
    )
    propulsive_net_impulse = integrate_between(
        time_arr, net_force, propulsive_time, takeoff_time
    )
    eccentric_net_impulse = integrate_between(
        time_arr, net_force, movement_start_time, propulsive_time
    )
    unloading_impulse = integrate_between(
        time_arr, net_force, movement_start_time, minimum_velocity_time
    )
    braking_impulse = integrate_between(
        time_arr, net_force, minimum_velocity_time, propulsive_time
    )
    pre_propulsive_net_impulse = eccentric_net_impulse

    velocity_from_total_impulse = total_net_impulse / body_mass
    velocity_from_corrected_impulse = total_net_impulse_corrected / body_mass
    # Nilai utama ditetapkan langsung dari impulse/mass. Interpolasi kurva
    # digunakan hanya sebagai pemeriksaan konsistensi numerik.
    takeoff_velocity = float(velocity_from_total_impulse)
    takeoff_velocity_raw = float(takeoff_velocity_curve)
    impulse_velocity_difference = abs(
        takeoff_velocity_curve - velocity_from_total_impulse
    )
    force_drift_takeoff_adjustment = (
        takeoff_velocity_force_corrected - takeoff_velocity_curve
    )
    # Audit closure v7.0 dimulai setelah landing. Karena itu penyesuaian
    # take-off dibandingkan terhadap kurva primer mentah, bukan terhadap kurva
    # force-drift audit yang memang dapat berbeda sebelum take-off.
    zupt_takeoff_adjustment = (
        takeoff_velocity_audit - takeoff_velocity_curve
    )
    total_takeoff_adjustment = takeoff_velocity_audit - takeoff_velocity

    impulse_height_threshold_values: list[float] = []
    for threshold_takeoff_time in threshold_takeoff_times:
        if threshold_takeoff_time <= movement_start_time:
            continue
        threshold_impulse = integrate_between(
            time_arr,
            net_force,
            movement_start_time,
            threshold_takeoff_time,
        )
        threshold_velocity = threshold_impulse / body_mass
        if np.isfinite(threshold_velocity) and threshold_velocity > 0.0:
            impulse_height_threshold_values.append(
                threshold_velocity**2 / (2.0 * prc.GRAVITY)
            )
    impulse_height_threshold_range_m = (
        float(max(impulse_height_threshold_values) - min(impulse_height_threshold_values))
        if len(impulse_height_threshold_values) >= 2
        else np.nan
    )

    left_net = left_force - baseline_left
    right_net = right_force - baseline_right
    left_net_impulse = integrate_between(time_arr, left_net, propulsive_time, takeoff_time)
    right_net_impulse = integrate_between(time_arr, right_net, propulsive_time, takeoff_time)
    net_leg_sum = left_net_impulse + right_net_impulse

    left_gross_impulse = integrate_between(time_arr, left_force, propulsive_time, takeoff_time)
    right_gross_impulse = integrate_between(time_arr, right_force, propulsive_time, takeoff_time)
    gross_leg_sum = left_gross_impulse + right_gross_impulse

    def contribution(left_value, right_value):
        total = left_value + right_value
        if left_value <= 0 or right_value <= 0 or total <= 1e-12:
            return np.nan, np.nan, np.nan
        left_percent = left_value / total * 100.0
        right_percent = right_value / total * 100.0
        difference = abs(left_value - right_value) / total * 100.0
        return left_percent, right_percent, difference

    left_net_contrib, right_net_contrib, net_contrib_diff = contribution(
        left_net_impulse,
        right_net_impulse,
    )
    left_gross_contrib, right_gross_contrib, gross_contrib_diff = contribution(
        left_gross_impulse,
        right_gross_impulse,
    )

    peak_force = float(np.max(grf[propulsive_idx:takeoff_idx]))
    minimum_force = float(np.min(grf[unweighting_idx : propulsive_idx + 1]))

    landing_metrics = _compute_landing_impact_metrics(
        impact_grf,
        time_arr,
        landing_idx,
        landing_time,
        body_weight,
        fs,
        config,
    )
    peak_landing_idx = int(landing_metrics["peak_idx"])
    peak_landing_force = float(landing_metrics["peak_force"])
    peak_landing_time = float(landing_metrics["peak_time"])
    landing_time_to_peak = float(landing_metrics["time_to_peak"])
    landing_loading_rate = float(landing_metrics["average_loading_rate"])
    landing_loading_rate_20_80 = float(landing_metrics["rate_20_80"])
    landing_time_to_peak_samples = float(landing_metrics["time_to_peak_samples"])
    landing_20_80_duration_s = float(landing_metrics["duration_20_80"])
    landing_20_80_samples = float(landing_metrics["samples_20_80"])
    landing_peak_count = int(landing_metrics["peak_count"])
    landing_peak_forces = list(landing_metrics["peak_forces"])
    landing_peak_times = list(landing_metrics["peak_times"])
    landing_peak_prominences = list(landing_metrics["peak_prominences"])
    landing_second_peak_ratio = float(landing_metrics["second_peak_ratio"])
    landing_first_peak_force = float(landing_metrics["first_peak_force"])
    landing_first_peak_time = float(landing_metrics["first_peak_time"])
    landing_second_peak_force = float(landing_metrics["second_peak_force"])
    landing_second_peak_time = float(landing_metrics["second_peak_time"])
    landing_interpeak_interval_s = float(landing_metrics["interpeak_interval_s"])
    landing_dominant_peak_sequence = int(
        landing_metrics["dominant_peak_sequence"]
    )
    early_loading_metrics = list(landing_metrics["early_loading_metrics"])
    early_loading_primary_window_s = float(
        landing_metrics["early_loading_primary_window_s"]
    )
    early_loading_primary_actual_window_s = float(
        landing_metrics["early_loading_primary_actual_window_s"]
    )
    early_loading_primary_rate_n_s = float(
        landing_metrics["early_loading_primary_rate_n_s"]
    )
    early_loading_primary_r2 = float(
        landing_metrics["early_loading_primary_r2"]
    )
    early_loading_primary_sample_count = int(
        landing_metrics["early_loading_primary_sample_count"]
    )
    early_loading_primary_equivalent_samples = float(
        landing_metrics["early_loading_primary_equivalent_samples"]
    )
    early_loading_primary_net_impulse_n_s = float(
        landing_metrics["early_loading_primary_net_impulse_n_s"]
    )
    landing_impulses = list(landing_metrics.get("landing_impulses", []))

    def landing_impulse_at(window_s: float, key: str) -> float:
        if not landing_impulses:
            return np.nan
        selected = min(
            landing_impulses,
            key=lambda item: abs(
                float(item.get("requested_window_s", np.inf)) - float(window_s)
            ),
        )
        return float(selected.get(key, np.nan))

    landing_gross_impulse_50ms = landing_impulse_at(
        0.050, "gross_impulse_n_s"
    )
    landing_net_impulse_50ms = landing_impulse_at(
        0.050, "net_impulse_n_s"
    )
    landing_gross_impulse_100ms = landing_impulse_at(
        0.100, "gross_impulse_n_s"
    )
    landing_net_impulse_100ms = landing_impulse_at(
        0.100, "net_impulse_n_s"
    )

    landing_window_end = min(len(grf), landing_idx + max(2, int(round(0.50 * fs))))
    peak_landing_force_kinetic = float(np.max(grf[landing_idx:landing_window_end]))

    configured_cutoffs = config.get(
        "cmj_impact_sensitivity_cutoffs_hz", [40.0, 50.0, impact_filter_cutoff_hz]
    )
    sensitivity_cutoffs = sorted({
        float(min(float(cutoff), impact_filter_cutoff_hz))
        for cutoff in configured_cutoffs
        if np.isfinite(float(cutoff)) and float(cutoff) > 0.0
    } | {float(impact_filter_cutoff_hz)})
    impact_sensitivity: list[dict[str, float]] = []
    for cutoff in sensitivity_cutoffs:
        filtered = prc.butter_lowpass_filter(raw_grf, cutoff, fs, order=4)
        metrics = _compute_landing_impact_metrics(
            filtered,
            time_arr,
            landing_idx,
            landing_time,
            body_weight,
            fs,
            config,
        )
        sensitivity_impulses = list(metrics.get("landing_impulses", []))

        def sensitivity_impulse(window_s: float, key: str) -> float:
            if not sensitivity_impulses:
                return np.nan
            selected = min(
                sensitivity_impulses,
                key=lambda item: abs(
                    float(item.get("requested_window_s", np.inf))
                    - float(window_s)
                ),
            )
            return float(selected.get(key, np.nan))

        impact_sensitivity.append({
            "cutoff_hz": float(cutoff),
            "peak_force": float(metrics["peak_force"]),
            "rate_20_80": float(metrics["rate_20_80"]),
            "early_rate_primary": float(
                metrics["early_loading_primary_rate_n_s"]
            ),
            "gross_impulse_50ms": sensitivity_impulse(
                0.050, "gross_impulse_n_s"
            ),
            "net_impulse_50ms": sensitivity_impulse(
                0.050, "net_impulse_n_s"
            ),
            "gross_impulse_100ms": sensitivity_impulse(
                0.100, "gross_impulse_n_s"
            ),
            "net_impulse_100ms": sensitivity_impulse(
                0.100, "net_impulse_n_s"
            ),
        })
    impact_peak_variation_percent = _relative_range_percent(
        [item["peak_force"] for item in impact_sensitivity]
    )
    impact_rfd_variation_percent = _relative_range_percent(
        [item["early_rate_primary"] for item in impact_sensitivity]
    )
    impact_rfd_20_80_variation_percent = _relative_range_percent(
        [item["rate_20_80"] for item in impact_sensitivity]
    )
    impact_peak_values = np.asarray(
        [item["peak_force"] for item in impact_sensitivity],
        dtype=float,
    )
    impact_peak_values = impact_peak_values[np.isfinite(impact_peak_values)]
    impact_peak_min_force = (
        float(np.min(impact_peak_values)) if impact_peak_values.size else np.nan
    )
    impact_peak_max_force = (
        float(np.max(impact_peak_values)) if impact_peak_values.size else np.nan
    )
    impact_peak_mean_force = (
        float(np.mean(impact_peak_values)) if impact_peak_values.size else np.nan
    )
    impact_peak_range_force = (
        float(impact_peak_max_force - impact_peak_min_force)
        if np.isfinite(impact_peak_min_force) and np.isfinite(impact_peak_max_force)
        else np.nan
    )
    impact_peak_half_range_force = (
        0.5 * impact_peak_range_force
        if np.isfinite(impact_peak_range_force)
        else np.nan
    )

    landing_impulse_50ms_variation_percent = _relative_range_percent(
        [item["net_impulse_50ms"] for item in impact_sensitivity]
    )
    landing_impulse_100ms_variation_percent = _relative_range_percent(
        [item["net_impulse_100ms"] for item in impact_sensitivity]
    )

    design_max_force = float(config.get("design_max_force_n", 4905.0))
    peak_landing_utilization_percent = peak_landing_force / design_max_force * 100.0

    jump_height_impulse = takeoff_velocity**2 / (2.0 * prc.GRAVITY)
    jump_height_impulse_raw = takeoff_velocity_raw**2 / (2.0 * prc.GRAVITY)
    jump_height_impulse_force_corrected = (
        takeoff_velocity_force_corrected**2 / (2.0 * prc.GRAVITY)
    )
    jump_height_impulse_audit = (
        takeoff_velocity_audit**2 / (2.0 * prc.GRAVITY)
    )
    jump_height_flight = prc.GRAVITY * flight_time**2 / 8.0
    height_difference_percent = _percentage_difference(jump_height_impulse, jump_height_flight)

    ballistic_landing_velocity = takeoff_velocity - prc.GRAVITY * flight_time
    takeoff_landing_displacement = (
        takeoff_velocity * flight_time
        - 0.5 * prc.GRAVITY * flight_time**2
    )
    symmetric_flight_time = 2.0 * takeoff_velocity / prc.GRAVITY
    flight_time_deficit_s = symmetric_flight_time - flight_time
    landing_velocity_difference = abs(landing_velocity - ballistic_landing_velocity)

    unweighting_time = float(time_arr[unweighting_idx])
    unloading_duration = minimum_velocity_time - movement_start_time
    braking_duration = propulsive_time - minimum_velocity_time
    eccentric_duration = propulsive_time - movement_start_time
    unweight_to_propulsive_duration = propulsive_time - unweighting_time
    concentric_duration = takeoff_time - propulsive_time
    time_to_takeoff = takeoff_time - movement_start_time
    rsimod = jump_height_impulse / time_to_takeoff if time_to_takeoff > 0 else np.nan

    baseline_sample_count = max(1, baseline_end_idx - baseline_start_idx)
    # Filter membuat sampel berdekatan tidak sepenuhnya independen. Karena itu
    # ketidakpastian baseline memakai nilai konservatif maksimum antara SEM dan
    # 10% SD window, lalu dipropagasikan ke velocity/height impulse.
    baseline_force_uncertainty_n = max(
        body_weight_sd / np.sqrt(baseline_sample_count),
        0.10 * body_weight_sd,
    )
    baseline_velocity_uncertainty_m_s = (
        baseline_force_uncertainty_n * time_to_takeoff / body_mass
    )
    baseline_height_uncertainty_m = (
        abs(takeoff_velocity / prc.GRAVITY)
        * baseline_velocity_uncertainty_m_s
    )

    power = np.full(len(grf), np.nan, dtype=float)
    power[valid_velocity_primary] = (
        grf[valid_velocity_primary] * velocity_primary[valid_velocity_primary]
    )
    peak_power = float(np.nanmax(power[propulsive_idx:takeoff_idx]))
    peak_power_per_kg = peak_power / body_mass

    flight_start_idx = int(np.searchsorted(time_arr, takeoff_time, side="left"))
    flight_end_idx = int(np.searchsorted(time_arr, landing_time, side="left"))
    if flight_end_idx <= flight_start_idx:
        raise ValueError("Sampel flight tidak tersedia.")
    flight_zero = _compute_flight_zero_metrics(
        time_arr,
        raw_grf,
        raw_left,
        raw_right,
        takeoff_time,
        landing_time,
        fs,
        config,
    )
    # QC memakai central-flight, bukan sampel transisi tepat di sekitar crossing.
    flight_grf_mean = float(flight_zero["central_mean"])
    flight_grf_sd = float(flight_zero["central_sd"])
    flight_left_mean = float(flight_zero["left_mean"])
    flight_right_mean = float(flight_zero["right_mean"])
    flight_grf_full_mean = float(flight_zero["full_mean"])
    flight_grf_full_sd = float(flight_zero["full_sd"])
    flight_zero_guard_s = float(flight_zero["guard_s"])
    flight_zero_sample_count = int(flight_zero["central_sample_count"])
    flight_zero_equivalent_samples = float(
        flight_zero["central_equivalent_samples"]
    )
    flight_zero_start_time = float(flight_zero["central_start_time"])
    flight_zero_end_time = float(flight_zero["central_end_time"])
    flight_offset_percent_bw = abs(flight_grf_mean) / body_weight * 100.0

    mux_skew = np.asarray(data.get("mux_skew", []), dtype=float)
    mux_diag = _mux_order_diagnostics(mux_skew, fs, config)
    mux_skew_mean_fraction = (
        mux_diag["mean_abs_us"] / (1_000_000.0 / fs)
        if np.isfinite(mux_diag["mean_abs_us"])
        else np.nan
    )
    mux_skew_p95_fraction = float(mux_diag["p95_fraction"])
    mux_p95_abs_us = float(mux_diag["p95_abs_us"])
    mux_mean_abs_us = float(mux_diag["mean_abs_us"])
    mux_alignment_applied = bool(data.get("mux_alignment_applied", False))
    mux_alignment_coverage = float(data.get("mux_alignment_coverage", np.nan))
    mux_alignment_min_coverage = float(
        config.get("mux_alignment_min_coverage", 0.98)
    )

    mass_error = None
    if expected_mass is not None:
        mass_error = abs(body_mass - expected_mass) / expected_mass * 100.0

    baseline_left_share = baseline_left / body_weight * 100.0
    baseline_right_share = baseline_right / body_weight * 100.0
    baseline_share_difference = abs(baseline_left - baseline_right) / body_weight * 100.0

    record_adc_rejected = int(data.get("adc_rejected_record", data.get("adc_rejected_frames", 0)))
    record_malformed = int(data.get("malformed_rows_record", data.get("malformed_rows", 0)))
    post_landing_available_s = float(time_arr[-1] - landing_time)
    terminal_velocity_median = float(terminal_velocity["median"])
    terminal_velocity_sd = float(terminal_velocity["sd"])
    terminal_corrected_median = float(terminal_velocity_corrected["median"])
    terminal_corrected_sd = float(terminal_velocity_corrected["sd"])
    terminal_limit = float(config.get("cmj_terminal_velocity_limit_m_s", 0.03))
    raw_velocity_closure_ok = bool(
        np.isfinite(terminal_velocity_median)
        and abs(terminal_velocity_median) <= 0.05
        and (not np.isfinite(terminal_velocity_sd) or terminal_velocity_sd <= 0.05)
    )
    corrected_velocity_closure_ok = bool(
        np.isfinite(terminal_corrected_median)
        and abs(terminal_corrected_median) <= terminal_limit
        and (
            not np.isfinite(terminal_corrected_sd)
            or terminal_corrected_sd <= terminal_limit
        )
    )
    force_bias_percent_bw = (
        abs(force_bias_n) / body_weight * 100.0
        if np.isfinite(force_bias_n) and body_weight > 0.0
        else np.nan
    )
    live_post_stable = bool(data.get("cmj_post_stable"))
    post_stability_mismatch = bool(live_post_stable and post_window is None)
    live_takeoff_s = data.get("cmj_takeoff_live_s")
    live_landing_s = data.get(
        "cmj_landing_confirm_live_s", data.get("cmj_landing_live_s")
    )
    live_takeoff_difference_s = (
        abs(float(live_takeoff_s) - takeoff_time)
        if live_takeoff_s is not None and np.isfinite(live_takeoff_s)
        else np.nan
    )
    # Live landing memakai threshold konfirmasi, sedangkan landing_time adalah
    # first contact pada flight threshold. Perbandingan v6.2 tidak sejenis dan
    # menghasilkan warning palsu. v7.0 tetap membandingkan confirmation-to-confirmation.
    live_landing_difference_s = (
        abs(float(live_landing_s) - landing_confirmation_time)
        if live_landing_s is not None and np.isfinite(live_landing_s)
        else np.nan
    )
    landing_contact_to_confirmation_s = float(
        landing_confirmation_time - landing_time
    )

    acquisition_reasons: list[str] = []
    acquisition_note_reasons: list[str] = []
    primary_review_reasons: list[str] = []
    primary_caution_reasons: list[str] = []
    flight_review_reasons: list[str] = []
    flight_caution_reasons: list[str] = []
    bilateral_review_reasons: list[str] = []
    bilateral_caution_reasons: list[str] = []
    impact_review_reasons: list[str] = []
    impact_caution_reasons: list[str] = []
    rfd_review_reasons: list[str] = []
    rfd_caution_reasons: list[str] = []
    audit_caution_reasons: list[str] = []
    audit_notes: list[str] = []

    gap_status, gap_reasons = prc.classify_gap_severity(
        sampling_diag,
        hard_max_consecutive_missing=int(
            config.get("cmj_gap_hard_max_consecutive_missing", 3)
        ),
        hard_missing_fraction=float(
            config.get("cmj_gap_hard_missing_fraction", 0.0025)
        ),
        hard_duration_s=float(
            config.get("cmj_gap_hard_duration_s", 0.010)
        ),
    )
    if gap_status == prc.TRIAL_STATUS_REPEAT:
        acquisition_reasons.extend(gap_reasons)
    elif gap_status == "NOTE":
        acquisition_note_reasons.extend(gap_reasons)

    if record_adc_rejected != 0:
        acquisition_reasons.append("terdapat frame ADC yang ditolak")

    checksum_error_rows = int(data.get("checksum_error_rows", 0) or 0)
    protocol_error_counts = data.get("protocol_error_counts", {}) or {}
    transport_diag = data.get("serial_transport_diag", {}) or {}
    nul_noise_lines = int(transport_diag.get("noise_only_lines", 0) or 0)
    nul_bytes_seen = int(transport_diag.get("nul_bytes_seen", 0) or 0)

    # Malformed/NUL tidak otomatis menggagalkan trial. Yang menentukan apakah
    # data benar-benar hilang adalah sequence frame dan gap severity di atas.
    # Dengan firmware FP1, checksum memastikan frame korup tidak lolos parser.
    if record_malformed != 0:
        audit_notes.append(
            f"terdapat {record_malformed} baris non-frame/malformed; "
            "validitas data dinilai dari CRC dan kontinuitas frame"
        )
    if checksum_error_rows > 0:
        acquisition_note_reasons.append(
            f"terdapat {checksum_error_rows} frame dengan checksum CRC tidak valid"
        )
    if nul_noise_lines > 0 or nul_bytes_seen > 0:
        audit_notes.append(
            f"transport mendeteksi noise NUL: line={nul_noise_lines}, byte={nul_bytes_seen}; "
            "noise tanpa frame diabaikan dan tidak dianggap sampel"
        )
    if protocol_error_counts:
        audit_notes.append(
            "diagnostik parser serial: "
            + ", ".join(f"{key}={value}" for key, value in sorted(protocol_error_counts.items()))
        )

    if data.get("cmj_capture_complete") is False:
        acquisition_reasons.append(
            "akuisisi berhenti sebelum landing/post-monitor selesai"
        )

    primary_baseline_cv_limit = float(
        config.get("cmj_primary_baseline_cv_limit", 0.010)
    )
    primary_baseline_slope_limit = float(
        config.get("cmj_primary_baseline_slope_limit_n_s", 5.0)
    )
    if float(baseline["cv"]) > primary_baseline_cv_limit:
        primary_review_reasons.append(
            "baseline sebelum gerakan terlalu bervariasi untuk impulse–momentum"
        )
    if abs(float(baseline["slope"])) > primary_baseline_slope_limit:
        primary_review_reasons.append(
            "baseline sebelum gerakan masih memiliki tren gaya yang besar"
        )
    if baseline_gap_to_movement_s < -1.0 / fs:
        primary_review_reasons.append(
            "window baseline bertumpang tindih dengan movement onset"
        )
    elif baseline_gap_to_movement_s > 2.0:
        primary_caution_reasons.append(
            "window baseline terakhir masih lebih dari dua detik sebelum gerakan"
        )
    if not baseline_refined:
        primary_caution_reasons.append(
            "baseline dua-pass tidak tersedia; program memakai baseline stabil awal"
        )

    if impulse_velocity_difference > 0.010:
        primary_review_reasons.append(
            "integral kurva dan impulse/mass berbeda lebih dari 0.01 m/s"
        )
    impulse_height_threshold_range_cm = (
        impulse_height_threshold_range_m * 100.0
        if np.isfinite(impulse_height_threshold_range_m)
        else np.nan
    )
    threshold_caution_cm = float(
        config.get("cmj_impulse_height_threshold_caution_cm", 0.75)
    )
    threshold_review_cm = float(
        config.get("cmj_impulse_height_threshold_review_cm", 1.50)
    )
    if (
        np.isfinite(impulse_height_threshold_range_cm)
        and impulse_height_threshold_range_cm > threshold_review_cm
    ):
        primary_review_reasons.append(
            "tinggi impulse terlalu sensitif terhadap threshold take-off"
        )
    elif (
        np.isfinite(impulse_height_threshold_range_cm)
        and impulse_height_threshold_range_cm > threshold_caution_cm
    ):
        primary_caution_reasons.append(
            "tinggi impulse cukup sensitif terhadap threshold take-off"
        )

    if mass_error is not None and mass_error > 5.0:
        primary_review_reasons.append("error massa referensi lebih dari 5%")
    elif mass_error is not None and mass_error > 3.0:
        primary_caution_reasons.append("error massa referensi sedikit melebihi 3%")

    minimum_effective_fs = float(
        config.get("cmj_min_effective_sampling_hz", 200.0)
    )
    if fs < 150.0:
        primary_review_reasons.append("sampling rate efektif di bawah 150 Hz")
    elif fs < minimum_effective_fs:
        impact_caution_reasons.append(
            "sampling rate efektif di bawah 200 Hz; peak landing memiliki resolusi terbatas"
        )
        rfd_caution_reasons.append(
            "sampling rate efektif di bawah 200 Hz; loading rate memiliki resolusi terbatas"
        )

    flight_offset_caution = float(
        config.get("cmj_flight_offset_caution_percent_bw", 1.0)
    )
    flight_offset_review = float(
        config.get("cmj_flight_offset_review_percent_bw", 3.0)
    )
    if (
        flight_offset_percent_bw > flight_offset_review
        or flight_grf_sd > max(10.0, 0.02 * body_weight)
    ):
        flight_review_reasons.append(
            "central-flight GRF tidak cukup dekat nol"
        )
    elif flight_offset_percent_bw > flight_offset_caution:
        flight_caution_reasons.append(
            "offset gaya central-flight melebihi 1% BW"
        )
    if flight_zero_sample_count < int(
        config.get("cmj_flight_zero_min_samples", 12)
    ):
        flight_caution_reasons.append(
            "window central-flight memiliki terlalu sedikit sampel untuk audit zero-force"
        )

    if np.isfinite(height_difference_percent) and height_difference_percent > 10.0:
        flight_review_reasons.append(
            "tinggi flight-time berbeda lebih dari 10% dari hasil impulse utama"
        )
    elif np.isfinite(height_difference_percent) and height_difference_percent > 5.0:
        flight_caution_reasons.append(
            "tinggi flight-time berbeda lebih dari 5% dari hasil impulse utama"
        )
    if (
        np.isfinite(flight_time_threshold_range_s)
        and flight_time_threshold_range_s > 0.020
    ):
        flight_review_reasons.append(
            "flight time sensitif lebih dari 20 ms terhadap threshold ±5 N"
        )

    ballistic_caution = float(
        config.get("cmj_ballistic_velocity_caution_m_s", 0.080)
    )
    ballistic_review = float(
        config.get("cmj_ballistic_velocity_review_m_s", 0.150)
    )
    if landing_velocity_difference > ballistic_review:
        flight_review_reasons.append(
            "velocity landing dari integral tidak konsisten dengan prediksi balistik"
        )
    elif landing_velocity_difference > ballistic_caution:
        flight_caution_reasons.append(
            "velocity landing berbeda cukup besar dari prediksi balistik"
        )

    if np.isfinite(sampling_jitter_fraction) and sampling_jitter_fraction > 0.05:
        acquisition_reasons.append(
            "jitter timestamp melebihi 5% periode sampel"
        )

    # Audit closure dimulai sesudah landing, sehingga secara konstruksi
    # tidak mengubah take-off velocity maupun jump height utama.
    if not zupt_applied:
        if raw_velocity_closure_ok:
            audit_notes.append(
                "audit closure tidak diperlukan; velocity terminal mentah sudah memadai"
            )
        elif post_window is None:
            audit_caution_reasons.append(
                "audit closure tidak dapat diterapkan karena window post-landing stabil tidak ditemukan"
            )
        else:
            audit_caution_reasons.append(
                "audit closure ditolak karena gaya ekuivalen/durasi koreksi tidak memenuhi batas"
            )
    elif not corrected_velocity_closure_ok:
        audit_caution_reasons.append(
            "velocity terminal audit setelah closure belum kembali dekat nol"
        )
    if zupt_applied and abs(zupt_takeoff_adjustment) > 1e-6:
        audit_caution_reasons.append(
            "audit closure mengubah velocity pra-landing; periksa implementasi karena seharusnya nol"
        )
    if np.isfinite(force_bias_percent_bw) and force_bias_percent_bw > 3.0:
        audit_caution_reasons.append(
            "perubahan offset gaya pre–post melebihi 3% BW; koreksi drift tidak dipakai sebagai hasil utama"
        )
    elif np.isfinite(force_bias_percent_bw) and force_bias_percent_bw > 1.5:
        audit_caution_reasons.append("perubahan offset gaya pre–post melebihi 1.5% BW")
    if post_stability_mismatch:
        audit_caution_reasons.append(
            "deteksi stabil live dan offline tidak identik; window offline dipakai untuk audit"
        )
    live_event_limit_s = float(
        config.get("cmj_live_event_tolerance_samples", 2.0)
    ) / fs
    # v8.5: perbedaan event LIVE vs OFFLINE adalah audit diagnostik saja.
    # Hasil akhir menggunakan event OFFLINE dari rekaman lengkap. Selama event offline
    # dapat dianalisis dan core validity tetap PASS, selisih >2 sampel tidak menurunkan
    # trial menjadi USABLE_WITH_NOTE dan tidak memaksa pengulangan.
    if (
        np.isfinite(live_takeoff_difference_s)
        and live_takeoff_difference_s > live_event_limit_s
    ):
        audit_notes.append(
            "diagnostik: waktu take-off live dan offline berbeda lebih dari dua sampel; "
            "event offline dipakai untuk hasil akhir"
        )
    if (
        np.isfinite(live_landing_difference_s)
        and live_landing_difference_s > live_event_limit_s
    ):
        audit_notes.append(
            "diagnostik: waktu landing-confirm live dan offline berbeda lebih dari dua sampel; "
            "event offline dipakai untuk hasil akhir"
        )
    if post_landing_available_s < 2.0:
        audit_caution_reasons.append("durasi post-landing kurang dari 2 detik")
    if not live_post_stable and post_window is None:
        audit_caution_reasons.append(
            "akuisisi live dan offline tidak menemukan kestabilan post-landing"
        )

    # ------------------------------------------------------------------
    # TEMPORAL / MUX QC v8.0
    #
    # Raw AIN01<->AIN23 separation sebagai FRAKSI frame tetap dilaporkan
    # sebagai diagnostik saja. Pada dua kelompok MUX sequential, rasio ini
    # secara arsitektur cenderung berada dekat setengah frame.
    # Kualitas temporal dinilai dari separation absolut, alternating order,
    # alignment coverage, dan effective frame rate.
    # ------------------------------------------------------------------
    mux_abs_caution_us = float(
        config.get("cmj_mux_skew_abs_caution_us", 1500.0)
    )
    mux_abs_review_us = float(
        config.get("cmj_mux_skew_abs_review_us", 2500.0)
    )
    temporal_target_fs = float(
        config.get("cmj_temporal_target_frame_rate_hz", 300.0)
    )
    alignment_good = bool(
        mux_alignment_applied
        and np.isfinite(mux_alignment_coverage)
        and mux_alignment_coverage >= mux_alignment_min_coverage
        and mux_diag["policy"] == "BALANCED_ALTERNATING"
    )

    temporal_qc = "PASS"
    temporal_reasons: list[str] = []

    if not mux_alignment_applied:
        temporal_qc = "REVIEW"
        temporal_reasons.append("MUX time-alignment tidak diterapkan")
    elif (
        np.isfinite(mux_alignment_coverage)
        and mux_alignment_coverage < mux_alignment_min_coverage
    ):
        temporal_qc = "REVIEW"
        temporal_reasons.append("cakupan MUX time-alignment di bawah batas minimum")
    elif mux_diag["policy"] != "BALANCED_ALTERNATING":
        temporal_qc = "REVIEW"
        temporal_reasons.append("urutan MUX CMJ tidak balanced-alternating")
    elif np.isfinite(mux_p95_abs_us) and mux_p95_abs_us > mux_abs_review_us:
        temporal_qc = "REVIEW"
        temporal_reasons.append(
            f"raw MUX separation P95 {mux_p95_abs_us:.0f} us melebihi batas review "
            f"{mux_abs_review_us:.0f} us walaupun alignment valid"
        )
    elif np.isfinite(mux_p95_abs_us) and mux_p95_abs_us > mux_abs_caution_us:
        temporal_qc = "CAUTION"
        temporal_reasons.append(
            f"raw MUX separation P95 {mux_p95_abs_us:.0f} us melebihi target "
            f"{mux_abs_caution_us:.0f} us"
        )

    if np.isfinite(fs) and fs < temporal_target_fs:
        if temporal_qc == "PASS":
            temporal_qc = "CAUTION"
        temporal_reasons.append(
            f"effective frame rate {fs:.1f} Hz masih di bawah target "
            f"{temporal_target_fs:.0f} Hz untuk metrik bilateral cepat"
        )

    # v8.5: Temporal/MUX dipisahkan dari biomekanik bilateral.
    # REVIEW temporal adalah masalah sistem/core, sedangkan CAUTION temporal
    # tetap menjadi catatan tanpa mengubah hasil utama impulse–momentum.

    if np.isfinite(net_contrib_diff) and net_contrib_diff > 20.0:
        bilateral_review_reasons.append("asimetri net propulsive impulse lebih dari 20%")
    elif np.isfinite(net_contrib_diff) and net_contrib_diff > 15.0:
        bilateral_caution_reasons.append("asimetri net propulsive impulse 15–20%")
    if np.isfinite(gross_contrib_diff) and gross_contrib_diff > 20.0:
        bilateral_review_reasons.append("asimetri gross propulsive impulse lebih dari 20%")
    elif np.isfinite(gross_contrib_diff) and gross_contrib_diff > 15.0:
        bilateral_caution_reasons.append("asimetri gross propulsive impulse 15–20%")

    min_20_80_samples = float(
        config.get("cmj_rfd_20_80_min_samples", 6.0)
    )
    early_min_samples = int(
        config.get("cmj_early_loading_min_samples", 6)
    )
    early_r2_caution = float(
        config.get("cmj_early_loading_r2_caution", 0.80)
    )
    if (
        np.isfinite(landing_20_80_samples)
        and landing_20_80_samples < min_20_80_samples
    ):
        rfd_caution_reasons.append(
            "RFD 20–80% direpresentasikan kurang dari enam sampel; gunakan slope window tetap sebagai estimasi yang lebih stabil"
        )
    if early_loading_primary_sample_count < early_min_samples:
        rfd_review_reasons.append(
            "window loading-rate utama memiliki terlalu sedikit sampel"
        )
    elif (
        np.isfinite(early_loading_primary_r2)
        and early_loading_primary_r2 < early_r2_caution
    ):
        rfd_caution_reasons.append(
            "kurva gaya pada window loading-rate utama tidak cukup linear"
        )
    peak_var_limit = float(
        config.get("cmj_impact_peak_variation_caution_percent", 5.0)
    )
    rfd_var_limit = float(
        config.get("cmj_rfd_variation_caution_percent", 20.0)
    )
    if (
        np.isfinite(impact_peak_variation_percent)
        and impact_peak_variation_percent > peak_var_limit
    ):
        cutoff_text = "/".join(
            f"{item['cutoff_hz']:.0f}Hz={item['peak_force']:.0f}N"
            for item in impact_sensitivity
            if np.isfinite(item.get("peak_force", np.nan))
        )
        impact_caution_reasons.append(
            f"peak landing berubah {impact_peak_variation_percent:.1f}% pada audit "
            f"cutoff ({cutoff_text}); nilai utama tetap memakai "
            f"{impact_filter_cutoff_hz:.0f} Hz"
        )
    if (
        np.isfinite(impact_rfd_variation_percent)
        and impact_rfd_variation_percent > rfd_var_limit
    ):
        rfd_caution_reasons.append(
            "loading rate window tetap sensitif terhadap pilihan cutoff filter impact"
        )
    second_peak_limit = float(
        config.get("cmj_landing_second_peak_ratio_caution", 0.50)
    )
    landing_pattern = "SINGLE_DOMINANT_PEAK"
    if landing_peak_count >= 2:
        meaningful_second_peak = bool(
            np.isfinite(landing_second_peak_ratio)
            and landing_second_peak_ratio >= second_peak_limit
        )
        if meaningful_second_peak:
            if bilateral_landing_order == "LEFT_FIRST":
                landing_pattern = "STAGGERED_LEFT_FIRST"
            elif bilateral_landing_order == "RIGHT_FIRST":
                landing_pattern = "STAGGERED_RIGHT_FIRST"
            elif bilateral_landing_order == "SIMULTANEOUS":
                landing_pattern = "DOUBLE_IMPACT_SIMULTANEOUS"
            else:
                landing_pattern = "BIMODAL_UNRESOLVED"
        else:
            landing_pattern = "MINOR_SECONDARY_PEAK"

    if landing_pattern == "STAGGERED_LEFT_FIRST":
        impact_caution_reasons.append(
            "landing bimodal dijelaskan oleh kontak bilateral bertahap: kaki kiri lebih dulu"
        )
    elif landing_pattern == "STAGGERED_RIGHT_FIRST":
        impact_caution_reasons.append(
            "landing bimodal dijelaskan oleh kontak bilateral bertahap: kaki kanan lebih dulu"
        )
    elif landing_pattern == "DOUBLE_IMPACT_SIMULTANEOUS":
        # Dua peak dengan onset bilateral hampir simultan lebih tepat
        # dilaporkan sebagai pola biomekanik, bukan otomatis error data.
        audit_notes.append(
            "landing memiliki dua puncak bermakna dengan onset kiri-kanan hampir "
            "simultan; pola ini kompatibel dengan forefoot-heel/rebound atau "
            "koreksi postur dan tidak otomatis dianggap kegagalan akuisisi"
        )
    elif landing_pattern == "BIMODAL_UNRESOLVED":
        impact_caution_reasons.append(
            "landing memiliki dua puncak bermakna tetapi urutan kontak kiri-kanan tidak dapat dipastikan"
        )
    elif landing_peak_count >= 3:
        impact_caution_reasons.append(
            "terdapat beberapa puncak impact landing dalam 0.5 detik pertama"
        )

    if peak_landing_utilization_percent > 80.0:
        impact_review_reasons.append(
            "peak landing melebihi 80% batas gaya desain"
        )
    elif peak_landing_utilization_percent > 60.0:
        impact_caution_reasons.append("peak landing melebihi 60% batas gaya desain")

    # ------------------------------------------------------------------
    # FINAL DECISION POLICY v8.5
    # ------------------------------------------------------------------
    # Core validity = hal yang menentukan apakah hasil utama boleh dipakai.
    # Secondary QC = informasi teknik gerakan / audit yang tetap dilaporkan,
    # tetapi tidak memaksa subjek mengulang bila core validity tetap baik.
    # Threshold individual TIDAK dilonggarkan.
    primary_review_all = acquisition_reasons + primary_review_reasons
    if primary_review_all:
        impulse_primary_quality = "REVIEW"
    elif primary_caution_reasons:
        impulse_primary_quality = "PASS_WITH_CAUTION"
    else:
        impulse_primary_quality = "PASS"

    if flight_review_reasons:
        flight_time_qc = "REVIEW"
    elif flight_caution_reasons:
        flight_time_qc = "CAUTION"
    else:
        flight_time_qc = "CONSISTENT"

    # Bilateral di sini murni biomekanik/asimetri; masalah timing MUX sudah
    # dipisahkan dalam temporal_qc.
    if bilateral_review_reasons:
        bilateral_qc = "REVIEW"
    elif bilateral_caution_reasons:
        bilateral_qc = "CAUTION"
    else:
        bilateral_qc = "PASS"

    if impact_review_reasons:
        impact_qc = "REVIEW"
    elif impact_caution_reasons:
        impact_qc = "CAUTION"
    else:
        impact_qc = "PASS"

    if rfd_review_reasons:
        rfd_qc = "REVIEW"
    elif rfd_caution_reasons:
        rfd_qc = "CAUTION"
    else:
        rfd_qc = "PASS"

    # Wajib ulang / jangan dipakai sebagai hasil utama:
    # 1) akuisisi rusak, 2) impulse utama tidak valid, 3) temporal/MUX hard review,
    # 4) peak landing >80% batas desain (periksa keamanan sebelum mengulang).
    core_repeat_reasons = list(primary_review_all)
    temporal_secondary_reasons: list[str] = []
    if temporal_qc == "REVIEW":
        # Cabang temporal hard memakai elif, sehingga alasan pertama adalah
        # penyebab REVIEW. Catatan tambahan (mis. target frame-rate 300 Hz)
        # tetap sekunder dan tidak perlu ikut menjadi alasan wajib ulang.
        if temporal_reasons:
            core_repeat_reasons.append(temporal_reasons[0])
            temporal_secondary_reasons.extend(temporal_reasons[1:])
    core_repeat_reasons.extend(impact_review_reasons)

    # Semua QC sekunder tetap disimpan sebagai catatan. REVIEW pada komponen
    # sekunder berarti metrik tersebut perlu hati-hati, bukan otomatis trial gagal.
    secondary_note_reasons = (
        list(acquisition_note_reasons)
        + list(primary_caution_reasons)
        + list(flight_review_reasons) + list(flight_caution_reasons)
        + list(bilateral_review_reasons) + list(bilateral_caution_reasons)
        + list(impact_caution_reasons)
        + list(rfd_review_reasons) + list(rfd_caution_reasons)
        + list(audit_caution_reasons)
        + list(temporal_secondary_reasons)
    )
    if temporal_qc == "CAUTION":
        secondary_note_reasons.extend(temporal_reasons)

    trial_quality = prc.final_trial_status(
        core_repeat_reasons,
        secondary_note_reasons,
    )
    core_validity = "FAIL" if core_repeat_reasons else "PASS"
    trial_usable = not prc.status_requires_repeat(trial_quality)

    # Alias dipertahankan untuk kompatibilitas output lama.
    review_reasons = core_repeat_reasons
    kinetic_review_reasons = primary_review_reasons
    secondary_qc_reasons = (
        list(flight_review_reasons) + list(flight_caution_reasons)
        + list(bilateral_review_reasons) + list(bilateral_caution_reasons)
        + list(impact_caution_reasons)
        + list(rfd_review_reasons) + list(rfd_caution_reasons)
    )
    caution_reasons = secondary_note_reasons

    report = {
        "software_version": SOFTWARE_VERSION,
        "calibration_version": str(
            data.get("calibration_version", config.get("calibration_version", "-"))
        ),
        "zero_gate_info": data.get("zero_gate_info"),
        "mux_alignment_applied": mux_alignment_applied,
        "mux_alignment_coverage": mux_alignment_coverage,
        "primary_method": PRIMARY_METHOD,
        "impulse_primary_quality": impulse_primary_quality,
        "flight_time_qc": flight_time_qc,
        "bilateral_qc": bilateral_qc,
        "impact_qc": impact_qc,
        "rfd_qc": rfd_qc,
        "time": time_arr,
        "fs": fs,
        "sampling_diag": sampling_diag,
        "sampling_jitter_fraction": sampling_jitter_fraction,
        "serial_diag": serial_diag,
        "grf": grf,
        "grf_event": grf_event,
        "impact_grf": impact_grf,
        "impact_filter_cutoff_hz": impact_filter_cutoff_hz,
        "adc_nominal_sps": int(data.get("adc_nominal_sps", 0)),
        "left_force": left_force,
        "right_force": right_force,
        "velocity": velocity_primary,
        "velocity_primary": velocity_primary,
        "velocity_raw": velocity_raw,
        "velocity_force_corrected": velocity_force_corrected,
        "velocity_audit": velocity_audit,
        "net_force_raw": net_force,
        "net_force_corrected": net_force_corrected,
        "dynamic_baseline_force": dynamic_baseline_force,
        "power": power,
        "baseline": baseline,
        "initial_baseline": initial_baseline,
        "baseline_refined": baseline_refined,
        "baseline_gap_to_movement_s": baseline_gap_to_movement_s,
        "baseline_force_uncertainty_n": baseline_force_uncertainty_n,
        "baseline_velocity_uncertainty_m_s": baseline_velocity_uncertainty_m_s,
        "baseline_height_uncertainty_m": baseline_height_uncertainty_m,
        "body_weight": body_weight,
        "body_mass": body_mass,
        "mass_error": mass_error,
        "baseline_left": baseline_left,
        "baseline_right": baseline_right,
        "baseline_left_share": baseline_left_share,
        "baseline_right_share": baseline_right_share,
        "baseline_share_difference": baseline_share_difference,
        "onset_threshold": onset_threshold,
        "unweight_threshold": unweight_threshold,
        "flight_threshold": flight_threshold,
        "landing_confirmation_threshold": landing_confirmation_threshold,
        "landing_confirmation_idx": landing_confirmation_idx,
        "landing_confirmation_time": landing_confirmation_time,
        "landing_contact_to_confirmation_s": landing_contact_to_confirmation_s,
        "flight_time_threshold_range_s": flight_time_threshold_range_s,
        "impulse_height_threshold_range_m": impulse_height_threshold_range_m,
        "impulse_height_threshold_range_cm": impulse_height_threshold_range_cm,
        "movement_start_idx": movement_start_idx,
        "movement_start_time": movement_start_time,
        "unweighting_idx": unweighting_idx,
        "unweighting_time": unweighting_time,
        "propulsive_idx": propulsive_idx,
        "propulsive_time": propulsive_time,
        "takeoff_idx": takeoff_idx,
        "takeoff_time": takeoff_time,
        "landing_idx": landing_idx,
        "landing_time": landing_time,
        "flight_time": flight_time,
        "minimum_force": minimum_force,
        "peak_force": peak_force,
        "peak_landing_force": peak_landing_force,
        "peak_landing_force_kinetic": peak_landing_force_kinetic,
        "peak_landing_time": peak_landing_time,
        "landing_time_to_peak": landing_time_to_peak,
        "landing_loading_rate": landing_loading_rate,
        "landing_loading_rate_20_80": landing_loading_rate_20_80,
        "landing_time_to_peak_samples": landing_time_to_peak_samples,
        "landing_20_80_duration_s": landing_20_80_duration_s,
        "landing_20_80_samples": landing_20_80_samples,
        "landing_peak_count": landing_peak_count,
        "landing_peak_forces": landing_peak_forces,
        "landing_peak_times": landing_peak_times,
        "landing_peak_prominences": landing_peak_prominences,
        "landing_second_peak_ratio": landing_second_peak_ratio,
        "landing_first_peak_force": landing_first_peak_force,
        "landing_first_peak_time": landing_first_peak_time,
        "landing_second_peak_force": landing_second_peak_force,
        "landing_second_peak_time": landing_second_peak_time,
        "landing_interpeak_interval_s": landing_interpeak_interval_s,
        "landing_dominant_peak_sequence": landing_dominant_peak_sequence,
        "landing_pattern": landing_pattern,
        "bilateral_landing_order": bilateral_landing_order,
        "side_landing_threshold_n": side_landing_threshold,
        "left_landing_time": float(left_landing_time),
        "right_landing_time": float(right_landing_time),
        "left_landing_idx": left_landing_idx,
        "right_landing_idx": right_landing_idx,
        "landing_lr_delay_s": float(landing_lr_delay_s),
        "landing_lr_abs_delay_s": float(landing_lr_abs_delay_s),
        "staggered_landing_threshold_s": staggered_threshold_s,
        "early_loading_metrics": early_loading_metrics,
        "early_loading_primary_window_s": early_loading_primary_window_s,
        "early_loading_primary_actual_window_s": early_loading_primary_actual_window_s,
        "early_loading_primary_rate_n_s": early_loading_primary_rate_n_s,
        "early_loading_primary_r2": early_loading_primary_r2,
        "early_loading_primary_sample_count": early_loading_primary_sample_count,
        "early_loading_primary_equivalent_samples": early_loading_primary_equivalent_samples,
        "early_loading_primary_net_impulse_n_s": early_loading_primary_net_impulse_n_s,
        "landing_impulses": landing_impulses,
        "landing_gross_impulse_50ms": landing_gross_impulse_50ms,
        "landing_net_impulse_50ms": landing_net_impulse_50ms,
        "landing_gross_impulse_100ms": landing_gross_impulse_100ms,
        "landing_net_impulse_100ms": landing_net_impulse_100ms,
        "impact_sensitivity": impact_sensitivity,
        "impact_peak_variation_percent": impact_peak_variation_percent,
        "impact_peak_min_force": impact_peak_min_force,
        "impact_peak_max_force": impact_peak_max_force,
        "impact_peak_mean_force": impact_peak_mean_force,
        "impact_peak_range_force": impact_peak_range_force,
        "impact_peak_half_range_force": impact_peak_half_range_force,
        "landing_impulse_50ms_variation_percent": landing_impulse_50ms_variation_percent,
        "landing_impulse_100ms_variation_percent": landing_impulse_100ms_variation_percent,
        "impact_rfd_variation_percent": impact_rfd_variation_percent,
        "impact_rfd_20_80_variation_percent": impact_rfd_20_80_variation_percent,
        "design_max_force": design_max_force,
        "peak_landing_utilization_percent": peak_landing_utilization_percent,
        "total_net_impulse": total_net_impulse,
        "total_net_impulse_corrected": total_net_impulse_corrected,
        "pre_propulsive_net_impulse": pre_propulsive_net_impulse,
        "eccentric_net_impulse": eccentric_net_impulse,
        "unloading_impulse": unloading_impulse,
        "braking_impulse": braking_impulse,
        "propulsive_net_impulse": propulsive_net_impulse,
        "left_net_impulse": left_net_impulse,
        "right_net_impulse": right_net_impulse,
        "left_net_contrib": left_net_contrib,
        "right_net_contrib": right_net_contrib,
        "net_contrib_diff": net_contrib_diff,
        "left_gross_impulse": left_gross_impulse,
        "right_gross_impulse": right_gross_impulse,
        "left_gross_contrib": left_gross_contrib,
        "right_gross_contrib": right_gross_contrib,
        "gross_contrib_diff": gross_contrib_diff,
        "net_leg_sum": net_leg_sum,
        "gross_leg_sum": gross_leg_sum,
        "takeoff_velocity": takeoff_velocity,
        "takeoff_velocity_primary": takeoff_velocity,
        "takeoff_velocity_curve": takeoff_velocity_curve,
        "takeoff_velocity_raw": takeoff_velocity_raw,
        "takeoff_velocity_force_corrected": takeoff_velocity_force_corrected,
        "takeoff_velocity_audit": takeoff_velocity_audit,
        "velocity_from_total_impulse": velocity_from_total_impulse,
        "velocity_from_corrected_impulse": velocity_from_corrected_impulse,
        "impulse_velocity_difference": impulse_velocity_difference,
        "force_drift_takeoff_adjustment": force_drift_takeoff_adjustment,
        "zupt_takeoff_adjustment": zupt_takeoff_adjustment,
        "total_takeoff_adjustment": total_takeoff_adjustment,
        "force_bias_n": force_bias_n,
        "force_bias_percent_bw": force_bias_percent_bw,
        "force_bias_applied": force_bias_applied,
        "raw_post_residual": raw_post_residual,
        "force_corrected_post_residual": force_corrected_post_residual,
        "landing_velocity": landing_velocity,
        "ballistic_landing_velocity": ballistic_landing_velocity,
        "landing_velocity_difference": landing_velocity_difference,
        "zupt_applied": zupt_applied,
        "zupt_residual": zupt_residual,
        "closure_start_time": closure_start_time,
        "closure_end_time": closure_end_time,
        "closure_duration_s": closure_duration_s,
        "closure_equivalent_force_n": closure_equivalent_force_n,
        "closure_equivalent_force_percent_bw": closure_equivalent_force_percent_bw,
        "closure_post_residual": closure_post_residual,
        "closure_source": closure_source,
        "post_window": post_window,
        "post_landing_available_s": post_landing_available_s,
        "terminal_velocity": terminal_velocity,
        "terminal_velocity_corrected": terminal_velocity_corrected,
        "terminal_velocity_median": terminal_velocity_median,
        "terminal_velocity_sd": terminal_velocity_sd,
        "terminal_corrected_median": terminal_corrected_median,
        "terminal_corrected_sd": terminal_corrected_sd,
        "raw_velocity_closure_ok": raw_velocity_closure_ok,
        "corrected_velocity_closure_ok": corrected_velocity_closure_ok,
        "live_post_stable": live_post_stable,
        "post_stability_mismatch": post_stability_mismatch,
        "live_takeoff_difference_s": live_takeoff_difference_s,
        "live_landing_difference_s": live_landing_difference_s,
        "jump_height_impulse": jump_height_impulse,
        "jump_height_impulse_raw": jump_height_impulse_raw,
        "jump_height_impulse_force_corrected": jump_height_impulse_force_corrected,
        "jump_height_impulse_audit": jump_height_impulse_audit,
        "jump_height_flight": jump_height_flight,
        "height_difference_percent": height_difference_percent,
        "takeoff_landing_displacement": takeoff_landing_displacement,
        "symmetric_flight_time": symmetric_flight_time,
        "flight_time_deficit_s": flight_time_deficit_s,
        "minimum_velocity_time": minimum_velocity_time,
        "unloading_duration": unloading_duration,
        "braking_duration": braking_duration,
        "eccentric_duration": eccentric_duration,
        "unweight_to_propulsive_duration": unweight_to_propulsive_duration,
        "concentric_duration": concentric_duration,
        "time_to_takeoff": time_to_takeoff,
        "rsimod": rsimod,
        "peak_power": peak_power,
        "peak_power_per_kg": peak_power_per_kg,
        "flight_grf_mean": flight_grf_mean,
        "flight_grf_sd": flight_grf_sd,
        "flight_grf_full_mean": flight_grf_full_mean,
        "flight_grf_full_sd": flight_grf_full_sd,
        "flight_left_mean": flight_left_mean,
        "flight_right_mean": flight_right_mean,
        "flight_zero_guard_s": flight_zero_guard_s,
        "flight_zero_sample_count": flight_zero_sample_count,
        "flight_zero_equivalent_samples": flight_zero_equivalent_samples,
        "flight_zero_start_time": flight_zero_start_time,
        "flight_zero_end_time": flight_zero_end_time,
        "flight_offset_percent_bw": flight_offset_percent_bw,
        "minimum_velocity_before_propulsion": minimum_velocity_before_propulsion,
        "mux_skew": mux_skew,
        "mux_skew_mean_fraction": mux_skew_mean_fraction,
        "mux_skew_p95_fraction": mux_skew_p95_fraction,
        "mux_signed_mean_us": mux_diag["signed_mean_us"],
        "mux_mean_abs_us": mux_diag["mean_abs_us"],
        "mux_p95_abs_us": mux_diag["p95_abs_us"],
        "mux_alternation_fraction": mux_diag["alternation_fraction"],
        "mux_signed_bias_fraction": mux_diag["signed_bias_fraction"],
        "mux_policy": mux_diag["policy"],
        "mux_frame_period_us": mux_diag["frame_period_us"],
        "mux_temporal_resolution_ms": mux_diag["temporal_resolution_ms"],
        "temporal_qc": temporal_qc,
        "temporal_reasons": temporal_reasons,
        "record_adc_rejected": record_adc_rejected,
        "record_malformed": record_malformed,
        "acquisition_reasons": acquisition_reasons,
        "acquisition_note_reasons": acquisition_note_reasons,
        "gap_status": gap_status,
        "gap_reasons": gap_reasons,
        "kinetic_review_reasons": kinetic_review_reasons,
        "primary_review_reasons": primary_review_reasons,
        "secondary_qc_reasons": secondary_qc_reasons,
        "primary_caution_reasons": primary_caution_reasons,
        "flight_review_reasons": flight_review_reasons,
        "flight_caution_reasons": flight_caution_reasons,
        "bilateral_review_reasons": bilateral_review_reasons,
        "bilateral_caution_reasons": bilateral_caution_reasons,
        "impact_review_reasons": impact_review_reasons,
        "impact_caution_reasons": impact_caution_reasons,
        "rfd_review_reasons": rfd_review_reasons,
        "rfd_caution_reasons": rfd_caution_reasons,
        "audit_caution_reasons": audit_caution_reasons,
        "audit_notes": audit_notes,
        "caution_reasons": caution_reasons,
        "review_reasons": review_reasons,
        "core_repeat_reasons": core_repeat_reasons,
        "secondary_note_reasons": secondary_note_reasons,
        "core_validity": core_validity,
        "trial_usable": trial_usable,
        "trial_quality": trial_quality,
    }
    return report


def print_report(report: dict[str, Any], data: dict[str, Any], expected_mass):
    fs = report["fs"]
    mux_skew = report["mux_skew"]
    print("\n" + "=" * 82)
    print("CMJ KINETIC REPORT v8.5 — CRC-FRAMED / GAP-ROBUST / MUX-ALIGNED")
    print("=" * 82)
    print(f"Calibration Version : {report.get('calibration_version', '-')}")
    zero_gate_info = report.get("zero_gate_info") or {}
    print(f"Pre-trial Zero Gate : {zero_gate_info.get('status', 'NOT_AVAILABLE')}")
    print(f"Primary Method      : {report['primary_method']}")
    print(f"Primary Impulse Q   : {report['impulse_primary_quality']}")
    print(f"Flight-time QC      : {report['flight_time_qc']}")
    print(f"Bilateral QC        : {report['bilateral_qc']}")
    print(f"Impact QC           : {report['impact_qc']}")
    print(f"Landing Rate QC     : {report['rfd_qc']}")
    print(f"ADC Nominal Rate    : {report['adc_nominal_sps']} SPS")
    print(f"Effective Frame Rate: {fs:.2f} Hz")
    if np.isfinite(report["sampling_jitter_fraction"]):
        print(
            f"Timestamp Jitter    : {report['sampling_diag']['jitter_sd'] * 1000:.3f} ms "
            f"({report['sampling_jitter_fraction'] * 100:.2f}% periode)"
        )
    print(f"Impact Filter       : {report['impact_filter_cutoff_hz']:.1f} Hz")
    print(f"Serial Baud         : {int(data.get('serial_baud', prc.BAUD_RATE))}")
    print(f"Serial Lost (full)  : {report['serial_diag']['serial_lost_frames']}")
    print(f"ADC Rejected record : {report['record_adc_rejected']}")
    print(f"Malformed/non-frame: {report['record_malformed']}")
    print(f"Analysis Gaps       : {report['sampling_diag']['analysis_frame_gaps']}")
    print(
        f"Gap Severity        : {report.get('gap_status', '-')} | "
        f"events={report['sampling_diag'].get('gap_events', 0)}, "
        f"max-missing={report['sampling_diag'].get('max_consecutive_missing', 0)}, "
        f"fraction={report['sampling_diag'].get('missing_fraction', 0.0) * 100.0:.3f}%"
    )
    if len(mux_skew):
        print(
            f"MUX Skew abs        : mean={report['mux_mean_abs_us']:.1f} us, "
            f"P95={report['mux_p95_abs_us']:.1f} us, "
            f"max={np.max(np.abs(mux_skew)):.1f} us "
            f"(P95={report['mux_skew_p95_fraction'] * 100:.1f}% periode frame)"
        )
        alternation_text = (
            f"{report['mux_alternation_fraction'] * 100:.1f}%"
            if np.isfinite(report['mux_alternation_fraction']) else "n/a"
        )
        bias_text = (
            f"{report['mux_signed_bias_fraction'] * 100:.1f}%"
            if np.isfinite(report['mux_signed_bias_fraction']) else "n/a"
        )
        print(
            f"MUX Order Audit     : {report['mux_policy']} | "
            f"alternation={alternation_text} | signed bias={bias_text}"
        )
        coverage_text = (
            f"{report['mux_alignment_coverage'] * 100:.2f}%"
            if np.isfinite(report.get("mux_alignment_coverage", np.nan))
            else "n/a"
        )
        print(
            f"MUX Time Alignment  : "
            f"{'APPLIED' if report.get('mux_alignment_applied', False) else 'NOT_APPLIED'} "
            f"| coverage={coverage_text}"
        )
    print(f"CMJ Auto Capture    : {data.get('cmj_capture_message', 'legacy_or_unknown')}")
    if np.isfinite(report["live_takeoff_difference_s"]):
        print(
            f"Live/offline takeoff: {report['live_takeoff_difference_s'] * 1000:.1f} ms"
        )
    if np.isfinite(report["live_landing_difference_s"]):
        print(
            f"Live/offline land conf: {report['live_landing_difference_s'] * 1000:.1f} ms"
        )
    print(f"Core Validity       : {report.get('core_validity', '-')}")
    print(f"Trial Quality       : {report['trial_quality']}")
    print(f"Trial Usable        : {'YES' if report.get('trial_usable', False) else 'NO'}")
    print("-" * 82)

    if expected_mass is not None:
        print(
            f"Measured Mass       : {report['body_mass']:.2f} kg "
            f"(error terhadap referensi {report['mass_error']:.2f}%)"
        )
    else:
        print(f"Measured Mass       : {report['body_mass']:.2f} kg")
    print(
        f"Baseline L / R      : {report['baseline_left']:.1f} / "
        f"{report['baseline_right']:.1f} N "
        f"({report['baseline_left_share']:.1f}% / {report['baseline_right_share']:.1f}%)"
    )
    print(
        f"Baseline window     : {'REFINED/LATEST' if report['baseline_refined'] else 'INITIAL'} "
        f"| gap ke gerakan {report['baseline_gap_to_movement_s']:.3f} s"
    )
    print(
        f"Baseline GRF CV     : {report['baseline']['cv'] * 100:.3f}% "
        f"| slope {report['baseline']['slope']:+.3f} N/s"
    )
    print(
        f"Baseline uncertainty: ±{report['baseline_force_uncertainty_n']:.3f} N "
        f"→ ±{report['baseline_height_uncertainty_m'] * 100:.3f} cm pada tinggi impulse"
    )
    print(f"Movement Threshold  : {report['onset_threshold']:.1f} N")
    print(f"Unweight Threshold  : {report['unweight_threshold']:.1f} N")
    print(f"Flight Threshold    : {report['flight_threshold']:.1f} N")
    print(f"Landing Confirm Th. : {report['landing_confirmation_threshold']:.1f} N")
    print(
        f"Contact→confirmation: {report['landing_contact_to_confirmation_s'] * 1000:.1f} ms"
    )
    print(
        f"Central-flight GRF  : {report['flight_grf_mean']:.2f} ± "
        f"{report['flight_grf_sd']:.2f} N "
        f"({report['flight_offset_percent_bw']:.2f}% BW; "
        f"guard {report['flight_zero_guard_s'] * 1000:.0f} ms)"
    )
    print(
        f"Full-flight GRF     : {report['flight_grf_full_mean']:.2f} ± "
        f"{report['flight_grf_full_sd']:.2f} N"
    )
    print(
        f"Central offset L/R  : {report['flight_left_mean']:.2f} / "
        f"{report['flight_right_mean']:.2f} N "
        f"({report['flight_zero_equivalent_samples']:.1f} sampel ekuivalen)"
    )
    print("-" * 82)

    print(
        f"Minimum Force       : {report['minimum_force']:.2f} N "
        f"({report['minimum_force'] / report['body_weight']:.2f} BW)"
    )
    print(
        f"Peak Propulsive     : {report['peak_force']:.2f} N "
        f"({report['peak_force'] / report['body_weight']:.2f} BW)"
    )
    print(
        f"Peak Landing impact : {report['peak_landing_force']:.2f} N "
        f"({report['peak_landing_force'] / report['body_weight']:.2f} BW)"
    )
    print(
        f"Peak Landing kinetic: {report['peak_landing_force_kinetic']:.2f} N "
        "(30 Hz; untuk kinetika, bukan impact peak)"
    )
    print(
        f"Landing load use    : {report['peak_landing_utilization_percent']:.1f}% "
        f"dari batas desain {report['design_max_force']:.0f} N"
    )
    print(
        f"Landing time-to-peak: {report['landing_time_to_peak'] * 1000:.1f} ms "
        f"({report['landing_time_to_peak_samples']:.1f} sampel)"
    )
    if np.isfinite(report['landing_loading_rate']):
        print(f"Avg landing rate    : {report['landing_loading_rate']:.1f} N/s")
    if np.isfinite(report['landing_loading_rate_20_80']):
        resolution_text = (
            " | RESOLUTION-LIMITED"
            if report['landing_20_80_samples'] < 6.0
            else ""
        )
        print(
            f"20–80% landing rate : {report['landing_loading_rate_20_80']:.1f} N/s "
            f"({report['landing_20_80_samples']:.1f} sampel{resolution_text})"
        )
    for metric in report['early_loading_metrics']:
        if not np.isfinite(metric['slope_n_s']):
            continue
        print(
            f"Early {metric['requested_window_s'] * 1000:.0f} ms slope : "
            f"{metric['slope_n_s']:.1f} N/s "
            f"({metric['equivalent_samples']:.1f} sampel ekuivalen, "
            f"R²={metric['r2']:.3f})"
        )
    second_ratio_text = (
        f"{report['landing_second_peak_ratio'] * 100:.1f}%"
        if np.isfinite(report['landing_second_peak_ratio']) else "n/a"
    )
    print(
        f"Landing pattern     : {report['landing_pattern']} | "
        f"peak count={report['landing_peak_count']} | second/main={second_ratio_text}"
    )
    lr_delay_text = (
        f"{report['landing_lr_delay_s'] * 1000:+.1f} ms"
        if np.isfinite(report.get("landing_lr_delay_s", np.nan))
        else "n/a"
    )
    print(
        f"Bilateral landing   : {report['bilateral_landing_order']} | "
        f"R-L delay={lr_delay_text} | "
        f"threshold={report['side_landing_threshold_n']:.1f} N"
    )
    if report['landing_peak_count'] >= 2:
        print(
            f"Peak 1 / Peak 2     : {report['landing_first_peak_force']:.1f} / "
            f"{report['landing_second_peak_force']:.1f} N | "
            f"Δt={report['landing_interpeak_interval_s'] * 1000:.1f} ms | "
            f"dominant=#{report['landing_dominant_peak_sequence']}"
        )
    print(
        f"Impact filter sens. : peak range={report['impact_peak_variation_percent']:.2f}% | "
        f"early-rate range={report['impact_rfd_variation_percent']:.2f}% | "
        f"20–80 range={report['impact_rfd_20_80_variation_percent']:.2f}%"
    )
    if report.get("impact_sensitivity"):
        cutoff_summary = " | ".join(
            f"{item['cutoff_hz']:.0f}Hz={item['peak_force']:.1f}N"
            for item in report["impact_sensitivity"]
            if np.isfinite(item.get("peak_force", np.nan))
        )
        print(f"Impact cutoff peaks : {cutoff_summary}")
    if np.isfinite(report.get("impact_peak_min_force", np.nan)):
        print(
            f"Impact peak range   : {report['impact_peak_min_force']:.1f}–"
            f"{report['impact_peak_max_force']:.1f} N "
            f"(half-range ±{report['impact_peak_half_range_force']:.1f} N)"
        )
    print(
        f"Landing impulse 50ms: gross={report['landing_gross_impulse_50ms']:.3f} N.s | "
        f"net={report['landing_net_impulse_50ms']:.3f} N.s | "
        f"cutoff variation={report['landing_impulse_50ms_variation_percent']:.2f}%"
    )
    print(
        f"Landing impulse100ms: gross={report['landing_gross_impulse_100ms']:.3f} N.s | "
        f"net={report['landing_net_impulse_100ms']:.3f} N.s | "
        f"cutoff variation={report['landing_impulse_100ms_variation_percent']:.2f}%"
    )
    print(f"Total Net Impulse   : {report['total_net_impulse']:.2f} N.s")
    print(f"Unloading Impulse   : {report['unloading_impulse']:.2f} N.s")
    print(f"Braking Impulse     : {report['braking_impulse']:.2f} N.s")
    print(f"Eccentric Net Imp.  : {report['eccentric_net_impulse']:.2f} N.s")
    print(f"Propulsive Net Imp. : {report['propulsive_net_impulse']:.2f} N.s")
    print(
        f"Net L/R Impulse     : {report['left_net_impulse']:.2f} / "
        f"{report['right_net_impulse']:.2f} N.s"
    )
    print(
        f"Net L/R Contrib.    : {report['left_net_contrib']:.1f}% / "
        f"{report['right_net_contrib']:.1f}% "
        f"(diff {report['net_contrib_diff']:.1f}%)"
    )
    print(
        f"Gross L/R Impulse   : {report['left_gross_impulse']:.2f} / "
        f"{report['right_gross_impulse']:.2f} N.s"
    )
    print(
        f"Gross L/R Contrib.  : {report['left_gross_contrib']:.1f}% / "
        f"{report['right_gross_contrib']:.1f}% "
        f"(diff {report['gross_contrib_diff']:.1f}%)"
    )
    print("-" * 82)

    print(f"Minimum CoM Velocity: {report['minimum_velocity_before_propulsion']:.3f} m/s")
    print(f"Primary TO Velocity : {report['takeoff_velocity_primary']:.3f} m/s")
    print(f"Curve-integral check: {report['takeoff_velocity_curve']:.3f} m/s")
    print(f"Curve vs Imp./Mass  : {report['impulse_velocity_difference']:.4f} m/s")
    print(f"Force-drift audit v : {report['takeoff_velocity_force_corrected']:.3f} m/s")
    print(f"Post-ZUPT audit v   : {report['takeoff_velocity_audit']:.3f} m/s")
    print(f"Closure Δv at TO   : {report['zupt_takeoff_adjustment']:+.4f} m/s")
    print(
        f"Raw post residual   : {report['raw_post_residual']:+.3f} m/s "
        "(audit; tidak mengubah hasil utama)"
    )
    if np.isfinite(report['force_corrected_post_residual']):
        print(
            f"Force-corr residual : {report['force_corrected_post_residual']:+.3f} m/s"
        )
    if np.isfinite(report['closure_equivalent_force_n']):
        print(
            f"Closure equivalent F: {report['closure_equivalent_force_n']:+.3f} N "
            f"({report['closure_equivalent_force_percent_bw']:.3f}% BW) "
            f"selama {report['closure_duration_s']:.3f} s"
        )
    if np.isfinite(report['closure_post_residual']):
        print(
            f"Closure post residual: {report['closure_post_residual']:+.4f} m/s"
        )
    print(
        f"Terminal primary vel: {report['terminal_velocity_median']:+.3f} "
        f"± {report['terminal_velocity_sd']:.3f} m/s"
    )
    print(
        f"Terminal audit vel  : {report['terminal_corrected_median']:+.3f} "
        f"± {report['terminal_corrected_sd']:.3f} m/s"
    )
    print(f"Landing Velocity    : {report['landing_velocity']:.3f} m/s")
    print(
        f"Ballistic landing v : {report['ballistic_landing_velocity']:.3f} m/s "
        f"(Δ {report['landing_velocity_difference']:.3f} m/s)"
    )
    print(f"Flight Time         : {report['flight_time']:.4f} s")
    print(
        f"Symmetric Flight t  : {report['symmetric_flight_time']:.4f} s "
        f"(defisit {report['flight_time_deficit_s'] * 1000:+.1f} ms)"
    )
    if np.isfinite(report['flight_time_threshold_range_s']):
        print(
            f"Flight threshold sens: {report['flight_time_threshold_range_s'] * 1000:.1f} ms "
            "untuk ±5 N"
        )
    if np.isfinite(report['impulse_height_threshold_range_cm']):
        print(
            f"Impulse threshold sens: {report['impulse_height_threshold_range_cm']:.3f} cm "
            "untuk ±5 N"
        )
    print(f"Unloading Duration  : {report['unloading_duration']:.4f} s")
    print(f"Braking Duration    : {report['braking_duration']:.4f} s")
    print(f"Eccentric Duration  : {report['eccentric_duration']:.4f} s")
    print(f"Concentric Duration : {report['concentric_duration']:.4f} s")
    print(f"Time to Take-off    : {report['time_to_takeoff']:.4f} s")
    print("-" * 82)

    print(
        f"PRIMARY Jump Height : {report['jump_height_impulse'] * 100:.2f} cm "
        "— Impulse–Momentum"
    )
    print(
        f"Post-ZUPT audit h   : {report['jump_height_impulse_audit'] * 100:.2f} cm "
        "(bukan hasil utama)"
    )
    print(
        f"Flight-time QC h    : {report['jump_height_flight'] * 100:.2f} cm"
    )
    print(f"Method Difference   : {report['height_difference_percent']:.2f}%")
    print(
        f"Takeoff→Landing ΔCoM: {report['takeoff_landing_displacement'] * 100:+.2f} cm"
    )
    print(f"Peak Power          : {report['peak_power']:.1f} W")
    print(f"Peak Power / kg     : {report['peak_power_per_kg']:.1f} W/kg")
    print(f"RSImod              : {report['rsimod']:.3f} m/s")
    print(f"Post-landing data   : {report['post_landing_available_s']:.2f} s")
    print(f"Post-landing ZUPT   : {'AUDIT APPLIED' if report['zupt_applied'] else 'NOT APPLIED'}")
    print(f"Post stable live    : {'YES' if report['live_post_stable'] else 'NO/LEGACY'}")
    if report["post_window"] is not None:
        post = report["post_window"]
        print(
            "Post stable CV T/L/R: "
            f"{post['total_cv'] * 100:.2f}/"
            f"{post['left_cv'] * 100:.2f}/"
            f"{post['right_cv'] * 100:.2f}%"
        )
        print(
            "Post CoP range AP/ML: "
            f"{post['cop_ap_range'] * 100:.2f}/"
            f"{post['cop_ml_range'] * 100:.2f} cm"
        )
    print("=" * 82)

    for reason in report.get("core_repeat_reasons", []):
        print(f"[CORE / REPEAT REQUIRED] {reason}.")
    for reason in report.get("acquisition_note_reasons", []):
        print(f"[ACQUISITION NOTE] {reason}.")
    for reason in report["primary_caution_reasons"]:
        print(f"[PRIMARY NOTE] {reason}.")
    for reason in report["flight_review_reasons"]:
        print(f"[FLIGHT REVIEW] {reason}.")
    for reason in report["flight_caution_reasons"]:
        print(f"[FLIGHT CAUTION] {reason}.")
    for reason in report["bilateral_review_reasons"]:
        print(f"[BILATERAL REVIEW] {reason}.")
    for reason in report["bilateral_caution_reasons"]:
        print(f"[BILATERAL CAUTION] {reason}.")
    for reason in report["impact_review_reasons"]:
        print(f"[IMPACT REVIEW] {reason}.")
    for reason in report["impact_caution_reasons"]:
        print(f"[IMPACT CAUTION] {reason}.")
    for reason in report["rfd_review_reasons"]:
        print(f"[RFD REVIEW] {reason}.")
    for reason in report["rfd_caution_reasons"]:
        print(f"[RFD CAUTION] {reason}.")
    for reason in report["audit_caution_reasons"]:
        print(f"[AUDIT] {reason}.")
    for reason in report["audit_notes"]:
        print(f"[AUDIT INFO] {reason}.")

    if report["trial_quality"] == prc.TRIAL_STATUS_REPEAT:
        print(
            "[DECISION] Trial tidak dipakai sebagai data utama. Periksa alasan core/system "
            "sebelum mengulang pengukuran."
        )
    elif report["trial_quality"] == prc.TRIAL_STATUS_NOTE:
        print(
            "[DECISION] Trial tetap dapat dipakai sebagai data utama. Catatan sekunder "
            "disimpan dan tidak mewajibkan pengulangan."
        )
    else:
        print("[DECISION] Trial PASS tanpa catatan QC bermakna.")

    if report["baseline_share_difference"] > 15.0:
        print(
            "[WARNING] Distribusi beban awal kiri-kanan berbeda besar. "
            "Pastikan posisi kaki simetris dan mapping plate benar."
        )
    if np.isfinite(report["net_contrib_diff"]) and report["net_contrib_diff"] > 20.0:
        print(
            "[WARNING] Gunakan gross contribution untuk force sharing. "
            "Net contribution sangat sensitif terhadap baseline masing-masing kaki."
        )
    if report.get("temporal_qc") != "PASS":
        print("[TEMPORAL QC] " + "; ".join(report.get("temporal_reasons", [])))
    elif np.isfinite(report.get("mux_p95_abs_us", np.nan)):
        print(
            "[TEMPORAL OK] Raw MUX P95 "
            f"{report['mux_p95_abs_us']:.1f} us; alignment "
            f"{report.get('mux_alignment_coverage', np.nan) * 100.0:.2f}%."
        )

def write_summary_csv(report: dict[str, Any], data: dict[str, Any]) -> Path:
    output_csv = Path(data["filename"]).with_suffix("").with_name(
        Path(data["filename"]).stem + "_cmj_summary_v85.csv"
    )
    rows = [
        ("software_version", report["software_version"], "-"),
        ("primary_method", report["primary_method"], "-"),
        ("primary_impulse_quality", report["impulse_primary_quality"], "-"),
        ("flight_time_qc", report["flight_time_qc"], "-"),
        ("bilateral_qc", report["bilateral_qc"], "-"),
        ("impact_qc", report["impact_qc"], "-"),
        ("landing_rate_qc", report["rfd_qc"], "-"),
        ("trial_quality", report["trial_quality"], "-"),
        ("core_validity", report.get("core_validity", "-"), "-"),
        ("trial_usable", report.get("trial_usable", False), "bool"),
        ("calibration_version", report.get("calibration_version", "-"), "-"),
        ("zero_gate_status", (report.get("zero_gate_info") or {}).get("status", "NOT_AVAILABLE"), "-"),
        ("adc_nominal_rate", report["adc_nominal_sps"], "SPS"),
        ("effective_frame_rate", report["fs"], "Hz"),
        ("sampling_jitter_fraction", report["sampling_jitter_fraction"], "fraction"),
        ("analysis_missing_frames", report["sampling_diag"].get("analysis_frame_gaps", 0), "count"),
        ("analysis_gap_events", report["sampling_diag"].get("gap_events", 0), "count"),
        ("analysis_max_consecutive_missing", report["sampling_diag"].get("max_consecutive_missing", 0), "count"),
        ("analysis_missing_fraction", report["sampling_diag"].get("missing_fraction", 0.0), "fraction"),
        ("analysis_estimated_max_missing_duration", report["sampling_diag"].get("estimated_max_missing_duration_s", 0.0), "s"),
        ("gap_policy_status", report.get("gap_status", "-"), "-"),
        ("serial_lost_frames_full", report["serial_diag"].get("serial_lost_frames", 0), "count"),
        ("malformed_nonframe_record", report.get("record_malformed", 0), "count"),
        ("crc_checksum_errors", int(data.get("checksum_error_rows", 0) or 0), "count"),
        ("serial_nul_noise_lines", int((data.get("serial_transport_diag") or {}).get("noise_only_lines", 0)), "count"),
        ("serial_nul_bytes_seen", int((data.get("serial_transport_diag") or {}).get("nul_bytes_seen", 0)), "count"),
        ("mux_alignment_applied", report.get("mux_alignment_applied", False), "bool"),
        ("mux_alignment_coverage", report.get("mux_alignment_coverage", np.nan), "fraction"),
        ("mux_p95_abs", report.get("mux_p95_abs_us", np.nan), "us"),
        ("mux_p95_fraction_legacy", report.get("mux_skew_p95_fraction", np.nan), "fraction"),
        ("temporal_qc", report.get("temporal_qc", "-"), "-"),
        ("effective_temporal_resolution", report.get("mux_temporal_resolution_ms", np.nan), "ms"),
        ("impact_filter_cutoff", report["impact_filter_cutoff_hz"], "Hz"),
        ("body_mass", report["body_mass"], "kg"),
        ("mass_error", report["mass_error"], "%"),
        ("baseline_refined", report["baseline_refined"], "bool"),
        ("baseline_cv", report["baseline"]["cv"], "fraction"),
        ("baseline_slope", report["baseline"]["slope"], "N/s"),
        ("baseline_gap_to_movement", report["baseline_gap_to_movement_s"], "s"),
        ("baseline_force_uncertainty", report["baseline_force_uncertainty_n"], "N"),
        ("baseline_height_uncertainty", report["baseline_height_uncertainty_m"], "m"),
        ("primary_takeoff_velocity", report["takeoff_velocity_primary"], "m/s"),
        ("curve_takeoff_velocity", report["takeoff_velocity_curve"], "m/s"),
        ("post_zupt_audit_takeoff_velocity", report["takeoff_velocity_audit"], "m/s"),
        ("curve_vs_impulse_velocity_difference", report["impulse_velocity_difference"], "m/s"),
        ("primary_jump_height_impulse", report["jump_height_impulse"], "m"),
        ("post_zupt_audit_height", report["jump_height_impulse_audit"], "m"),
        ("flight_time_qc_height", report["jump_height_flight"], "m"),
        ("height_method_difference", report["height_difference_percent"], "%"),
        ("impulse_height_threshold_range", report["impulse_height_threshold_range_m"], "m"),
        ("flight_time", report["flight_time"], "s"),
        ("ballistic_landing_velocity", report["ballistic_landing_velocity"], "m/s"),
        ("landing_velocity_difference", report["landing_velocity_difference"], "m/s"),
        ("symmetric_flight_time", report["symmetric_flight_time"], "s"),
        ("flight_time_deficit", report["flight_time_deficit_s"], "s"),
        ("flight_time_threshold_range", report["flight_time_threshold_range_s"], "s"),
        ("live_takeoff_difference", report["live_takeoff_difference_s"], "s"),
        ("live_landing_confirmation_difference", report["live_landing_difference_s"], "s"),
        ("landing_contact_to_confirmation", report["landing_contact_to_confirmation_s"], "s"),
        ("peak_propulsive_force", report["peak_force"], "N"),
        ("peak_landing_force_impact", report["peak_landing_force"], "N"),
        ("peak_landing_force_kinetic", report["peak_landing_force_kinetic"], "N"),
        ("peak_landing_utilization", report["peak_landing_utilization_percent"], "%"),
        ("impact_peak_filter_variation", report["impact_peak_variation_percent"], "%"),
        ("impact_peak_min_force", report["impact_peak_min_force"], "N"),
        ("impact_peak_max_force", report["impact_peak_max_force"], "N"),
        ("impact_peak_half_range_uncertainty", report["impact_peak_half_range_force"], "N"),
        ("landing_gross_impulse_50ms", report["landing_gross_impulse_50ms"], "N.s"),
        ("landing_net_impulse_50ms", report["landing_net_impulse_50ms"], "N.s"),
        ("landing_gross_impulse_100ms", report["landing_gross_impulse_100ms"], "N.s"),
        ("landing_net_impulse_100ms", report["landing_net_impulse_100ms"], "N.s"),
        ("landing_net_impulse_50ms_filter_variation", report["landing_impulse_50ms_variation_percent"], "%"),
        ("landing_net_impulse_100ms_filter_variation", report["landing_impulse_100ms_variation_percent"], "%"),
        ("landing_time_to_peak", report["landing_time_to_peak"], "s"),
        ("landing_loading_rate_average", report["landing_loading_rate"], "N/s"),
        ("landing_loading_rate_20_80", report["landing_loading_rate_20_80"], "N/s"),
        ("landing_time_to_peak_samples", report["landing_time_to_peak_samples"], "samples"),
        ("landing_20_80_samples", report["landing_20_80_samples"], "samples"),
        ("landing_peak_count", report["landing_peak_count"], "count"),
        ("landing_pattern", report["landing_pattern"], "-"),
        ("bilateral_landing_order", report["bilateral_landing_order"], "-"),
        ("left_landing_time", report["left_landing_time"], "s"),
        ("right_landing_time", report["right_landing_time"], "s"),
        ("landing_right_minus_left_delay", report["landing_lr_delay_s"], "s"),
        ("landing_second_peak_ratio", report["landing_second_peak_ratio"], "fraction"),
        ("landing_first_peak_force", report["landing_first_peak_force"], "N"),
        ("landing_second_peak_force", report["landing_second_peak_force"], "N"),
        ("landing_interpeak_interval", report["landing_interpeak_interval_s"], "s"),
        ("landing_dominant_peak_sequence", report["landing_dominant_peak_sequence"], "index"),
        ("early_loading_primary_window", report["early_loading_primary_window_s"], "s"),
        ("early_loading_primary_actual_window", report["early_loading_primary_actual_window_s"], "s"),
        ("early_loading_primary_rate", report["early_loading_primary_rate_n_s"], "N/s"),
        ("early_loading_primary_r2", report["early_loading_primary_r2"], "-"),
        ("early_loading_primary_samples", report["early_loading_primary_equivalent_samples"], "samples"),
        ("early_loading_primary_net_impulse", report["early_loading_primary_net_impulse_n_s"], "N.s"),
        ("impact_early_rate_filter_variation", report["impact_rfd_variation_percent"], "%"),
        ("impact_20_80_filter_variation", report["impact_rfd_20_80_variation_percent"], "%"),
        ("total_net_impulse", report["total_net_impulse"], "N.s"),
        ("unloading_impulse", report["unloading_impulse"], "N.s"),
        ("braking_impulse", report["braking_impulse"], "N.s"),
        ("gross_left_contribution", report["left_gross_contrib"], "%"),
        ("gross_right_contribution", report["right_gross_contrib"], "%"),
        ("peak_power", report["peak_power"], "W"),
        ("rsimod", report["rsimod"], "m/s"),
        ("central_flight_offset", report["flight_grf_mean"], "N"),
        ("central_flight_sd", report["flight_grf_sd"], "N"),
        ("full_flight_offset", report["flight_grf_full_mean"], "N"),
        ("full_flight_sd", report["flight_grf_full_sd"], "N"),
        ("flight_zero_guard", report["flight_zero_guard_s"], "s"),
        ("flight_zero_equivalent_samples", report["flight_zero_equivalent_samples"], "samples"),
        ("flight_offset_percent_bw", report["flight_offset_percent_bw"], "%"),
        ("post_landing_zupt_audit", report["zupt_applied"], "bool"),
        ("raw_post_residual", report["raw_post_residual"], "m/s"),
        ("closure_equivalent_force", report["closure_equivalent_force_n"], "N"),
        ("closure_equivalent_force_percent_bw", report["closure_equivalent_force_percent_bw"], "%"),
        ("closure_duration", report["closure_duration_s"], "s"),
        ("closure_post_residual", report["closure_post_residual"], "m/s"),
        ("terminal_primary_velocity", report["terminal_velocity_median"], "m/s"),
        ("terminal_audit_velocity", report["terminal_corrected_median"], "m/s"),
        ("mux_skew_p95_fraction", report["mux_skew_p95_fraction"], "fraction"),
        ("mux_policy", report["mux_policy"], "-"),
        ("mux_alternation_fraction", report["mux_alternation_fraction"], "fraction"),
        ("mux_signed_bias_fraction", report["mux_signed_bias_fraction"], "fraction"),
    ]
    for item in report.get("impact_sensitivity", []):
        cutoff = float(item.get("cutoff_hz", np.nan))
        if not np.isfinite(cutoff):
            continue
        rows.append(
            (
                f"impact_peak_{cutoff:.0f}hz",
                item.get("peak_force", np.nan),
                "N",
            )
        )
        rows.append(
            (
                f"landing_net_impulse_50ms_{cutoff:.0f}hz",
                item.get("net_impulse_50ms", np.nan),
                "N.s",
            )
        )
        rows.append(
            (
                f"landing_net_impulse_100ms_{cutoff:.0f}hz",
                item.get("net_impulse_100ms", np.nan),
                "N.s",
            )
        )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Value", "Unit"])
        writer.writerows(rows)
        writer.writerow([])
        writer.writerow(["Core repeat reasons", *report.get("core_repeat_reasons", [])])
        writer.writerow(["Secondary note reasons", *report.get("secondary_note_reasons", [])])
        writer.writerow(["Primary review reasons", *report["primary_review_reasons"]])
        writer.writerow(["Flight-time QC reasons", *report["flight_review_reasons"], *report["flight_caution_reasons"]])
        writer.writerow(["Bilateral QC reasons", *report["bilateral_review_reasons"], *report["bilateral_caution_reasons"]])
        writer.writerow(["Impact reasons", *report["impact_review_reasons"], *report["impact_caution_reasons"]])
        writer.writerow(["Landing-rate reasons", *report["rfd_review_reasons"], *report["rfd_caution_reasons"]])
        writer.writerow(["Audit notes", *report["audit_notes"]])
    print(f"[OK] Ringkasan tersimpan: {output_csv}")
    return output_csv


def plot_report(
    report: dict[str, Any],
    data: dict[str, Any],
) -> tuple[dict[str, plt.Figure], dict[str, Path]]:
    """Bangun dua figure dashboard CMJ tanpa menghitung ulang metrik kinetik."""
    time_arr = np.asarray(report["time"], dtype=float)

    overview_figure, overview_axes = plt.subplots(
        2, 1, figsize=(12.4, 6.4), dpi=100, sharex=True,
        gridspec_kw={"height_ratios": [1.12, 1.0]},
    )
    overview_figure.patch.set_facecolor("white")
    ax_force, ax_velocity = overview_axes

    ax_force.plot(time_arr, report["grf"], label="GRF total")
    ax_force.plot(time_arr, report["left_force"], label="Kaki kiri", alpha=0.75)
    ax_force.plot(time_arr, report["right_force"], label="Kaki kanan", alpha=0.75)
    ax_force.axhline(report["body_weight"], linestyle="--", label="Body weight")
    force_handles, force_labels = ax_force.get_legend_handles_labels()
    event_lines = (
        ("MOV", report["movement_start_time"]),
        ("UW", report["unweighting_time"]),
        ("PROP", report["propulsive_time"]),
        ("TO", report["takeoff_time"]),
        ("LAND", report["landing_time"]),
    )
    event_handles: list[Any] = []
    event_labels: list[str] = []
    for label, event_time in event_lines:
        if np.isfinite(event_time):
            event_handles.append(
                ax_force.axvline(
                    float(event_time), linestyle=":", linewidth=1.1, alpha=0.90
                )
            )
            event_labels.append(label)
    if np.isfinite(report["takeoff_time"]) and np.isfinite(report["landing_time"]):
        ax_force.axvspan(report["takeoff_time"], report["landing_time"], alpha=0.12)
    ax_force.set_title("GRF dan Fase Utama CMJ")
    ax_force.set_ylabel("Gaya (N)")
    ax_force.grid(True)
    force_legend = ax_force.legend(
        force_handles, force_labels, loc="upper left", fontsize=7.5,
        ncol=4, framealpha=0.92, handlelength=2.0, columnspacing=1.0,
    )
    ax_force.add_artist(force_legend)
    if event_handles:
        ax_force.legend(
            event_handles, event_labels, loc="upper right", fontsize=7.0,
            ncol=len(event_handles), framealpha=0.92, title="Fase",
            title_fontsize=7.0, handlelength=1.2, columnspacing=0.8,
            borderpad=0.35, labelspacing=0.25,
        )

    ax_velocity.plot(time_arr, report["velocity_primary"], label="Velocity CoM")
    ax_velocity.axhline(0.0, linestyle="--", linewidth=0.8)
    ax_velocity.axvline(report["takeoff_time"], linestyle=":", label="Take-off")
    ax_velocity.axvline(report["landing_time"], linestyle=":", label="Landing")
    ax_velocity.set_title("Kecepatan Pusat Massa dan Daya")
    ax_velocity.set_xlabel("Waktu (s)")
    ax_velocity.set_ylabel("Velocity (m/s)")
    ax_velocity.grid(True)
    ax_power = ax_velocity.twinx()
    ax_power.plot(time_arr, report["power"], label="Power", alpha=0.55)
    ax_power.set_ylabel("Power (W)")
    lines_1, labels_1 = ax_velocity.get_legend_handles_labels()
    lines_2, labels_2 = ax_power.get_legend_handles_labels()
    ax_velocity.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8, ncol=2, framealpha=0.92)

    for axis in (ax_force, ax_velocity, ax_power):
        axis.tick_params(axis="both", labelsize=8, pad=3)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
    for axis in (ax_force, ax_velocity):
        axis.set_title(axis.get_title(), fontsize=11, pad=9)
    overview_figure.subplots_adjust(
        left=0.07, right=0.92, bottom=0.11, top=0.91, hspace=0.48
    )

    landing_figure, landing_axes = plt.subplots(
        1, 2, figsize=(12.4, 6.4), dpi=100,
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )
    landing_figure.patch.set_facecolor("white")
    landing_figure.suptitle("Landing dan Kontribusi Bilateral", fontsize=12, fontweight="bold")
    ax_landing, ax_bilateral = landing_axes

    landing_start = float(report["landing_time"])
    landing_end = min(float(time_arr[-1]), landing_start + 0.300)
    landing_mask = (time_arr >= landing_start) & (time_arr <= landing_end)
    if np.count_nonzero(landing_mask) >= 2:
        landing_time_ms = (time_arr[landing_mask] - landing_start) * 1000.0
        ax_landing.plot(landing_time_ms, np.asarray(report["impact_grf"])[landing_mask], label="GRF impact")
        ax_landing.plot(landing_time_ms, np.asarray(report["left_force"])[landing_mask], label="Kiri", alpha=0.70)
        ax_landing.plot(landing_time_ms, np.asarray(report["right_force"])[landing_mask], label="Kanan", alpha=0.70)
        peak_time_ms = (float(report["peak_landing_time"]) - landing_start) * 1000.0
        ax_landing.axvline(peak_time_ms, linestyle=":", label=f"Peak {peak_time_ms:.0f} ms")
    else:
        ax_landing.text(0.5, 0.5, "Window landing tidak mencukupi", ha="center", va="center", transform=ax_landing.transAxes)
    ax_landing.set_title("Landing 0–300 ms")
    ax_landing.set_xlabel("Waktu setelah kontak (ms)")
    ax_landing.set_ylabel("Gaya (N)")
    ax_landing.grid(True)
    ax_landing.legend(loc="upper right", fontsize=8, framealpha=0.92)

    categories = ["Net impulse", "Gross impulse", "Baseline load"]
    left_values = [report["left_net_contrib"], report["left_gross_contrib"], report["baseline_left_share"]]
    right_values = [report["right_net_contrib"], report["right_gross_contrib"], report["baseline_right_share"]]
    positions = np.arange(len(categories), dtype=float)
    width = 0.36
    left_bars = ax_bilateral.bar(positions - width / 2.0, left_values, width, label="Kiri")
    right_bars = ax_bilateral.bar(positions + width / 2.0, right_values, width, label="Kanan")
    ax_bilateral.axhline(50.0, linestyle="--", linewidth=0.8)
    ax_bilateral.set_xticks(positions, categories)
    ax_bilateral.set_ylim(0.0, max(100.0, float(np.nanmax(left_values + right_values)) * 1.20))
    ax_bilateral.set_title("Kontribusi Kiri–Kanan")
    ax_bilateral.set_ylabel("Kontribusi (%)")
    ax_bilateral.grid(True, axis="y")
    ax_bilateral.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=8, framealpha=0.92)
    ax_bilateral.bar_label(left_bars, fmt="%.1f", padding=3, fontsize=8)
    ax_bilateral.bar_label(right_bars, fmt="%.1f", padding=3, fontsize=8)
    ax_bilateral.set_xlim(-0.62, len(categories) - 0.38)

    for axis in landing_axes:
        axis.set_title(axis.get_title(), fontsize=11, pad=10)
        axis.tick_params(axis="both", labelsize=8, pad=3)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
    landing_figure.subplots_adjust(
        left=0.07, right=0.95, bottom=0.13, top=0.84, wspace=0.30
    )

    overview_png = prc.derived_output_path(data["filename"], "_cmj_overview_v85", ".png")
    landing_png = prc.derived_output_path(data["filename"], "_cmj_landing_bilateral_v85", ".png")
    overview_figure.savefig(overview_png, dpi=200, bbox_inches="tight")
    landing_figure.savefig(landing_png, dpi=200, bbox_inches="tight")
    print(f"[OK] Grafik overview tersimpan: {overview_png}")
    print(f"[OK] Grafik landing/bilateral tersimpan: {landing_png}")
    return {"overview": overview_figure, "landing": landing_figure}, {"overview": overview_png, "landing": landing_png}

def _cmj_detail_text(report: dict[str, Any], data: dict[str, Any]) -> str:
    """Ringkasan rinci untuk tab Detail."""
    impact_cutoff_text = " | ".join(
        f"{float(item.get('cutoff_hz', np.nan)):.0f} Hz="
        f"{prc.format_metric(item.get('peak_force', np.nan), 'N', 1)}"
        for item in report.get("impact_sensitivity", [])
        if np.isfinite(float(item.get("cutoff_hz", np.nan)))
    ) or "-"
    values = [
        ("Primary method", report["primary_method"]),
        ("Primary impulse quality", report["impulse_primary_quality"]),
        ("Flight-time QC", report["flight_time_qc"]),
        ("Bilateral QC", report["bilateral_qc"]),
        ("Impact QC", report["impact_qc"]),
        ("Landing-rate QC", report["rfd_qc"]),
        ("Body mass", prc.format_metric(report["body_mass"], "kg", 3)),
        ("Jump height impulse", prc.format_metric(report["jump_height_impulse"] * 100.0, "cm", 2)),
        ("Jump height flight-time", prc.format_metric(report["jump_height_flight"] * 100.0, "cm", 2)),
        ("Height difference", prc.format_metric(report["height_difference_percent"], "%", 2)),
        ("Take-off velocity", prc.format_metric(report["takeoff_velocity_primary"], "m/s", 3)),
        ("Peak propulsive force", prc.format_metric(report["peak_force"], "N", 1)),
        ("Peak power", prc.format_metric(report["peak_power"], "W", 1)),
        ("Peak power per kg", prc.format_metric(report["peak_power_per_kg"], "W/kg", 2)),
        ("Time to take-off", prc.format_metric(report["time_to_takeoff"], "s", 3)),
        ("RSImod", prc.format_metric(report["rsimod"], "m/s", 3)),
        ("Peak landing primary (60 Hz)", prc.format_metric(report["peak_landing_force"], "N", 1)),
        ("Peak landing 40/50/60 Hz", impact_cutoff_text),
        ("Peak landing filter variation", prc.format_metric(report["impact_peak_variation_percent"], "%", 2)),
        ("Peak landing sensitivity range", (
            f"{prc.format_metric(report['impact_peak_min_force'], 'N', 1)} – "
            f"{prc.format_metric(report['impact_peak_max_force'], 'N', 1)}"
        )),
        ("Peak landing half-range uncertainty", prc.format_metric(report["impact_peak_half_range_force"], "N", 1)),
        ("Landing impulse 0–50 ms gross", prc.format_metric(report["landing_gross_impulse_50ms"], "N·s", 3)),
        ("Landing impulse 0–50 ms net", prc.format_metric(report["landing_net_impulse_50ms"], "N·s", 3)),
        ("Landing impulse 0–100 ms gross", prc.format_metric(report["landing_gross_impulse_100ms"], "N·s", 3)),
        ("Landing impulse 0–100 ms net", prc.format_metric(report["landing_net_impulse_100ms"], "N·s", 3)),
        ("Impulse filter variation 50/100 ms", (
            f"{prc.format_metric(report['landing_impulse_50ms_variation_percent'], '%', 2)} / "
            f"{prc.format_metric(report['landing_impulse_100ms_variation_percent'], '%', 2)}"
        )),
        ("Landing utilization", prc.format_metric(report["peak_landing_utilization_percent"], "%", 1)),
        ("Landing loading rate 20–80%", prc.format_metric(report["landing_loading_rate_20_80"], "N/s", 1)),
        ("Early loading 50 ms", prc.format_metric(report["early_loading_primary_rate_n_s"], "N/s", 1)),
        ("Left/right net contribution", (
            f"{prc.format_metric(report['left_net_contrib'], '%', 2)} / "
            f"{prc.format_metric(report['right_net_contrib'], '%', 2)}"
        )),
        ("Left/right gross contribution", (
            f"{prc.format_metric(report['left_gross_contrib'], '%', 2)} / "
            f"{prc.format_metric(report['right_gross_contrib'], '%', 2)}"
        )),
        ("Raw data", str(data["filename"])),
    ]
    return "\n".join(f"{name:<34}: {value}" for name, value in values)


def _cmj_status_message(report: dict[str, Any]) -> str:
    """Interpretasi keputusan akhir v8.5: core, secondary QC, dan audit info dipisahkan."""
    status = str(report.get("trial_quality", "-"))
    if status == prc.TRIAL_STATUS_PASS:
        return (
            "Trial PASS. Hasil utama impulse–momentum layak digunakan dan tidak ada "
            "catatan QC sekunder bermakna."
        )
    if status == prc.TRIAL_STATUS_NOTE:
        reasons = report.get("secondary_note_reasons", [])
        return (
            "Trial dapat digunakan tanpa wajib diulang. Ada catatan QC sekunder: "
            f"{prc.compact_reason_text(reasons, limit=3)}."
        )
    reasons = report.get("core_repeat_reasons", [])
    return (
        "Trial tidak digunakan sebagai data utama. Periksa penyebab core/system sebelum "
        f"mengulang: {prc.compact_reason_text(reasons, limit=3)}."
    )


def _configure_cmj_tree_tags(tree: object) -> None:
    for level, status in (
        ("success", "PASS"),
        ("warning", "CAUTION"),
        ("danger", "REVIEW"),
        ("neutral", "UNKNOWN"),
    ):
        palette = prc.status_palette(status)
        tree.tag_configure(level, background=palette["background"], foreground=palette["foreground"])


def show_cmj_result_window(
    report: dict[str, Any],
    data: dict[str, Any],
    figures: dict[str, plt.Figure],
    summary_csv: Path,
    output_paths: dict[str, Path],
) -> bool:
    """Tampilkan dashboard CMJ responsif. Return True bila pengguna mengulang."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from tkinter.scrolledtext import ScrolledText
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    except ImportError as exc:
        print(f"[WARNING] GUI Tkinter tidak tersedia: {exc}")
        plt.show()
        return False

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"[WARNING] Window GUI tidak dapat dibuka: {exc}")
        plt.show()
        return False

    repeat_state = {"requested": False}
    root.title("Force Plate — CMJ Result")
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

    status = str(report["trial_quality"])
    display_status = prc.humanize_status(status)
    badge_status = (
        "WITH NOTE" if status == prc.TRIAL_STATUS_NOTE
        else "REPEAT REQUIRED" if status == prc.TRIAL_STATUS_REPEAT
        else "PASS"
    )
    palette = prc.status_palette(status)
    header = tk.Frame(main, bg="#111827", padx=16, pady=11)
    header.grid(row=0, column=0, sticky="ew")
    tk.Label(header, text="COUNTERMOVEMENT JUMP", bg="#111827", fg="#FFFFFF", font=("Segoe UI", 18, "bold")).pack(side="left")
    tk.Label(header, text=badge_status, bg=palette["background"], fg=palette["foreground"], padx=12, pady=6, font=("Segoe UI", 11, "bold")).pack(side="right")

    metadata_text = (
        f"Massa {report['body_mass']:.2f} kg  |  "
        f"Sampling efektif {report['fs']:.2f} Hz  |  "
        f"ADC nominal {report['adc_nominal_sps']} SPS  |  "
        f"Data {Path(data['filename']).name}"
    )
    ttk.Label(
        main, text=metadata_text, style="HeaderMeta.TLabel", padding=(16, 6)
    ).grid(row=1, column=0, sticky="ew")

    banner = tk.Frame(main, bg=palette["background"], highlightbackground=palette["border"], highlightthickness=1)
    banner.grid(row=2, column=0, sticky="ew", pady=(8, 2))
    banner_label = tk.Label(
        banner, text=_cmj_status_message(report), bg=palette["background"], fg=palette["foreground"],
        anchor="w", justify="left", padx=12, pady=7, font=("Segoe UI", 9, "bold"),
    )
    banner_label.pack(fill="x")
    banner.bind(
        "<Configure>",
        lambda event: banner_label.configure(
            wraplength=max(280, int(event.width) - 28)
        ),
        add="+",
    )

    cards = [
        ("Jump Height", prc.format_metric(report["jump_height_impulse"] * 100.0, "cm", 2)),
        ("Take-off Velocity", prc.format_metric(report["takeoff_velocity_primary"], "m/s", 3)),
        ("Peak Propulsive Force", f"{prc.format_metric(report['peak_force'], 'N', 0)} | {prc.format_metric(report['peak_force'] / report['body_weight'], 'BW', 2)}"),
        ("Peak Power", prc.format_metric(report["peak_power_per_kg"], "W/kg", 2)),
        ("Time to Take-off", prc.format_metric(report["time_to_takeoff"], "s", 3)),
        ("RSImod", prc.format_metric(report["rsimod"], "m/s", 3)),
        ("Peak Landing (60 Hz)", f"{prc.format_metric(report['peak_landing_force'], 'N', 0)} | {prc.format_metric(report['peak_landing_force'] / report['body_weight'], 'BW', 2)}"),
        ("Net Contribution", f"Kiri {report['left_net_contrib']:.1f}% | Kanan {report['right_net_contrib']:.1f}%"),
    ]
    cards_frame = ttk.Frame(main, style="Dashboard.TFrame")
    cards_frame.grid(row=3, column=0, sticky="ew", pady=(7, 7))
    card_widgets: list[ttk.Frame] = []
    for title, value in cards:
        card = ttk.Frame(cards_frame, style="Card.TFrame", padding=9)
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text=value, style="CardValue.TLabel", justify="left", wraplength=230).pack(anchor="w", pady=(3, 0))
        card_widgets.append(card)

    layout_state = {"columns": 0}
    def layout_cards(event: object | None = None) -> None:
        width = max(cards_frame.winfo_width(), root.winfo_width())
        columns = 8 if width >= 1680 else 4 if width >= 980 else 2
        if layout_state["columns"] == columns:
            return
        layout_state["columns"] = columns
        for index in range(8):
            cards_frame.columnconfigure(index, weight=1 if index < columns else 0, uniform="card" if index < columns else "")
        for index, card in enumerate(card_widgets):
            card.grid_forget()
            card.grid(row=index // columns, column=index % columns, sticky="nsew", padx=3, pady=3)
    cards_frame.bind("<Configure>", layout_cards)
    root.after_idle(layout_cards)

    notebook = ttk.Notebook(main)
    notebook.grid(row=4, column=0, sticky="nsew")
    overview_tab = ttk.Frame(notebook)
    landing_tab = ttk.Frame(notebook)
    qc_tab = ttk.Frame(notebook, padding=10)
    detail_tab = ttk.Frame(notebook, padding=10)
    system_tab = ttk.Frame(notebook, padding=10)
    note_tab = ttk.Frame(notebook, padding=10)
    notebook.add(overview_tab, text="Performa Utama")
    notebook.add(landing_tab, text="Landing & Bilateral")
    notebook.add(qc_tab, text="Quality Control")
    notebook.add(detail_tab, text="Detail")
    notebook.add(system_tab, text="System Health")
    notebook.add(note_tab, text="Catatan Operator")

    canvas_refs: list[FigureCanvasTkAgg] = []
    resize_callbacks: list[callable] = []

    def attach_figure(tab: ttk.Frame, figure: plt.Figure, note: str) -> None:
        """Embed figure responsif, termasuk figure pada tab tersembunyi."""
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
        "Metode utama: impulse–momentum. Penanda fase: MOV=movement, UW=unweighting, PROP=propulsive, TO=take-off, LAND=landing.",
    )
    attach_figure(
        landing_tab, figures["landing"],
        "Landing zoom menampilkan 300 ms pertama setelah kontak. Peak utama hanya memakai low-pass 60 Hz; 40/50 Hz dihitung sebagai audit sensitivitas internal dan tidak menjadi hasil utama. Landing impulse 0–50 dan 0–100 ms dilaporkan sebagai metrik yang lebih robust.",
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

    qc_tree = ttk.Treeview(
        qc_table_frame, columns=("qc", "role", "status", "reason"),
        show="headings", height=9
    )
    qc_scroll_y = ttk.Scrollbar(qc_table_frame, orient="vertical", command=qc_tree.yview)
    qc_scroll_x = ttk.Scrollbar(qc_table_frame, orient="horizontal", command=qc_tree.xview)
    qc_tree.configure(yscrollcommand=qc_scroll_y.set, xscrollcommand=qc_scroll_x.set)
    for column, title, width in (
        ("qc", "Pemeriksaan", 220),
        ("role", "Peran", 150),
        ("status", "Status", 170),
        ("reason", "Alasan utama", 820),
    ):
        qc_tree.heading(column, text=title)
        qc_tree.column(column, width=width, minwidth=130, anchor="w", stretch=(column == "reason"))
    qc_tree.grid(row=0, column=0, sticky="nsew")
    qc_scroll_y.grid(row=0, column=1, sticky="ns")
    qc_scroll_x.grid(row=1, column=0, sticky="ew")
    qc_table_frame.rowconfigure(0, weight=1)
    qc_table_frame.columnconfigure(0, weight=1)
    _configure_cmj_tree_tags(qc_tree)

    acquisition_qc_status = (
        "REVIEW" if report.get("acquisition_reasons")
        else "CAUTION" if report.get("acquisition_note_reasons")
        else "PASS"
    )
    qc_items = [
        ("Acquisition / frame integrity", "CORE bila hard gap/ADC", acquisition_qc_status,
         report.get("acquisition_reasons", []) + report.get("acquisition_note_reasons", [])),
        ("Primary impulse", "CORE", report["impulse_primary_quality"],
         report["primary_review_reasons"] + report["primary_caution_reasons"]),
        ("Temporal / MUX", "CORE bila REVIEW", report.get("temporal_qc", "-"),
         report.get("temporal_reasons", [])),
        ("Flight-time", "SECONDARY", report["flight_time_qc"],
         report["flight_review_reasons"] + report["flight_caution_reasons"]),
        ("Bilateral biomechanics", "SECONDARY", report["bilateral_qc"],
         report["bilateral_review_reasons"] + report["bilateral_caution_reasons"]),
        ("Impact landing", "SECONDARY; >80% desain = CORE", report["impact_qc"],
         report["impact_review_reasons"] + report["impact_caution_reasons"]),
        ("Landing rate / RFD", "SECONDARY", report["rfd_qc"],
         report["rfd_review_reasons"] + report["rfd_caution_reasons"]),
    ]
    qc_details_by_iid: dict[str, str] = {}
    for name, role, qc_status, reasons in qc_items:
        reason_text = "; ".join(str(reason) for reason in reasons) if reasons else "Tidak ada masalah terdeteksi."
        iid = qc_tree.insert(
            "", "end",
            values=(name, role, prc.humanize_status(qc_status), reason_text),
            tags=(prc.status_level(qc_status),),
        )
        qc_details_by_iid[iid] = (
            f"{name}\nPeran: {role}\nStatus: {prc.humanize_status(qc_status)}"
            f"\n\n{reason_text}"
        )

    ttk.Label(qc_detail_frame, text="Detail pemeriksaan terpilih", font=("Segoe UI", 10, "bold")).pack(anchor="w")
    qc_detail_text = ScrolledText(qc_detail_frame, wrap="word", height=6, font=("Segoe UI", 9))
    qc_detail_text.pack(fill="both", expand=True, pady=(4, 0))
    qc_detail_text.configure(state="disabled")
    def update_qc_detail(event: object | None = None) -> None:
        selection = qc_tree.selection()
        if not selection:
            return
        qc_detail_text.configure(state="normal")
        qc_detail_text.delete("1.0", "end")
        qc_detail_text.insert("1.0", qc_details_by_iid.get(selection[0], "-"))
        qc_detail_text.configure(state="disabled")
    qc_tree.bind("<<TreeviewSelect>>", update_qc_detail)
    first_items = qc_tree.get_children()
    if first_items:
        qc_tree.selection_set(first_items[0])
        root.after_idle(update_qc_detail)

    detail_text = ScrolledText(detail_tab, wrap="word", font=("Consolas", 10))
    detail_text.insert("1.0", _cmj_detail_text(report, data))
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
        ("Zero gate", (report.get("zero_gate_info") or {}).get("status", "NOT_AVAILABLE")),
        ("Calibration version", report.get("calibration_version", "-")),
        ("ADC nominal rate", f"{report['adc_nominal_sps']} SPS"),
        ("Effective frame rate", f"{report['fs']:.3f} Hz"),
        ("Timestamp jitter", prc.format_metric(report["sampling_diag"]["jitter_sd"] * 1000.0, "ms", 4)),
        ("Serial lost frames", report["serial_diag"]["serial_lost_frames"]),
        ("ADC rejected record", report["record_adc_rejected"]),
        ("Malformed/non-frame record", report["record_malformed"]),
        ("CRC checksum errors", int(data.get("checksum_error_rows", 0) or 0)),
        ("Analysis missing frames", report["sampling_diag"]["analysis_frame_gaps"]),
        ("Gap events", report["sampling_diag"].get("gap_events", 0)),
        ("Max consecutive missing", report["sampling_diag"].get("max_consecutive_missing", 0)),
        ("Missing fraction", prc.format_metric(report["sampling_diag"].get("missing_fraction", 0.0) * 100.0, "%", 4)),
        ("Estimated max missing duration", prc.format_metric(report["sampling_diag"].get("estimated_max_missing_duration_s", 0.0) * 1000.0, "ms", 3)),
        ("Gap policy status", report.get("gap_status", "-")),
        ("Serial transport decode errors", int((data.get("serial_transport_diag") or {}).get("decode_errors", 0))),
        ("Serial transport resync events", int((data.get("serial_transport_diag") or {}).get("resync_events", 0))),
        ("Serial NUL noise lines", int((data.get("serial_transport_diag") or {}).get("noise_only_lines", 0))),
        ("Serial NUL bytes seen", int((data.get("serial_transport_diag") or {}).get("nul_bytes_seen", 0))),
        ("Recovered prefixed lines", int((data.get("serial_transport_diag") or {}).get("recovered_prefixed_lines", 0))),
        ("Serial max pending lines", int((data.get("serial_transport_diag") or {}).get("max_pending_lines", 0))),
        ("Protocol error counts", str(data.get("protocol_error_counts", {}))),
        ("Malformed examples", " | ".join(data.get("malformed_examples", [])) if data.get("malformed_examples") else "-"),
        ("MUX mean abs (raw)", prc.format_metric(report["mux_mean_abs_us"], "µs", 1)),
        ("MUX P95 abs (raw)", prc.format_metric(report["mux_p95_abs_us"], "µs", 1)),
        ("MUX P95 / frame (diagnostic only)", prc.format_metric(report["mux_skew_p95_fraction"] * 100.0, "%", 1)),
        ("MUX alternation", prc.format_metric(report["mux_alternation_fraction"] * 100.0, "%", 1)),
        ("MUX alignment", "APPLIED" if report.get("mux_alignment_applied", False) else "NOT_APPLIED"),
        ("MUX alignment coverage", prc.format_metric(report.get("mux_alignment_coverage", np.nan) * 100.0, "%", 2)),
        ("Temporal QC", report.get("temporal_qc", "-")),
        ("Live/offline take-off diff", prc.format_metric(prc.finite_number(report.get("live_takeoff_difference_s")) * 1000.0, "ms", 2)),
        ("Live/offline landing diff", prc.format_metric(prc.finite_number(report.get("live_landing_difference_s")) * 1000.0, "ms", 2)),
        ("Audit info", "; ".join(str(item) for item in report.get("audit_notes", [])) or "Tidak ada"),
        ("Final decision policy", "CORE/hard gap -> REPEAT REQUIRED; small isolated gap/secondary -> USABLE WITH NOTE; noise-only/audit -> tidak menurunkan status"),
        ("Impact primary filter", prc.format_metric(report["impact_filter_cutoff_hz"], "Hz", 1)),
        ("Impact peak sensitivity", (
            " | ".join(
                f"{float(item.get('cutoff_hz', np.nan)):.0f}Hz="
                f"{prc.format_metric(item.get('peak_force', np.nan), 'N', 0)}"
                for item in report.get("impact_sensitivity", [])
                if np.isfinite(float(item.get("cutoff_hz", np.nan)))
            ) or "-"
        )),
        ("Impact peak variation", prc.format_metric(report.get("impact_peak_variation_percent", np.nan), "%", 2)),
        ("Impact peak range", (
            f"{prc.format_metric(report.get('impact_peak_min_force', np.nan), 'N', 0)} – "
            f"{prc.format_metric(report.get('impact_peak_max_force', np.nan), 'N', 0)}"
        )),
        ("Landing impulse 0-50 ms net", prc.format_metric(report.get("landing_net_impulse_50ms", np.nan), "N·s", 3)),
        ("Landing impulse 0-100 ms net", prc.format_metric(report.get("landing_net_impulse_100ms", np.nan), "N·s", 3)),
        ("Software version", report["software_version"]),
        ("Serial baud", data.get("serial_baud", "-")),
    ]
    for metric, value in system_rows:
        system_tree.insert("", "end", values=(metric, value))

    ttk.Label(note_tab, text="Catatan disimpan sebagai file teks pendamping dan tidak mengubah data mentah.").pack(anchor="w", pady=(0, 8))
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
            title="Ekspor grafik CMJ",
            defaultextension=".pdf",
            filetypes=[("PDF multi-halaman", "*.pdf"), ("PNG", "*.png")],
            initialfile=Path(data["filename"]).with_suffix("").name + "_cmj_dashboard.pdf",
        )
        if not selected:
            return
        target = Path(selected)
        try:
            saved: list[Path] = []
            if target.suffix.lower() == ".pdf":
                with PdfPages(target) as pdf:
                    pdf.savefig(figures["overview"], bbox_inches="tight")
                    pdf.savefig(figures["landing"], bbox_inches="tight")
                saved.append(target)
            else:
                overview_target = target.with_suffix(".png")
                landing_target = overview_target.with_name(overview_target.stem + "_landing_bilateral.png")
                figures["overview"].savefig(overview_target, dpi=200, bbox_inches="tight")
                figures["landing"].savefig(landing_target, dpi=200, bbox_inches="tight")
                saved.extend([overview_target, landing_target])
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
            note_path = prc.write_operator_note(data["filename"], "CMJ", status, note)
        except OSError as exc:
            messagebox.showerror("Gagal menyimpan catatan", str(exc))
            return
        messagebox.showinfo("Catatan tersimpan", str(note_path))

    def request_repeat() -> None:
        repeat_state["requested"] = True
        root.destroy()

    ttk.Button(left_actions, text="Ekspor Grafik", style="Action.TButton", command=export_figures).pack(side="left", padx=3)
    ttk.Button(left_actions, text="Buka Ringkasan", style="Action.TButton", command=lambda: show_open_result(summary_csv)).pack(side="left", padx=3)
    ttk.Button(left_actions, text="Buka Data Mentah", style="Action.TButton", command=lambda: show_open_result(Path(data["filename"]))).pack(side="left", padx=3)
    ttk.Button(left_actions, text="Buka Folder Hasil", style="Action.TButton", command=lambda: show_open_result(Path(data["filename"]).parent)).pack(side="left", padx=3)
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
    print("\n--- MODUL COUNTERMOVEMENT JUMP v8.5 + CRC / GAP-ROBUST DASHBOARD ---")
    print(
        "[INFO] Hasil utama: impulse–momentum pra-take-off. "
        "Flight time, bilateral, landing impact/RFD, dan ZUPT adalah QC sekunder; "
        "QC sekunder tidak otomatis mewajibkan pengulangan."
    )
    print(
        "[INFO] Faktor kalibrasi hasil refinement 12 kondisi statis: "
        "L1-L4=17553.7 dan R1-R4=17591.5 counts/kg."
    )
    try:
        expected_mass = ask_expected_mass()
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    while True:
        data = prc.acquire_data(
            filename_prefix="FP_CMJ",
            calculate_cop_flag=False,
            mode="cmj",
            acquisition_profile="human",
        )
        if data is None:
            return 1

        try:
            report = analyze_cmj(data, expected_mass)
        except (ValueError, KeyError, IndexError, FloatingPointError) as exc:
            print(f"[INVALID TRIAL] {exc}")
            return 1

        print_report(report, data, expected_mass)
        summary_csv = write_summary_csv(report, data)
        figures, output_paths = plot_report(report, data)
        repeat_requested = show_cmj_result_window(
            report, data, figures, summary_csv, output_paths
        )
        for figure in figures.values():
            plt.close(figure)
        if not repeat_requested:
            return 0
        print("\n[REPEAT] Memulai pengukuran CMJ baru dengan massa referensi yang sama.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[SELESAI] Program dihentikan oleh pengguna.")
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"[ERROR INTERNAL] Modul CMJ dihentikan dengan aman: "
            f"{type(exc).__name__}: {exc}"
        )
        raise SystemExit(1)




