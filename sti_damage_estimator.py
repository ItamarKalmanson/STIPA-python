import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, resample, sosfiltfilt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# --- Utility and Prerequisite Functions ---

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_envelope(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def extract_envelope(signal, fs, cutoff_hz=100):
    """Extracts the envelope of a signal using a low-pass filter."""
    sos_lp = butter(2, cutoff_hz / (0.5 * fs), 'low', output='sos')
    return sosfiltfilt(sos_lp, np.abs(signal))


def align_signals(ref_sig, deg_sig, fs):
    """Aligns two signals using cross-correlation and returns the aligned parts."""
    print("Aligning signals using cross-correlation...")
    if ref_sig.ndim > 1: ref_sig = ref_sig.mean(axis=1)
    if deg_sig.ndim > 1: deg_sig = deg_sig.mean(axis=1)

    resample_rate = 8000
    if fs > resample_rate and len(ref_sig) > fs * 5:
        ref_resampled = resample(ref_sig, int(len(ref_sig) * resample_rate / fs))
        deg_resampled = resample(deg_sig, int(len(deg_sig) * resample_rate / fs))
        corr = np.correlate(deg_resampled, ref_resampled, mode='full')
        lag_resampled = np.argmax(corr) - (len(ref_resampled) - 1)
        lag = int(lag_resampled * fs / resample_rate)
    else:
        corr = np.correlate(deg_sig, ref_sig, mode='full')
        lag = np.argmax(corr) - (len(ref_sig) - 1)

    if lag > 0:
        deg_aligned = deg_sig[lag:]
        ref_aligned = ref_sig
    else:
        ref_aligned = ref_sig[-lag:]
        deg_aligned = deg_sig

    min_len = min(len(ref_aligned), len(deg_aligned))
    ref_aligned = ref_aligned[:min_len]
    deg_aligned = deg_aligned[:min_len]

    print(f"Signals aligned. Delay of {lag / fs:.3f}s corrected.")
    return ref_aligned, deg_aligned


# --- Core Analysis Functions ---

def estimate_snr_per_band(ref_aligned, deg_aligned, fs, octave_bands):
    """
    Estimates the Signal-to-Noise Ratio (SNR) in each octave band using a fixed time window for noise.
    Returns the SNR for each band and the boolean mask of silent regions.
    """
    print("Estimating SNR per octave band using fixed noise window (0.5s - 2.5s)...")
    snr_bands = []

    start_sample = int(0.5 * fs)
    end_sample = int(2.5 * fs)

    if end_sample > len(ref_aligned):
        print(f"Warning: The fixed noise window exceeds the signal length. Adjusting window.")
        end_sample = len(ref_aligned)
        if start_sample >= end_sample:
            print("Error: Cannot define a noise window. Signal is too short.")
            return [np.nan] * len(octave_bands), np.zeros_like(ref_aligned, dtype=bool)

    is_silent = np.zeros(len(ref_aligned), dtype=bool)
    is_silent[start_sample:end_sample] = True
    is_signal = ~is_silent

    if not np.any(is_silent):
        print("Warning: Could not define a noise region.")
        return [np.nan] * len(octave_bands), np.zeros_like(ref_aligned, dtype=bool)

    for band in octave_bands:
        lowcut, highcut = band / np.sqrt(2), band * np.sqrt(2)
        sos = butter_bandpass(lowcut, highcut, fs)
        deg_filtered = sosfiltfilt(sos, deg_aligned)

        if np.any(np.isnan(deg_filtered)):
            print(f"Warning: Filter for {band} Hz band produced NaN values. Assigning a floor SNR.")
            snr_bands.append(-15.0)
            continue

        signal_power = np.mean(deg_filtered[is_signal] ** 2)
        noise_power = np.mean(deg_filtered[is_silent] ** 2)

        if noise_power < 1e-20:
            snr_db = 99.0
        else:
            true_signal_power = signal_power - noise_power
            snr_db = 10 * np.log10(true_signal_power / noise_power) if true_signal_power > 0 else 0.0
        snr_bands.append(snr_db)

    return snr_bands, is_silent


def calculate_mtf_matrix(ref_aligned, deg_aligned, fs, octave_bands, mod_freqs):
    """Calculates the full 7x14 Modulation Transfer Function (MTF) matrix."""
    print("Calculating full MTF matrix...")
    mtf_matrix = np.zeros((len(octave_bands), len(mod_freqs)))

    for i, band in enumerate(octave_bands):
        lowcut, highcut = band / np.sqrt(2), band * np.sqrt(2)
        sos = butter_bandpass(lowcut, highcut, fs)
        ref_filtered = sosfiltfilt(sos, ref_aligned)
        deg_filtered = sosfiltfilt(sos, deg_aligned)

        ref_env = extract_envelope(ref_filtered, fs)
        deg_env = extract_envelope(deg_filtered, fs)

        for j, f_mod in enumerate(mod_freqs):
            mod_low, mod_high = f_mod / np.sqrt(2), f_mod * np.sqrt(2)
            sos_mod = butter_bandpass_envelope(mod_low, mod_high, fs)
            ref_env_mod = sosfiltfilt(sos_mod, ref_env)
            deg_env_mod = sosfiltfilt(sos_mod, deg_env)

            m_ref = np.std(ref_env_mod) / (np.mean(ref_env) + 1e-9)
            m_deg = np.std(deg_env_mod) / (np.mean(deg_env) + 1e-9)

            mtf = m_deg / (m_ref + 1e-9)
            mtf_matrix[i, j] = np.clip(mtf, 0, 1)

    return mtf_matrix


def sti_from_mtf(mtf_matrix, snr_per_band, octave_bands, mod_freqs):
    """Calculates the final STI score from the MTF matrix and SNR values."""
    octave_weights = [0.13, 0.14, 0.11, 0.11, 0.19, 0.17, 0.14]

    mod_freq_map = {
        125: [1.60, 8.00], 250: [1.00, 5.00], 500: [0.63, 3.15],
        1000: [2.00, 10.00], 2000: [1.25, 6.30], 4000: [0.80, 4.00],
        8000: [2.50, 12.50],
    }

    with np.errstate(divide='ignore', invalid='ignore'):
        snr_eff_mtf_all_mod = 10 * np.log10(mtf_matrix / (1 - mtf_matrix))

    snr_eff_mtf_all_mod = np.nan_to_num(snr_eff_mtf_all_mod, posinf=15, neginf=-15)
    snr_eff_mtf_all_mod = np.clip(snr_eff_mtf_all_mod, -15, 15)

    avg_snr_eff_per_band_list = []
    for i, band in enumerate(octave_bands):
        specific_mod_freqs = mod_freq_map[band]
        mod_indices = [mod_freqs.index(f) for f in specific_mod_freqs]
        snrs_for_band = snr_eff_mtf_all_mod[i, mod_indices]
        avg_snr_eff_per_band_list.append(np.mean(snrs_for_band))

    avg_snr_eff_per_band = np.array(avg_snr_eff_per_band_list)
    ti_mtf = (avg_snr_eff_per_band + 15) / 30

    snr_per_band = np.array(snr_per_band)
    snr_per_band[np.isnan(snr_per_band)] = -15
    snr_per_band_clipped = np.clip(snr_per_band, -15, 15)
    ti_noise = (snr_per_band_clipped + 15) / 30

    final_ti = np.minimum(ti_mtf, ti_noise)
    sti_score = np.sum(final_ti * octave_weights)

    return sti_score, final_ti, ti_mtf, ti_noise


def calculate_damage_attribution(mtf_measured, snr_per_band, octave_bands, mod_freqs):
    """
    Estimates the contribution of noise vs. reverberation to STI loss.
    NOW RETURNS a dictionary with both absolute and percentage loss.
    """
    print("Attributing degradation sources...")
    mtf_noise_only_matrix = np.zeros_like(mtf_measured)
    for i, snr_db in enumerate(snr_per_band):
        if snr_db is not None and not np.isnan(snr_db):
            mtf_from_snr = 1 / (1 + 10 ** (-snr_db / 10.0))
            mtf_noise_only_matrix[i, :] = mtf_from_snr
        else:
            mtf_noise_only_matrix[i, :] = 1.0

    mtf_reverb_only_matrix = np.divide(mtf_measured, mtf_noise_only_matrix,
                                       out=np.ones_like(mtf_measured),
                                       where=mtf_noise_only_matrix > 1e-6)
    mtf_reverb_only_matrix = np.clip(mtf_reverb_only_matrix, 0, 1)

    for i, snr_db in enumerate(snr_per_band):
        if snr_db is not None and snr_db < 5:
            mtf_reverb_only_matrix[i, :] = 1.0

    infinite_snr = [99.0] * len(octave_bands)
    sti_noise_limited, _, _, _ = sti_from_mtf(mtf_noise_only_matrix, infinite_snr, octave_bands, mod_freqs)
    sti_reverb_limited, _, _, _ = sti_from_mtf(mtf_reverb_only_matrix, infinite_snr, octave_bands, mod_freqs)

    noise_loss = 1.0 - sti_noise_limited
    reverb_loss = 1.0 - sti_reverb_limited
    total_potential_loss = noise_loss + reverb_loss

    if total_potential_loss < 1e-6:
        noise_percent, reverb_percent = 0.0, 0.0
    else:
        noise_percent = (noise_loss / total_potential_loss) * 100
        reverb_percent = (reverb_loss / total_potential_loss) * 100

    return {
        "noise_loss": noise_loss, "reverb_loss": reverb_loss,
        "noise_percent": noise_percent, "reverb_percent": reverb_percent
    }, mtf_noise_only_matrix, mtf_reverb_only_matrix


def calculate_simple_damage_attribution(sti_measured, snr_per_band, octave_bands, mod_freqs):
    """
    Calculates damage attribution based on the simplified model where
    Reverb Damage = Total Damage - Noise Damage.
    """
    perfect_mtf = np.ones((len(octave_bands), len(mod_freqs)))
    sti_noise_limited, _, _, _ = sti_from_mtf(perfect_mtf, snr_per_band, octave_bands, mod_freqs)

    total_loss = 1.0 - sti_measured
    noise_loss = 1.0 - sti_noise_limited
    reverb_loss = total_loss - noise_loss
    
    reverb_loss = max(0, reverb_loss)
    noise_loss = max(0, noise_loss)
    total_loss = noise_loss + reverb_loss

    if total_loss < 1e-6:
        noise_percent, reverb_percent = 0.0, 0.0
    else:
        noise_percent = (noise_loss / total_loss) * 100
        reverb_percent = (reverb_loss / total_loss) * 100

    return {
        "total_loss": total_loss, "noise_loss": noise_loss, "reverb_loss": reverb_loss,
        "noise_percent": noise_percent, "reverb_percent": reverb_percent
    }

# --- Reporting and Visualization ---

def plot_spectrogram_with_noise_markers(signal, fs, is_silent, filename):
    fig, ax = plt.subplots(figsize=(15, 7))
    Pxx, freqs, t, im = ax.specgram(signal, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
    fig.colorbar(im, ax=ax).set_label('Intensity [dB]')
    ax.set_title(f'Spectrogram with Noise Regions Identified\n{os.path.basename(filename)}', fontsize=16)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_ylim(0, 10000)

    if np.any(is_silent):
        silent_samples = np.where(is_silent)[0]
        silent_diff = np.diff(silent_samples)
        starts = np.insert(silent_samples[np.where(silent_diff != 1)[0] + 1], 0, silent_samples[0])
        ends = np.append(silent_samples[np.where(silent_diff != 1)[0]], silent_samples[-1])
        for i, (start_sample, end_sample) in enumerate(zip(starts, ends)):
            label = 'Noise Region' if i == 0 else None
            ax.axvspan(start_sample / fs, end_sample / fs, color='red', alpha=0.3, label=label)
        if len(starts) > 0:
            ax.legend()
    plt.tight_layout()
    plt.show()


def plot_ti_breakdown(ti_per_band, ti_mtf, ti_noise, octave_bands, filename):
    fig, ax = plt.subplots(figsize=(14, 8))
    octave_labels = [f"{b} Hz" for b in octave_bands]
    x = np.arange(len(octave_labels))
    width = 0.25
    ax.bar(x - width, ti_mtf, width, label='TI from Reverb/Modulation', color='skyblue')
    ax.bar(x, ti_noise, width, label='TI from Noise (SNR)', color='salmon')
    ax.bar(x + width, ti_per_band, width, label='Final TI (Limiting Factor)', color='limegreen', hatch='//')
    ax.set_ylabel('Transmission Index (TI) Score')
    ax.set_title(f'Transmission Index Breakdown per Octave Band for\n{os.path.basename(filename)}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(octave_labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.axhline(0.45, color='red', linestyle=':', label='Fair/Poor Threshold (0.45)')
    fig.tight_layout()
    plt.show()


def generate_diagnostic_plots(
        ref_aligned, deg_aligned, fs,
        snr_per_band, damage_contrib,
        mtf_measured, mtf_noise, mtf_reverb,
        octave_bands, mod_freqs, filename
):
    noise_perc = damage_contrib['noise_percent']
    reverb_perc = damage_contrib['reverb_percent']
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    fig.suptitle(f'STI Diagnostic Analysis for: {os.path.basename(filename)}', fontsize=20)

    ax1 = axes[0, 0]
    threshold = 0.05 * np.max(np.abs(ref_aligned))
    try:
        start_idx = np.where(np.abs(ref_aligned) > threshold)[0][0]
    except IndexError:
        start_idx = 0
    samples_to_plot = int(fs * 0.1)
    end_idx = min(start_idx + samples_to_plot, len(ref_aligned))
    ref_plot, deg_plot = ref_aligned[start_idx:end_idx], deg_aligned[start_idx:end_idx]
    ref_plot_norm = ref_plot / (np.max(np.abs(ref_plot)) + 1e-9)
    deg_plot_norm = deg_plot / (np.max(np.abs(deg_plot)) + 1e-9)
    time_axis = np.linspace(0, len(ref_plot_norm) / fs, len(ref_plot_norm))
    ax1.plot(time_axis, ref_plot_norm, label="Reference (Clean)", alpha=0.8, color='blue', linewidth=1.5)
    ax1.plot(time_axis, deg_plot_norm, label="Degraded", alpha=0.7, color='orange', linestyle='--')
    ax1.set_title("1. Signal Alignment Check (0.1s snippet)", fontsize=14)
    ax1.set_xlabel("Time (seconds)"), ax1.set_ylabel("Normalized Amplitude"), ax1.legend(loc="upper right"), ax1.grid(True)

    ax2 = axes[0, 1]
    octave_labels = [f"{b} Hz" for b in octave_bands]
    colors = ['#4CAF50' if snr >= 15 else '#FFC107' if snr >= 5 else '#F44336' for snr in snr_per_band]
    ax2.bar(octave_labels, snr_per_band, color=colors)
    ax2.axhline(15, color='gray', linestyle='--', label='Good SNR (15 dB)')
    ax2.axhline(5, color='darkred', linestyle=':', label='Poor SNR (5 dB)')
    ax2.set_title("2. Estimated SNR per Octave Band", fontsize=14), ax2.set_ylabel("SNR (dB)"), ax2.tick_params(axis='x', rotation=45), ax2.legend()

    ax3 = axes[1, 0]
    labels = [f'Noise ({noise_perc:.1f}%)', f'Reverberation ({reverb_perc:.1f}%)']
    sizes = [noise_perc, reverb_perc]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0) if noise_perc > reverb_perc else (0, 0.05)
    ax3.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors, textprops={'fontsize': 12})
    ax3.axis('equal'), ax3.set_title("3. STI Damage Source Attribution", fontsize=14)

    ax4 = axes[1, 1]
    band_to_plot_idx = 4
    ax4.plot(mod_freqs, mtf_measured[band_to_plot_idx, :], 'o-', label='Measured MTF', color='black', markersize=5)
    ax4.plot(mod_freqs, mtf_noise[band_to_plot_idx, :], '--', label='Noise-only MTF', color='red')
    ax4.plot(mod_freqs, mtf_reverb[band_to_plot_idx, :], ':', label='Reverb-only MTF', color='blue')
    ax4.set_title(f"4. MTF Breakdown ({octave_bands[band_to_plot_idx]} Hz Band)", fontsize=14)
    ax4.set_xlabel("Modulation Frequency (Hz)"), ax4.set_ylabel("MTF Value"), ax4.set_ylim(-0.05, 1.05), ax4.set_xscale('log'), ax4.grid(True, which="both", ls="-", alpha=0.5), ax4.legend()
    plt.show()


def generate_report(sti_score, ti_per_band, snr_per_band, damage_contrib, simple_damage_results, octave_bands, filename):
    """Prints a clear, formatted report to the console including both models."""
    print("\n" + "=" * 80)
    print(f"📊 STI Analysis Report for: {os.path.basename(filename)}")
    print("=" * 80)

    print(f"\nOverall STI Score: {sti_score:.3f}")
    rating = "Excellent" if sti_score >= 0.75 else "Good" if sti_score >= 0.60 else "Fair" if sti_score >= 0.45 else "Poor" if sti_score >= 0.30 else "Bad"
    print(f"Speech Intelligibility Rating: {rating}")

    print("\n--- Standard Damage Attribution (Robust Model) ---")
    print(f"Absolute Loss (NOISE):                  {damage_contrib['noise_loss']:.3f} STI points")
    print(f"Absolute Loss (REVERBERATION):          {damage_contrib['reverb_loss']:.3f} STI points")
    print(f"Percentage Damage (NOISE):          {damage_contrib['noise_percent']:>5.1f}%")
    print(f"Percentage Damage (REVERBERATION):  {damage_contrib['reverb_percent']:>5.1f}%")

    print("\n--- Simplified Damage Attribution (Alternative Model) ---")
    print(f"Absolute Loss (NOISE):                  {simple_damage_results['noise_loss']:.3f} STI points")
    print(f"Absolute Loss (REVERB & OTHER):       {simple_damage_results['reverb_loss']:.3f} STI points")
    print(f"Percentage Damage (NOISE):          {simple_damage_results['noise_percent']:>5.1f}%")
    print(f"Percentage Damage (REVERB & OTHER): {simple_damage_results['reverb_percent']:>5.1f}%")
    
    print("\n--- Details per Octave Band ---")
    print(f"{'Octave Band':<15} | {'SNR (dB)':<10} | {'TI Score':<10} | {'Interpretation'}")
    print("-" * 70)
    for i, band in enumerate(octave_bands):
        snr_val = snr_per_band[i]
        ti_val = ti_per_band[i]
        interp = "SNR couldn't be estimated" if np.isnan(snr_val) else "Very noisy" if snr_val < 5 else "Noisy" if snr_val < 15 else "Clean"
        if ti_val < 0.45: interp += ", Poor intelligibility"
        elif ti_val < 0.60: interp += ", Fair intelligibility"
        else: interp += ", Good intelligibility"
        snr_str = f"{snr_val:<10.1f}" if not np.isnan(snr_val) else "N/A"
        print(f"{f'{band} Hz':<15} | {snr_str} | {ti_val:<10.3f} | {interp}")
    
    print("=" * 80)


# --- Main Orchestrator ---

def analyze_sti(ref_file, deg_file, align_the_signals=True, show_plots=True):
    octave_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    mod_freqs = [0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00, 10.00, 12.50]

    try:
        ref_sig, fs_ref = sf.read(ref_file)
        deg_sig, fs_deg = sf.read(deg_file)
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return

    fs = fs_ref
    if fs_ref != fs_deg:
        target_fs = 44100
        print(f"Warning: Sample rates do not match. Resampling both to {target_fs} Hz.")
        ref_sig = resample(ref_sig, int(len(ref_sig) * target_fs / fs_ref))
        deg_sig = resample(deg_sig, int(len(deg_sig) * target_fs / fs_deg))
        fs = target_fs

    if align_the_signals:
        ref_aligned, deg_aligned = align_signals(ref_sig, deg_sig, fs)
    else:
        print("Skipping signal alignment as requested.")
        if ref_sig.ndim > 1: ref_sig = ref_sig.mean(axis=1)
        if deg_sig.ndim > 1: deg_sig = deg_sig.mean(axis=1)
        min_len = min(len(ref_sig), len(deg_sig))
        ref_aligned, deg_aligned = ref_sig[:min_len], deg_sig[:min_len]

    snr_per_band, is_silent = estimate_snr_per_band(ref_aligned, deg_aligned, fs, octave_bands)
    mtf_matrix = calculate_mtf_matrix(ref_aligned, deg_aligned, fs, octave_bands, mod_freqs)
    sti_score, ti_per_band, ti_mtf, ti_noise = sti_from_mtf(mtf_matrix, snr_per_band, octave_bands, mod_freqs)
    
    # --- Run BOTH damage attribution models ---
    damage_contrib, mtf_noise, mtf_reverb = calculate_damage_attribution(mtf_matrix, snr_per_band, octave_bands, mod_freqs)
    simple_damage_results = calculate_simple_damage_attribution(sti_score, snr_per_band, octave_bands, mod_freqs)

    # --- Generate unified report ---
    generate_report(sti_score, ti_per_band, snr_per_band, damage_contrib, simple_damage_results, octave_bands, deg_file)
    
    if show_plots:
        plt.style.use('seaborn-v0_8-whitegrid')
        plot_spectrogram_with_noise_markers(deg_aligned, fs, is_silent, deg_file)
        plot_ti_breakdown(ti_per_band, ti_mtf, ti_noise, octave_bands, deg_file)
        generate_diagnostic_plots(ref_aligned, deg_aligned, fs, snr_per_band, damage_contrib, mtf_matrix, mtf_noise, mtf_reverb, octave_bands, mod_freqs, deg_file)
    
    # FIX: Add all the new values to the dictionary that is returned
    results = {
        "filepath": deg_file, "filename": os.path.basename(deg_file), "sti_score": sti_score,
        
        "standard_noise_percent": damage_contrib['noise_percent'], 
        "standard_reverb_percent": damage_contrib['reverb_percent'],
        "standard_noise_loss": damage_contrib['noise_loss'],
        "standard_reverb_loss": damage_contrib['reverb_loss'],

        "simple_noise_percent": simple_damage_results['noise_percent'],
        "simple_reverb_percent": simple_damage_results['reverb_percent'],
        "simple_noise_loss": simple_damage_results['noise_loss'],
        "simple_reverb_loss": simple_damage_results['reverb_loss'],

        "snr_per_band": snr_per_band, 
        "avg_snr": np.nanmean([s for s in snr_per_band if s is not None and not np.isnan(s)]),
        "ti_per_band": ti_per_band, 
        "ti_mtf": ti_mtf, 
        "ti_noise": ti_noise,
    }
    return results


if __name__ == '__main__':
    reference_audio = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\STIPA ref.wav"
    degraded_audio = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\small dataset\dataset\stipa_0.17s_ambient noise_SNR-10dB.wav"

    if not all(os.path.exists(f) for f in [reference_audio, degraded_audio]):
        print("Error: One or more of the specified audio files were not found. Please check the paths.")
    else:
        analyze_sti(
            ref_file=reference_audio,
            deg_file=degraded_audio,
            align_the_signals=False,
            show_plots=True,
        )
