import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample, sosfiltfilt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Prerequisite functions ---

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band', output='sos')

def butter_bandpass_envelope(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band', output='sos')

def extract_envelope(signal, fs, cutoff_hz=100):
    # The sosfiltfilt function requires the input signal to be longer than the filter's padding length.
    # This check prevents crashes on very short signals. The root cause of the padlen error, however,
    # was numerical instability from high sample rates, which is fixed in the main orchestrator.
    if len(signal) <= 10:
        return np.abs(signal)
    sos_lp = butter(2, cutoff_hz / (0.5 * fs), 'low', output='sos')
    return sosfiltfilt(sos_lp, np.abs(signal))

def align_signals(ref_sig, deg_sig, fs):
    print("Aligning signals...")
    if ref_sig.ndim > 1: ref_sig = ref_sig.mean(axis=1)
    if deg_sig.ndim > 1: deg_sig = deg_sig.mean(axis=1)
    
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
    
    print(f"Signals aligned. Delay of {lag/fs:.3f}s corrected.")
    return ref_aligned, deg_aligned

def estimate_snr_per_band(ref_aligned, deg_aligned, fs, octave_bands):
    print("Estimating SNR per octave band...")
    snr_bands = []
    
    ref_envelope = extract_envelope(ref_aligned, fs, cutoff_hz=50)
    silence_threshold = 0.02 * np.max(ref_envelope) if len(ref_envelope) > 0 else 0
    is_silent = ref_envelope < silence_threshold
    is_signal = ~is_silent

    if not np.any(is_silent):
        print("Warning: Could not find silent parts in reference signal to estimate noise.")
        return [np.nan] * len(octave_bands)

    for band in octave_bands:
        lowcut, highcut = band / np.sqrt(2), band * np.sqrt(2)
        sos = butter_bandpass(lowcut, highcut, fs)
        deg_filtered = sosfiltfilt(sos, deg_aligned)

        signal_power = np.mean(deg_filtered[is_signal]**2)
        noise_power = np.mean(deg_filtered[is_silent]**2)

        if noise_power < 1e-20:
            snr_db = 99.0
        else:
            true_signal_power = signal_power - noise_power
            snr_db = 10 * np.log10(true_signal_power / noise_power) if true_signal_power > 0 else 0.0
        snr_bands.append(snr_db)

    return snr_bands

def calculate_mtf_matrix(ref_aligned, deg_aligned, fs, octave_bands, mod_freqs):
    print("Calculating MTF matrix...")
    mtf_matrix = np.zeros((len(octave_bands), len(mod_freqs)))
    for i, band in enumerate(octave_bands):
        lowcut, highcut = band / np.sqrt(2), band * np.sqrt(2)
        sos_band = butter_bandpass(lowcut, highcut, fs)
        ref_filtered = sosfiltfilt(sos_band, ref_aligned)
        deg_filtered = sosfiltfilt(sos_band, deg_aligned)

        ref_env = extract_envelope(ref_filtered, fs)
        deg_env = extract_envelope(deg_filtered, fs)

        for j, f_mod in enumerate(mod_freqs):
            sos_mod = butter_bandpass_envelope(f_mod / np.sqrt(2), f_mod * np.sqrt(2), fs)
            ref_env_mod = sosfiltfilt(sos_mod, ref_env)
            deg_env_mod = sosfiltfilt(sos_mod, deg_env)

            m_ref = np.std(ref_env_mod) / (np.mean(ref_env) + 1e-9)
            m_deg = np.std(deg_env_mod) / (np.mean(deg_env) + 1e-9)
            
            mtf = m_deg / (m_ref + 1e-9)
            mtf_matrix[i, j] = np.clip(mtf, 0, 1)
    return mtf_matrix

def sti_from_mtf(mtf_matrix, snr_per_band, octave_bands, mod_freqs):
    octave_weights = [0.13, 0.14, 0.11, 0.11, 0.19, 0.17, 0.14]
    
    mod_freq_map = {
        125: [1.60, 8.00], 250: [1.00, 5.00], 500: [0.63, 3.15],
        1000: [2.00, 10.00], 2000: [1.25, 6.30], 4000: [0.80, 4.00],
        8000: [2.50, 12.50],
    }

    with np.errstate(divide='ignore', invalid='ignore'):
        snr_eff_mtf = 10 * np.log10(mtf_matrix / (1 - mtf_matrix))
    
    snr_eff_mtf = np.nan_to_num(snr_eff_mtf, posinf=15, neginf=-15)
    snr_eff_mtf = np.clip(snr_eff_mtf, -15, 15)
    
    avg_snr_eff_per_band_list = []
    for i, band in enumerate(octave_bands):
        specific_mod_freqs = mod_freq_map[band]
        mod_indices = [mod_freqs.index(f) for f in specific_mod_freqs]
        snrs_for_band = snr_eff_mtf[i, mod_indices]
        avg_snr_eff_per_band_list.append(np.mean(snrs_for_band))

    avg_snr_eff_per_band = np.array(avg_snr_eff_per_band_list)
    ti_mtf = (avg_snr_eff_per_band + 15) / 30

    snr_per_band = np.array(snr_per_band)
    snr_per_band[np.isnan(snr_per_band)] = -15
    snr_per_band_clipped = np.clip(snr_per_band, -15, 15)
    ti_noise = (snr_per_band_clipped + 15) / 30

    final_ti = np.minimum(ti_mtf, ti_noise)
    sti_score = np.sum(final_ti * octave_weights)
    return sti_score

def calculate_simple_damage_attribution(sti_measured, snr_per_band, octave_bands, mod_freqs):
    print("Calculating damage attribution with simplified model...")

    perfect_mtf = np.ones((len(octave_bands), len(mod_freqs)))
    sti_noise_limited = sti_from_mtf(perfect_mtf, snr_per_band, octave_bands, mod_freqs)

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

def generate_simple_report(sti_score, damage_results, snr_per_band, octave_bands, filename):
    print("\n" + "="*70)
    print(f"📊 Simplified STI Analysis Report for: {os.path.basename(filename)}")
    print("="*70)
    print(f"\nOverall STI Score: {sti_score:.3f}")
    rating = "Excellent" if sti_score >= 0.75 else "Good" if sti_score >= 0.60 else "Fair" if sti_score >= 0.45 else "Poor" if sti_score >= 0.30 else "Bad"
    print(f"Speech Intelligibility Rating: {rating}")
    print("\n--- Absolute STI Loss Attribution ---")
    print(f"Total STI points lost from perfect (1.0): {damage_results['total_loss']:.3f}")
    print(f"  - Attributed to NOISE:                  {damage_results['noise_loss']:.3f}")
    print(f"  - Attributed to REVERB & OTHER:       {damage_results['reverb_loss']:.3f}")
    print("\n--- Percentage Damage Attribution ---")
    print(f"Estimated damage from NOISE:          {damage_results['noise_percent']:>5.1f}%")
    print(f"Estimated damage from REVERB & OTHER: {damage_results['reverb_percent']:>5.1f}%")
    primary_issue = "BACKGROUND NOISE" if damage_results['noise_percent'] > damage_results['reverb_percent'] else "REVERB & OTHER FACTORS"
    print(f"\nConclusion: The primary source of degradation is {primary_issue}.")
    print("\n--- Details per Octave Band (SNR) ---")
    print(f"{'Octave Band':<15} | {'SNR (dB)':<10} | {'Interpretation'}")
    print("-" * 50)
    for i, band in enumerate(octave_bands):
        snr_val = snr_per_band[i]
        interp = "SNR couldn't be estimated" if np.isnan(snr_val) else "Very noisy" if snr_val < 5 else "Noisy" if snr_val < 15 else "Clean"
        snr_str = f"{snr_val:<10.1f}" if not np.isnan(snr_val) else "N/A"
        print(f"{f'{band} Hz':<15} | {snr_str} | {interp}")
    print("="*70)

# --- Main Orchestrator ---

def analyze_sti_simple(ref_file, deg_file, align_the_signals=True):
    octave_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    mod_freqs = [0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00, 10.00, 12.50]

    try:
        ref_sig, fs_ref = sf.read(ref_file)
        deg_sig, fs_deg = sf.read(deg_file)
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return

    if len(ref_sig) < 200 or len(deg_sig) < 200:
        print("Fatal Error: One or both audio files are too short for analysis (< 200 samples).")
        return

    fs = fs_ref
    # CRITICAL FIX: If sample rates differ, resample BOTH signals to a standard, stable rate (44100 Hz).
    if fs_ref != fs_deg:
        target_fs = 44100
        print(f"Warning: Sample rates do not match (Ref: {fs_ref} Hz, Deg: {fs_deg} Hz). Resampling both to {target_fs} Hz.")
        
        num_samples_ref = int(len(ref_sig) * target_fs / fs_ref)
        ref_sig = resample(ref_sig, num_samples_ref)
        
        num_samples_deg = int(len(deg_sig) * target_fs / fs_deg)
        deg_sig = resample(deg_sig, num_samples_deg)
        
        fs = target_fs # The new, stable sample rate for all subsequent analysis

    if align_the_signals:
        ref_aligned, deg_aligned = align_signals(ref_sig, deg_sig, fs)
    else:
        print("Skipping signal alignment.")
        min_len = min(len(ref_sig), len(deg_sig))
        ref_aligned, deg_aligned = ref_sig[:min_len], deg_sig[:min_len]

    snr_per_band = estimate_snr_per_band(ref_aligned, deg_aligned, fs, octave_bands)
    mtf_matrix = calculate_mtf_matrix(ref_aligned, deg_aligned, fs, octave_bands, mod_freqs)
    sti_measured = sti_from_mtf(mtf_matrix, snr_per_band, octave_bands, mod_freqs)
    damage_results = calculate_simple_damage_attribution(sti_measured, snr_per_band, octave_bands, mod_freqs)

    generate_simple_report(sti_measured, damage_results, snr_per_band, octave_bands, deg_file)

    return {"sti_score": sti_measured, **damage_results}

if __name__ == '__main__':
    reference_audio = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\STIPA ref.wav"
    degraded_audio = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\small dataset\dataset\stipa_0.17s_ambient noise_SNR10dB.wav"

    if not all(os.path.exists(f) for f in [reference_audio, degraded_audio]):
        print("Error: One or more audio files not found. Please check the paths.")
    else:
        analyze_sti_simple(
            ref_file=reference_audio,
            deg_file=degraded_audio,
            align_the_signals=False 
        )
