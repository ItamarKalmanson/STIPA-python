import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, correlate, chirp, spectrogram
import matplotlib.pyplot as plt


def chirp_align_and_extract(ref_sig, deg_sig, fs, ref_path, deg_path):
    print(">> Synchronizing using precise Chirp Correlation...")

    # 1. נייצר את הצ'ירפ לחיפוש
    t_chirp = np.linspace(0, 1.5, int(1.5 * fs), endpoint=False)
    sync_chirp = chirp(t_chirp, f0=1000, f1=2000, t1=1.5, method='linear')

    # חיפוש בהקלטה (ב-30 השניות הראשונות)
    search_len = min(len(deg_sig), int(30 * fs))
    deg_chunk = deg_sig[:search_len]

    corr = correlate(deg_chunk, sync_chirp, mode='valid', method='fft')
    deg_chirp_start = np.argmax(np.abs(corr))

    print(f">> Found Chirp in recording at sample {deg_chirp_start} ({deg_chirp_start / fs:.3f} seconds).")

    # 2. חיתוך הקבצים המלאים (כולל הצ'ירפ!) לשמירה
    stipa_duration = int(25.0 * fs)
    offset_samples = int(3.5 * fs)  # זמן הצ'ירפ + השקט
    total_duration = offset_samples + stipa_duration

    # גזירת הקבצים המלאים (מהצ'ירפ ועד סוף ה-STIPA)
    aligned_deg_full = deg_sig[deg_chirp_start: deg_chirp_start + total_duration]
    # ברפרנס הצ'ירפ מתחיל בזמן 0
    aligned_ref_full = ref_sig[0: total_duration]

    # וידוא אורך זהה
    min_len = min(len(aligned_ref_full), len(aligned_deg_full))
    aligned_ref_full = aligned_ref_full[:min_len]
    aligned_deg_full = aligned_deg_full[:min_len]

    # 3. שמירת הקבצים המלאים (כולל הצ'ירפ)
    ref_name = os.path.basename(ref_path)
    deg_name = os.path.basename(deg_path)
    base_dir = os.path.dirname(ref_path)
    aligned_dir = os.path.join(base_dir, "aligned")

    if not os.path.exists(aligned_dir):
        os.makedirs(aligned_dir)

    ref_out = os.path.join(aligned_dir, f"FULL_ALIGNED_{ref_name+deg_name}")
    deg_out = os.path.join(aligned_dir, f"FULL_ALIGNED_{deg_name}")

    print(f">> Saving full aligned audio (with Chirp) to:\n   - {ref_out}\n   - {deg_out}\n")
    sf.write(ref_out, aligned_ref_full, fs)
    sf.write(deg_out, aligned_deg_full, fs)

    # 4. ציור הספקטרוגרמה של 5 השניות הראשונות לווידוא סנכרון
    print(">> Plotting Spectrograms for visual alignment verification...")
    plot_len = min(int(5.0 * fs), len(aligned_ref_full))

    plt.figure(figsize=(12, 8))

    # ספקטרוגרמה של הרפרנס
    plt.subplot(2, 1, 1)
    f, t, Sxx = spectrogram(aligned_ref_full[:plot_len], fs ,nperseg = 1024)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.title('Spectrogram - Reference Signal (First 5 Seconds)')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 8000)  # נתמקד בתדרים החשובים

    # ספקטרוגרמה של ההקלטה
    plt.subplot(2, 1, 2)
    f, t, Sxx = spectrogram(aligned_deg_full[:plot_len], fs, nperseg = 1024)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.title('Spectrogram - Recorded Signal (First 5 Seconds)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, 8000)

    plt.tight_layout()
    plt.show()

    # 5. חילוץ רק ה-STIPA (חיתוך הצ'ירפ והשקט) למטרת החישוב המתמטי בלבד
    stipa_ref_only = aligned_ref_full[offset_samples:]
    stipa_deg_only = aligned_deg_full[offset_samples:]

    return stipa_ref_only, stipa_deg_only


def calculate_stipa_from_scratch(ref_path, deg_path):
    print("1. Loading raw audio files...")
    ref_sig, fs = librosa.load(ref_path, sr=None, mono=True)
    deg_sig, _ = librosa.load(deg_path, sr=None, mono=True)

    # היישור והחיתוך
    ref_stipa, deg_stipa = chirp_align_and_extract(ref_sig, deg_sig, fs, ref_path, deg_path)

    # מעכשיו המתמטיקה רצה *אך ורק* על ה-STIPA
    time = np.arange(len(ref_stipa)) / fs

    octave_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    weights = [0.13, 0.14, 0.11, 0.12, 0.19, 0.17, 0.14]

    stipa_mod_freqs = {
        125: [1.6, 8.0], 250: [1.0, 5.0], 500: [0.63, 3.15],
        1000: [2.0, 10.0], 2000: [1.25, 6.3], 4000: [0.8, 4.0], 8000: [2.5, 12.5]
    }

    mtf_matrix = np.zeros((len(octave_bands), 2))
    mti_scores = np.zeros(len(octave_bands))

    print("2. Calculating STIPA Matrices on 25s clean signals...")
    for i, band in enumerate(octave_bands):
        nyq = 0.5 * fs
        low = (band / np.sqrt(2)) / nyq
        high = (band * np.sqrt(2)) / nyq

        sos_bp = butter(4, [low, high], btype='band', output='sos')
        ref_band = sosfiltfilt(sos_bp, ref_stipa)
        deg_band = sosfiltfilt(sos_bp, deg_stipa)

        ref_env = np.maximum(ref_band ** 2, 0)
        deg_env = np.maximum(deg_band ** 2, 0)

        sos_lp = butter(4, 20 / nyq, btype='low', output='sos')
        ref_env = sosfiltfilt(sos_lp, ref_env)
        deg_env = sosfiltfilt(sos_lp, deg_env)

        ref_dc = np.mean(ref_env)
        deg_dc = np.mean(deg_env)

        ti_sum = 0
        current_mod_freqs = stipa_mod_freqs[band]

        for j, f_mod in enumerate(current_mod_freqs):
            ref_cos = np.sum(ref_env * np.cos(2 * np.pi * f_mod * time))
            ref_sin = np.sum(ref_env * np.sin(2 * np.pi * f_mod * time))
            ref_mag = np.sqrt(ref_cos ** 2 + ref_sin ** 2) / (len(time) / 2)
            m_ref = ref_mag / (ref_dc + 1e-10)

            deg_cos = np.sum(deg_env * np.cos(2 * np.pi * f_mod * time))
            deg_sin = np.sum(deg_env * np.sin(2 * np.pi * f_mod * time))
            deg_mag = np.sqrt(deg_cos ** 2 + deg_sin ** 2) / (len(time) / 2)
            m_deg = deg_mag / (deg_dc + 1e-10)

            mtf = m_deg / (m_ref + 1e-6)
            mtf = np.clip(mtf, 0.001, 1.0)
            mtf_matrix[i, j] = mtf

            snr = 10 * np.log10(mtf / (1 - mtf + 1e-6))
            snr = np.clip(snr, -15, 15)

            ti = (snr + 15) / 30
            ti_sum += ti

        mti_scores[i] = ti_sum / 2.0

    overall_stipa = np.sum(mti_scores * weights)

    print("\n" + "=" * 65)
    print("STIPA MTF MATRIX (Modulation Transfer Function)")
    print("=" * 65)
    print(f"{'Band':<8} | {'Mod Freq 1':<15} | {'Mod Freq 2':<15}")
    print("-" * 65)

    for i, band in enumerate(octave_bands):
        f1, f2 = stipa_mod_freqs[band]
        mtf1, mtf2 = mtf_matrix[i]
        print(f"{band:<4} Hz | {f1:>5} Hz: {mtf1:.3f}   | {f2:>5} Hz: {mtf2:.3f}")

    print("\n" + "=" * 30)
    print("MTI (Scores per Octave Band)")
    print("=" * 30)
    for i, band in enumerate(octave_bands):
        print(f"{band:4} Hz: {mti_scores[i]:.3f}")

    print("\n" + "=" * 30)
    print(f"OVERALL STIPA SCORE: {overall_stipa:.3f}")
    print("=" * 30)


if __name__ == "__main__":
    reference_audio = r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\part2\MASTER_SYNC_STIPA.wav"
    test_audios = [r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\part2\2 מטר על השולחן.wav"
                    ,r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\part2\6 מטר למטה.wav"
                    ,r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\part2\קצה הבית.wav"
                    ,r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\part2\שירותים.wav"]

    for test_audio in test_audios:
        calculate_stipa_from_scratch(reference_audio, test_audio)