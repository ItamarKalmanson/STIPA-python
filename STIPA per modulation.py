import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, correlate, chirp, spectrogram
import matplotlib.pyplot as plt


def process_and_align_audio(ref_path, deg_path, output_dir):
    print(f"\n--- Processing: {os.path.basename(deg_path)} ---")
    ref_sig, fs = librosa.load(ref_path, sr=None, mono=True)
    deg_sig, _ = librosa.load(deg_path, sr=fs, mono=True)

    t_chirp = np.linspace(0, 1.5, int(1.5 * fs), endpoint=False)
    sync_chirp = chirp(t_chirp, f0=1000, f1=2000, t1=1.5, method='linear')

    search_len = min(len(deg_sig), int(30 * fs))
    deg_chunk = deg_sig[:search_len]

    corr = correlate(deg_chunk, sync_chirp, mode='valid', method='fft')
    deg_chirp_start = np.argmax(np.abs(corr))

    stipa_duration = int(25.0 * fs)
    offset_samples = int(3.5 * fs)
    total_duration = offset_samples + stipa_duration

    aligned_deg_full = deg_sig[deg_chirp_start: deg_chirp_start + total_duration]
    aligned_ref_full = ref_sig[0: total_duration]

    min_len = min(len(aligned_ref_full), len(aligned_deg_full))
    aligned_ref_full = aligned_ref_full[:min_len]
    aligned_deg_full = aligned_deg_full[:min_len]

    aligned_dir = os.path.join(output_dir, "aligned_audio")
    if not os.path.exists(aligned_dir):
        os.makedirs(aligned_dir)

    stipa_deg_only = aligned_deg_full[offset_samples:]
    deg_name = os.path.basename(deg_path)
    clean_deg_out = os.path.join(aligned_dir, f"CLEAN_{deg_name}")
    sf.write(clean_deg_out, stipa_deg_only, fs)

    # שמירת הספקטרוגרמה במקום להציג אותה כדי לא לעצור את הלולאה
    plot_len = min(int(5.0 * fs), len(aligned_ref_full))
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    f_axis, t_axis, Sxx = spectrogram(aligned_ref_full[:plot_len], fs)
    plt.pcolormesh(t_axis, f_axis, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.title('Reference Signal (First 5s)')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 8000)

    plt.subplot(2, 1, 2)
    f_axis, t_axis, Sxx = spectrogram(aligned_deg_full[:plot_len], fs)
    plt.pcolormesh(t_axis, f_axis, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    plt.title('Recorded Signal (First 5s)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, 8000)

    plt.tight_layout()
    plots_dir = os.path.join(output_dir, "spectrograms")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(os.path.join(plots_dir, f"SYNC_{deg_name}.png"))
    plt.close()

    return stipa_deg_only, fs


def calculate_stipa_core(deg_sig, fs):
    settle_time = 1.0
    settle_samples = int(settle_time * fs)
    time = np.arange(len(deg_sig) - settle_samples) / fs

    octave_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    weights = [0.13, 0.14, 0.11, 0.12, 0.19, 0.17, 0.14]

    stipa_mod_freqs = {
        125: [1.6, 8.0], 250: [1.0, 5.0], 500: [0.63, 3.15],
        1000: [2.0, 10.0], 2000: [1.25, 6.3], 4000: [0.8, 4.0], 8000: [2.5, 12.5]
    }

    mtf_matrix = np.zeros((len(octave_bands), 2))
    mti_scores = np.zeros(len(octave_bands))

    for i, band in enumerate(octave_bands):
        nyq = 0.5 * fs
        low = (band / np.sqrt(2)) / nyq
        high = (band * np.sqrt(2)) / nyq

        sos_bp = butter(4, [low, high], btype='band', output='sos')
        deg_band = sosfiltfilt(sos_bp, deg_sig)

        deg_env = np.maximum(deg_band ** 2, 0)

        sos_lp = butter(4, 20 / nyq, btype='low', output='sos')
        deg_env = sosfiltfilt(sos_lp, deg_env)

        deg_env_stable = deg_env[settle_samples:]
        deg_dc = np.mean(deg_env_stable)

        ti_sum = 0
        current_mod_freqs = stipa_mod_freqs[band]

        for j, f_mod in enumerate(current_mod_freqs):
            deg_cos = np.sum(deg_env_stable * np.cos(2 * np.pi * f_mod * time))
            deg_sin = np.sum(deg_env_stable * np.sin(2 * np.pi * f_mod * time))
            deg_mag = np.sqrt(deg_cos ** 2 + deg_sin ** 2) / (len(time) / 2)
            m_deg = deg_mag / (deg_dc + 1e-10)

            m_ref_theoretical = 0.5

            mtf = m_deg / m_ref_theoretical
            mtf = np.clip(mtf, 0.001, 1.0)
            mtf_matrix[i, j] = mtf

            snr = 10 * np.log10(mtf / (1 - mtf + 1e-6))
            snr = np.clip(snr, -15, 15)

            ti = (snr + 15) / 30
            ti_sum += ti

        mti_scores[i] = ti_sum / 2.0

    overall_stipa = np.sum(mti_scores * weights)

    return overall_stipa, mti_scores, mtf_matrix, octave_bands, stipa_mod_freqs


def plot_mtf_vs_mod_freq(mtf_matrix, octave_bands, stipa_mod_freqs, file_name, output_dir):
    freqs = []
    mtfs = []

    # חילוץ כל 14 תדרי המודולציה וה-MTF שלהם
    for i, band in enumerate(octave_bands):
        f1, f2 = stipa_mod_freqs[band]
        mtf1, mtf2 = mtf_matrix[i]
        freqs.extend([f1, f2])
        mtfs.extend([mtf1, mtf2])

    # סידור התדרים מהנמוך לגבוה
    sorted_indices = np.argsort(freqs)
    freqs = np.array(freqs)[sorted_indices]
    mtfs = np.array(mtfs)[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, mtfs, marker='o', linestyle='-', color='b', markersize=8)
    plt.title(f'MTF vs Modulation Frequency\n{file_name}')
    plt.xlabel('Modulation Frequency (Hz)')
    plt.ylabel('MTF (Modulation Transfer Function)')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)

    # הוספת התוויות של התדרים על ציר ה-X
    plt.xticks(freqs, rotation=45)

    plt.tight_layout()
    plots_dir = os.path.join(output_dir, "mtf_plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.savefig(os.path.join(plots_dir, f"MTF_Plot_{file_name}.png"))
    plt.close()


def run_batch_analysis(reference_audio, target_folder, plot_mtf_scores = True, save_scores_to_csv = False):
    # מציאת כל קבצי ה-WAV בתיקייה (פרט לרפרנס)
    wav_files = glob.glob(os.path.join(target_folder, "*.wav"))
    recordings = [f for f in wav_files if f != reference_audio and "MASTER" not in f and "CLEAN" not in f]

    if not recordings:
        print("No recordings found to process!")
        return

    summary_data = []
    octave_data = []

    # יצירת תיקיית פלט מסודרת
    output_dir = os.path.join(target_folder, "STIPA_Results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for rec_path in recordings:
        file_name = os.path.basename(rec_path)

        try:
            # יישור וחיתוך
            # clean_deg_sig, fs = process_and_align_audio(reference_audio, rec_path, output_dir)
            clean_deg_sig, fs = sf.read(rec_path)

            # חישוב מתמטי
            overall_stipa, mti_scores, mtf_matrix, octave_bands, mod_freqs = calculate_stipa_core(clean_deg_sig, fs)

            if plot_mtf_scores == True:

                # ציור הגרף (MTF vs Mod Freq)
                plot_mtf_vs_mod_freq(mtf_matrix, octave_bands, mod_freqs, file_name, output_dir)

            # איסוף הנתונים לדוחות
            summary_data.append({"File Name": file_name, "Overall STIPA Score": overall_stipa})

            octave_row = {"File Name": file_name}
            for idx, band in enumerate(octave_bands):
                octave_row[f"{band} Hz"] = mti_scores[idx]
            octave_data.append(octave_row)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if save_scores_to_csv == True:
            # שמירת הדוחות לקבצי אקסל (CSV)
        df_summary = pd.DataFrame(summary_data)
        df_octaves = pd.DataFrame(octave_data)

        summary_path = os.path.join(output_dir, "STIPA_Summary_Report.csv")
        octaves_path = os.path.join(output_dir, "STIPA_Octaves_Report.csv")

        df_summary.to_csv(summary_path, index=False)
        df_octaves.to_csv(octaves_path, index=False)

        print("\n" + "=" * 50)
        print("ALL PROCESSING COMPLETE!")
        print(f"Results saved in: {output_dir}")
        print("=" * 50)


if __name__ == "__main__":
    # נתיב לקובץ המאסטר
    reference_audio = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\MASTER_SYNC_STIPA.wav"

    # נתיב *לתיקייה* שבה נמצאות כל ההקלטות
    folder_with_recordings = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\stipa_dataset 2"
    run_batch_analysis(reference_audio, folder_with_recordings, plot_mtf_scores = True, save_scores_to_csv = True)