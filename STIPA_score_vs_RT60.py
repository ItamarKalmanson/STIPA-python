import os
import glob
import re
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
from collections import defaultdict


def calculate_stipa_core(deg_sig, fs):
    """
    חישוב מתמטי נטו של ציון ה-STIPA.
    """
    settle_time = 1.0
    settle_samples = int(settle_time * fs)
    time = np.arange(len(deg_sig) - settle_samples) / fs

    octave_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    weights = [0.13, 0.14, 0.11, 0.12, 0.19, 0.17, 0.14]

    stipa_mod_freqs = {
        125: [1.6, 8.0], 250: [1.0, 5.0], 500: [0.63, 3.15],
        1000: [2.0, 10.0], 2000: [1.25, 6.3], 4000: [0.8, 4.0], 8000: [2.5, 12.5]
    }

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
            mtf = np.clip(m_deg / m_ref_theoretical, 0.001, 1.0)
            snr = np.clip(10 * np.log10(mtf / (1 - mtf + 1e-6)), -15, 15)
            ti = (snr + 15) / 30
            ti_sum += ti

        mti_scores[i] = ti_sum / 2.0

    overall_stipa = np.sum(mti_scores * weights)
    return overall_stipa


def plot_stipa_vs_rt60_multi_snr(results, output_dir):
    """
    מייצר גרף של STIPA לעומת RT60, עם קווים נפרדים לכל רמת SNR.
    """
    # מילון שיחזיק לכל רמת SNR את רשימת הנקודות שלה (RT60, Score)
    grouped_data = defaultdict(list)

    for filename, score in results:
        # חילוץ זמן ההדהוד
        rt_match = re.search(r'stipa_([\d\.]+)s', filename)
        # חילוץ ה-SNR (תומך גם במינוס, למשל SNR-10dB)
        snr_match = re.search(r'SNR(-?\d+)dB', filename)

        if rt_match and snr_match:
            rt60 = float(rt_match.group(1))
            snr = int(snr_match.group(1))
            grouped_data[snr].append((rt60, score))
        else:
            print(f"Warning: Could not extract RT60 or SNR from '{filename}'. Skipping.")

    if not grouped_data:
        print("No valid data found to plot.")
        return

    plt.figure(figsize=(10, 6))

    # צבעים לקווים השונים (מסודרים מהכי טוב להכי גרוע)
    colors = ['blue', 'black', 'red', 'purple', 'orange']

    # מיון ה-SNRs מהגבוה לנמוך (למשל 10, 0, -10) כדי שיצוירו בסדר הגיוני
    snr_levels = sorted(grouped_data.keys(), reverse=True)

    all_rt60s = []

    # ציור קו נפרד לכל SNR
    for i, snr in enumerate(snr_levels):
        # מיון הנקודות של אותו SNR לפי זמן ההדהוד כדי שהקו ירוץ משמאל לימין
        data = sorted(grouped_data[snr], key=lambda x: x[0])
        rt60_values = [x[0] for x in data]
        stipa_scores = [x[1] for x in data]
        all_rt60s.extend(rt60_values)

        color = colors[i % len(colors)]
        plt.plot(rt60_values, stipa_scores, marker='o', linestyle='-', linewidth=2, markersize=8, color=color,
                 label=f'SNR {snr}dB', zorder=5)

    if not all_rt60s:
        return

    min_rt60 = min(all_rt60s)
    max_rt60 = max(all_rt60s)

    plt.title('STIPA Score vs. Reverberation (RT60) under different SNRs', fontsize=14, fontweight='bold')
    plt.xlabel('RT60 [seconds]', fontsize=12)
    plt.ylabel('STIPA Score', fontsize=12)
    plt.ylim(0, 1.05)
    plt.xlim(max(0, min_rt60 - 0.2), max_rt60 + 0.2)
    plt.grid(True, linestyle='--', alpha=0.6)

    # הוספת צבעי התקן של STIPA ברקע
    plt.axhspan(0.75, 1.05, color='green', alpha=0.15)
    plt.axhspan(0.60, 0.75, color='lightgreen', alpha=0.15)
    plt.axhspan(0.45, 0.60, color='yellow', alpha=0.15)
    plt.axhspan(0.30, 0.45, color='orange', alpha=0.15)
    plt.axhspan(0.00, 0.30, color='red', alpha=0.15)

    # יצירת טקסט למקרא צבעי הרקע בצד
    plt.text(max_rt60 + 0.22, 0.90, 'Excellent', color='green', verticalalignment='center')
    plt.text(max_rt60 + 0.22, 0.675, 'Good', color='olivedrab', verticalalignment='center')
    plt.text(max_rt60 + 0.22, 0.525, 'Fair', color='goldenrod', verticalalignment='center')
    plt.text(max_rt60 + 0.22, 0.375, 'Poor', color='darkorange', verticalalignment='center')
    plt.text(max_rt60 + 0.22, 0.15, 'Bad', color='red', verticalalignment='center')

    plt.legend(loc='upper right', framealpha=0.9)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "STIPA_vs_RT60_Multi_SNR.png")
    plt.savefig(plot_path, bbox_inches='tight')  # מוודא שהטקסט בצד לא נחתך
    plt.show()
    print(f"\n-> Graph saved at: {plot_path}")


def run_minimal_analysis(target_folder):
    """
    סורק את הקבצים, מחשב STIPA, ומעביר לשרטוט.
    """
    wav_files = glob.glob(os.path.join(target_folder, "*.wav"))
    recordings = [f for f in wav_files if "MASTER" not in f and "CLEAN" not in f]

    if not recordings:
        print("No recordings found to process!")
        return

    output_dir = os.path.join(target_folder, "STIPA_Results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    for rec_path in recordings:
        file_name = os.path.basename(rec_path)
        print(f"Processing: {file_name}")

        try:
            sig, fs = sf.read(rec_path)
            if len(sig.shape) > 1:
                sig = sig[:, 0]

            overall_stipa = calculate_stipa_core(sig, fs)
            results.append((file_name, overall_stipa))

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if results:
        plot_stipa_vs_rt60_multi_snr(results, output_dir)


if __name__ == "__main__":
    # החלף לנתיב שלך
    folder_with_recordings = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\reverb for check"

    run_minimal_analysis(folder_with_recordings)