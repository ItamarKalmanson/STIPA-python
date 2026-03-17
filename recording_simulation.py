import os
import glob
import numpy as np
import math
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly


def load_and_resample(file_path, target_fs=44100):
    """טעינת קובץ, המרה למונו, נרמול ו-Resampling במידת הצורך"""
    fs, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = data[:, 0]

    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    else:
        data = data.astype(np.float64)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

    if fs != target_fs:
        gcd = math.gcd(fs, target_fs)
        up = target_fs // gcd
        down = fs // gcd
        data = resample_poly(data, up, down)

    return data


def calculate_power(signal):
    """חישוב העוצמה של האות"""
    return np.mean(signal ** 2)


def generate_stipa_dataset_from_folder(clean_path, rir_dir, noise_path, snr_levels, output_dir="stipa_dataset",
                                       target_fs=44100):
    """
    סורק תיקיית הדהודים (RIRs) ומייצר אוטומטית סט נתונים של בדיקות STIPA.
    """
    # 1. מציאת כל קבצי ה-WAV בתיקיית ה-RIR בעזרת glob
    search_pattern = os.path.join(rir_dir, '*.wav')
    rir_paths = glob.glob(search_pattern)

    if not rir_paths:
        print(f"Error: No .wav files found in directory '{rir_dir}'")
        return

    print(f"Found {len(rir_paths)} RIR files in '{rir_dir}'.\n")

    # יצירת תיקיית פלט אם היא לא קיימת
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print("Loading clean STIPA signal and noise file...")
    clean_signal = load_and_resample(clean_path, target_fs)
    noise_signal = load_and_resample(noise_path, target_fs)

    # 2. ריצה על כל קובץ הדהוד שנמצא בתיקייה
    for rir_path in rir_paths:
        rir_name = os.path.splitext(os.path.basename(rir_path))[0]
        print(f"\n--- Processing RIR: {rir_name} ---")

        rir_signal = load_and_resample(rir_path, target_fs)

        # קונבולוציה פעם אחת לכל חדר
        reverberant_signal = fftconvolve(clean_signal, rir_signal, mode='full')
        reverberant_signal = reverberant_signal[:len(clean_signal)]
        sig_power = calculate_power(reverberant_signal)

        # התאמת אורך הרעש
        current_noise = np.copy(noise_signal)
        if len(current_noise) < len(reverberant_signal):
            repeats = int(np.ceil(len(reverberant_signal) / len(current_noise)))
            current_noise = np.tile(current_noise, repeats)
        current_noise = current_noise[:len(reverberant_signal)]

        noise_power = calculate_power(current_noise)
        if noise_power == 0:
            noise_power = 1e-10

        noise_name = os.path.splitext(os.path.basename(noise_path))[0]

        # 3. ייצור הקבצים לפי רמות ה-SNR
        for snr in snr_levels:
            target_noise_power = sig_power / (10 ** (snr / 10))
            noise_factor = np.sqrt(target_noise_power / noise_power)
            scaled_noise = current_noise * noise_factor

            final_signal = reverberant_signal + scaled_noise
            max_val = np.max(np.abs(final_signal))
            if max_val > 0:
                final_signal = final_signal / max_val

            final_signal_16bit = np.int16(final_signal * 32767)
            out_filename = f"stipa_{rir_name}_{noise_name}_SNR{snr}dB.wav"
            out_filepath = os.path.join(output_dir, out_filename)

            wavfile.write(out_filepath, target_fs, final_signal_16bit)
            print(f"  -> Saved: {out_filename}")

    print("\nDataset generation complete!")


# ==========================================
# דוגמה לאופן ההפעלה של הקוד:
# ==========================================

# 2. הגדרת רמות SNR מבוקשות (בדציבלים)
my_snrs = [-10, 0, 10]
my_rir_folder = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\reverb for check"
noise_file = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\ambient noise.wav"
clean_file = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\STIPA ref.wav"

# 3. הרצת הפונקציה
generate_stipa_dataset_from_folder(clean_file, my_rir_folder, noise_file, my_snrs, output_dir = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\reverb for check")