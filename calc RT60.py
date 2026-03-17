import os
import glob
import shutil
import numpy as np
from scipy.io import wavfile


def calculate_rt60(filepath):
    """
    מחשב את ה-RT60 של קובץ IR באמצעות אינטגרציית שרדר.
    מבוסס על מדידת זמן הדעיכה מ--5dB ועד -65dB (הכפלה ב-3).
    """
    try:
        fs, signal = wavfile.read(filepath)

        # המרה למונו אם זה סטריאו
        if len(signal.shape) > 1:
            signal = signal[:, 0]

        # המרה ל-float ונרמול
        signal = signal.astype(np.float64)
        signal = signal / np.max(np.abs(signal))

        # מציאת נקודת ההתחלה האמיתית (הפיק החזק ביותר של ה-Impulse)
        peak_idx = np.argmax(np.abs(signal))
        h = signal[peak_idx:]  # חיתוך האות מנקודת השיא והלאה

        # אינטגרל שרדר (חישוב סכום מצטבר הפוך)
        # שימוש באנרגיה (האות בריבוע)
        energy = h ** 2
        edc = np.cumsum(energy[::-1])[::-1]

        # מניעת לוגריתם של אפס
        edc[edc == 0] = 1e-10

        # המרה לדציבלים
        edc_db = 10 * np.log10(edc / np.max(edc))

        # חיפוש האינדקסים של -5dB ו--25dB
        # משתמשים ב--5dB במקום 0dB כדי לדלג על הפגיעה הראשונית (Direct Sound)
        idx_minus_5 = np.argmin(np.abs(edc_db - (-5)))
        idx_minus_25 = np.argmin(np.abs(edc_db - (-65)))

        # חישוב הזמן שלקח לרדת 20 דציבלים
        dt = (idx_minus_25 - idx_minus_5) / fs

        # הכפלה ב-3 כדי לקבל את זמן הירידה של 60 דציבלים (RT60)
        rt60 = dt * 1
        return rt60

    except Exception as e:
        print(f"Error processing {os.path.basename(filepath)}: {e}")
        return None


def process_and_rename_irs(input_dir, output_dir):
    """
    סורק תיקייה של IRs, מחשב RT60, ומעתיק לתיקייה חדשה עם השם החדש.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    search_pattern = os.path.join(input_dir, '*.wav')
    ir_paths = glob.glob(search_pattern)

    if not ir_paths:
        print(f"No .wav files found in '{input_dir}'")
        return

    print(f"Found {len(ir_paths)} files. Starting RT60 analysis...\n")

    for file_path in ir_paths:
        original_name = os.path.basename(file_path)
        print(f"Analyzing: {original_name}...", end=" ")

        rt60 = calculate_rt60(file_path)

        if rt60 is not None:
            # עיצוב השם החדש, למשל: 1.45s.wav
            base_new_name = f"{rt60:.2f}s"
            new_filename = f"{base_new_name}.wav"
            new_filepath = os.path.join(output_dir, new_filename)

            # טיפול בכפילויות (אם שני חדרים יצאו עם אותו RT60 בדיוק)
            counter = 1
            while os.path.exists(new_filepath):
                new_filename = f"{base_new_name}_{counter}.wav"
                new_filepath = os.path.join(output_dir, new_filename)
                counter += 1

            # העתקת הקובץ לתיקייה החדשה
            shutil.copy2(file_path, new_filepath)
            print(f"Saved as -> {new_filename}")
        else:
            print("Failed.")

    print("\nDone! All files have been analyzed and renamed.")

# ==========================================
# דוגמה להפעלה:
# ==========================================

input_folder = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\reverb for check\מקורי"         # התיקייה שבה נמצאות כל ההקלטות המקוריות שלך
output_folder = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\reverb for check"  # התיקייה החדשה שאליה הקבצים יועתקו

process_and_rename_irs(input_folder, output_folder)