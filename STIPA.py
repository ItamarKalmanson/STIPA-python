import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate
import warnings

warnings.filterwarnings("ignore")


# --- פונקציות עזר קודמות ---
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def extract_envelope(signal_data, fs):
    abs_sig = np.abs(signal_data)
    b, a = butter(2, 50 / (0.5 * fs), btype='low')
    return filtfilt(b, a, abs_sig)


# --- פונקציות חדשות ומשודרגות ---

def calculate_sti_for_file(ref_aligned, deg_aligned, fs):
    # הפונקציה עכשיו מחזירה רק את הציון, בלי להדפיס כל תדר למסך כדי לא להציף
    octave_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    weights = [0.13, 0.14, 0.11, 0.11, 0.19, 0.17, 0.14]
    sti_score = 0

    for i, band in enumerate(octave_bands):
        lowcut = band / np.sqrt(2)
        highcut = band * np.sqrt(2)

        b, a = butter_bandpass(lowcut, highcut, fs)
        ref_filtered = filtfilt(b, a, ref_aligned)
        deg_filtered = filtfilt(b, a, deg_aligned)

        ref_env = extract_envelope(ref_filtered, fs)
        deg_env = extract_envelope(deg_filtered, fs)

        correlation = np.corrcoef(ref_env, deg_env)[0, 1]
        mtf = max(0, correlation)
        sti_score += mtf * weights[i]

    return sti_score


def analyze_batch(ref_file, deg_files_list):
    print(f"טוען קובץ רפרנס: {ref_file}...")
    ref_sig, fs_ref = sf.read(ref_file)
    if len(ref_sig.shape) > 1: ref_sig = ref_sig[:, 0]

    results = {}

    for deg_file in deg_files_list:
        print(f"\n--- מנתח את: {deg_file} ---")
        try:
            deg_sig, fs_deg = sf.read(deg_file)
            if len(deg_sig.shape) > 1: deg_sig = deg_sig[:, 0]

            if fs_ref != fs_deg:
                print(f"דילוג על {deg_file}: קצב דגימה לא תואם לרפרנס.")
                continue

            # יישור וגרף
            ref_aligned, deg_aligned = align_signals(ref_sig, deg_sig, fs_ref, deg_file)

            # חישוב ציון
            score = calculate_sti_for_file(ref_aligned, deg_aligned, fs_ref)
            results[deg_file] = score
            print(f"ציון חושב בהצלחה: {score:.2f}")

        except Exception as e:
            print(f"שגיאה בניתוח הקובץ {deg_file}: {e}")

    # הדפסת דוח סיכום
    print("\n" + "=" * 40)
    print(" 📊 דוח סיכום בדיקות STI 📊")
    print("=" * 40)
    print(f"{'שם הקובץ':<25} | {'ציון STI':<10} | {'הערכה'}")
    print("-" * 50)

    for file, score in results.items():
        if score >= 0.75:
            rating = "מצוינת"
        elif score >= 0.60:
            rating = "טובה"
        elif score >= 0.45:
            rating = "סבירה"
        else:
            rating = "חלשה"

        # הדפסה מיושרת (עשוי לדרוש התאמה קלה בהתאם לפונט בקונסול)
        print(f"{file:<25} | {score:<10.2f} | {rating}")
    print("=" * 50)


def align_signals(ref_sig, deg_sig, fs, filename):
    print("מחשב סנכרון זמנים (Cross-Correlation)...")

    # חישוב הסנכרון ויישור המערכים
    corr = correlate(deg_sig, ref_sig, mode='full', method='fft')
    lag = np.argmax(corr) - (len(ref_sig) - 1)

    if lag > 0:
        deg_aligned = deg_sig[lag:]
        ref_aligned = ref_sig[:len(deg_aligned)]
    elif lag < 0:
        ref_aligned = ref_sig[-lag:]
        deg_aligned = deg_sig[:len(ref_aligned)]
    else:
        ref_aligned = ref_sig
        deg_aligned = deg_sig

    min_len = min(len(ref_aligned), len(deg_aligned))
    ref_aligned = ref_aligned[:min_len]
    deg_aligned = deg_aligned[:min_len]

    delay_sec = abs(lag) / fs
    print(f"הקבצים יושרו! (היסט של {delay_sec:.3f} שניות תוקן)")

    # --- תיקון הגרף ---

    # 1. מציאת הנקודה שבה הסאונד האמיתי מתחיל (חיפוש קפיצה בעוצמה של 5% מהמקסימום)
    threshold = 0.05 * np.max(np.abs(ref_aligned))
    start_idx = np.argmax(np.abs(ref_aligned) > threshold)

    # 2. לקיחת חלון זמן של 0.1 שניות מרגע תחילת הסאונד (ולא מתחילת הקובץ)
    samples_to_plot = int(fs * 0.1)
    end_idx = min(start_idx + samples_to_plot, len(ref_aligned))

    ref_plot = ref_aligned[start_idx:end_idx]
    deg_plot = deg_aligned[start_idx:end_idx]

    # 3. נרמול עוצמות עבור התצוגה הויזואלית בלבד (מחלק במקסימום כדי להביא את שניהם לאזור ה-1.0)
    ref_plot_norm = ref_plot / (np.max(np.abs(ref_plot)) + 1e-9)
    deg_plot_norm = deg_plot / (np.max(np.abs(deg_plot)) + 1e-9)

    time_axis = np.linspace(0, len(ref_plot_norm) / fs, len(ref_plot_norm))

    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, ref_plot_norm, label="Reference (Clean) - Normalized", alpha=0.8, color='blue', linewidth=2)
    plt.plot(time_axis, deg_plot_norm, label="Degraded (Toy) - Normalized", alpha=0.7, color='orange', linestyle='--')
    plt.title(f"Alignment Match: {filename}\n(Showing 0.1s from actual audio start)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return ref_aligned, deg_aligned


# ==========================================
# אזור ההרצה
# ==========================================
if __name__ == "__main__":
    reference_audio = r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\STIPA ref.wav"

    # הכנס כאן את כל הקבצים שתרצה לבדוק
    toy_recordings = [
        r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\ליד מחשב.wav"
        ,r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\מחוץ לחדר.wav"
        ,r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\ממד בלי רעש.wav"
        ,r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\מעל הארון עם רעש.wav"
        ,r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\על המדף חלש עם רעש.wav"
        ,r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\על המדף עם רעש.wav"
    ]

    # הסר את ה-# כדי להריץ:
    analyze_batch(reference_audio, toy_recordings)



