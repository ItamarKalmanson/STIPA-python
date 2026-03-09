import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt


def extract_envelope(signal, fs, cutoff=30.0):
    """
    מחלץ את המעטפת של האות על ידי יישור (ערך מוחלט) וסינון Low-pass.
    """
    # 1. יישור האות (Rectification)
    rectified_signal = np.abs(signal)

    # 2. בניית פילטר Low-pass לתפיסת התנודות האיטיות של ה-STIPA
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype='low', analog=False)

    # 3. החלת הפילטר
    envelope = filtfilt(b, a, rectified_signal)
    return envelope


def plot_stipa_envelopes(ref_wav_path, test_wav_path):
    # טעינת קבצי השמע (החלף את הנתיבים לקבצים שלך)
    fs_ref, ref_signal = wavfile.read(ref_wav_path)
    fs_test, test_signal = wavfile.read(test_wav_path)

    # המרה לחד-ערוצי (Mono) במידה וההקלטות בסטריאו
    if len(ref_signal.shape) > 1:
        ref_signal = ref_signal[:, 0]
    if len(test_signal.shape) > 1:
        test_signal = test_signal[:, 0]

    # נרמול האותות כדי שנוכל להשוות אותם פרופורציונלית על ציר Y
    ref_signal = ref_signal / np.max(np.abs(ref_signal))
    test_signal = test_signal / np.max(np.abs(test_signal))

    # חילוץ המעטפות
    ref_env = extract_envelope(ref_signal, fs_ref)
    test_env = extract_envelope(test_signal, fs_test)

    # יצירת ציר זמן בשניות
    time_ref = np.arange(len(ref_signal)) / fs_ref
    time_test = np.arange(len(test_signal)) / fs_test

    # שרטוט הגרפים אחד תחת השני
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # גרף עליון - רפרנס
    ax1.plot(time_ref, ref_env, color='blue')
    ax1.set_title('אות הרפרנס (המעטפת המקורית והנקייה)')
    ax1.set_ylabel('אמפליטודה (מנורמלת)')
    ax1.grid(True)

    # גרף תחתון - בדיקה
    ax2.plot(time_test, test_env, color='red')
    ax2.set_title('אות הבדיקה המוקלט (המעטפת עם רעשי הרקע/דפיקות)')
    ax2.set_xlabel('זמן (שניות)')
    ax2.set_ylabel('אמפליטודה (מנורמלת)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# כאן תכניס את השמות של הקבצים שלך
plot_stipa_envelopes(r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\MASTER_SYNC_STIPA.wav", r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\no noise\STIPA_Results\aligned_audio\CLEAN_שירותים רחוק .wav")
plot_stipa_envelopes(r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\MASTER_SYNC_STIPA.wav", r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\no noise\STIPA_Results\aligned_audio\CLEAN_שירותים קרוב.wav")
plot_stipa_envelopes(r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\MASTER_SYNC_STIPA.wav", r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\no noise\STIPA_Results\aligned_audio\CLEAN_רמקול מחוץ לשירותים על השידה.wav")
plot_stipa_envelopes(r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\MASTER_SYNC_STIPA.wav", r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\no noise\STIPA_Results\aligned_audio\CLEAN_רמקול בחדר של יובל.wav")
plot_stipa_envelopes(r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\MASTER_SYNC_STIPA.wav", r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\no noise\STIPA_Results\aligned_audio\CLEAN_רמקול בחדר רחוק בלי רעש.wav")
plot_stipa_envelopes(r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\MASTER_SYNC_STIPA.wav", r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\no noise\STIPA_Results\aligned_audio\CLEAN_רמקול בחדר קרוב בלי רעש.wav")