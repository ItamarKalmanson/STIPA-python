import subprocess
import sys

def get_stipa_from_library(clean_file, recorded_file):
    print("מחשב ציון STIPA (מדפיס ישירות לטרמינל)...")

    command = [
        sys.executable, "-m", "pyscreech.cli", "STIPA",
        recorded_file,
        "-r", clean_file,
        "-n", "1",
        "-d", "10",
        "--sync-waveforms",
        "--log-details"
    ]

    try:
        # הרצה ללא לכידת פלט - הכל יודפס ישירות למסך שלך!
        subprocess.run(command, check=True)
        print("\n=== החישוב הסתיים בהצלחה! ===")
        print("בדוק אם נוצר קובץ פלט (למשל CSV) בתיקייה של קבצי ה-WAV.")

    except subprocess.CalledProcessError as e:
        print(f"\n=== שגיאה מהספרייה (קוד {e.returncode}) ===")



if __name__ == "__main__":
    clean_audio = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\MASTER_SYNC_STIPA.wav"
    toy_audios = [r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part2\aligned\FULL_ALIGNED_6 מטר למטה.wav"
                  ,r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\רמקול בחדר קרוב בלי רעש.wav"
                  ,r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\רמקול בחדר רחוק בלי רעש.wav"
                  ,r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\רמקול בחדר של יובל.wav"
                  ,r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\רמקול מחוץ לשירותים על השידה.wav"
                  ,r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\שירותים קרוב.wav"
                  ,r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\part 3\שירותים רחוק .wav"]

for toy_audio in toy_audios:
    get_stipa_from_library(clean_audio, toy_audio)