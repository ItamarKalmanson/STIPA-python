import subprocess


def get_stipa_from_library(clean_file, recorded_file):
    print("מחשב ציון STIPA (לוכד את כל הפלט והלוגים)...")

    command = [
        "python", "-m", "pyscreech.cli", "STIPA",
        recorded_file,
        "-r", clean_file,
        "-n", "1",
        "-d", "20",
        "--sync-waveforms",
        "--log-details"
    ]

    try:
        # השינוי כאן: איחוד ערוץ הלוגים לתוך ערוץ ההדפסה
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                check=True)
        print("\n=== תוצאות הספרייה ===")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("\n=== שגיאה מהספרייה ===")
        print(e.stdout)


if __name__ == "__main__":
    clean_audio = r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\STIPA ref.wav"
    toy_audio = r"C:\Users\itama\OneDrive\שולחן העבודה\STIPA\מחוץ לחדר.wav"

    get_stipa_from_library(clean_audio, toy_audio)