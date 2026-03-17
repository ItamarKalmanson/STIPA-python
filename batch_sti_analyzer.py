import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import the main analysis function from your existing script
from sti_damage_estimator import analyze_sti


def parse_filename(filename):
    """
    Parses a filename like 'stipa_0.17s_ambient noise_SNR-10dB.wav'
    to extract theoretical RT60 and SNR values.
    """
    theoretical_rt60 = np.nan
    theoretical_snr = np.nan
    rt60_match = re.search(r'(\d+\.\d+)s', filename)
    if rt60_match:
        theoretical_rt60 = float(rt60_match.group(1))
    snr_match = re.search(r'SNR(-?\d+)', filename)
    if snr_match:
        theoretical_snr = float(snr_match.group(1))
    return theoretical_rt60, theoretical_snr


def run_batch_analysis(root_dir, ref_file):
    """
    Analyzes all .wav files in a directory, collects results,
    and generates summary plots for debugging.
    """
    degraded_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.wav') and 'ref' not in f.lower():
                degraded_files.append(os.path.join(dirpath, f))

    if not degraded_files:
        print(f"No .wav files found in '{root_dir}'. Exiting.")
        return

    print(f"Found {len(degraded_files)} files to analyze. Starting batch processing...")

    all_results = []
    for deg_file in tqdm(degraded_files, desc="Analyzing Files"):
        results = analyze_sti(ref_file, deg_file, align_the_signals=False, show_plots=False)
        if results:
            theo_rt60, theo_snr = parse_filename(results['filename'])
            results['theoretical_rt60'] = theo_rt60
            results['theoretical_snr'] = theo_snr
            all_results.append(results)

    df = pd.DataFrame(all_results)

    # --- 1. Print Summary Report ---
    print("\n" + "=" * 160)
    print("📊 Batch STI Analysis Summary Report 📊".center(160))
    print("=" * 160)
    display_cols = [
        'filename', 'sti_score', 
        'standard_noise_percent', 'simple_noise_percent',
        'standard_reverb_percent', 'simple_reverb_percent',
        'avg_snr', 'theoretical_snr'
    ]
    display_df = df[[col for col in display_cols if col in df.columns]].copy()
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)
    print(display_df.to_string(
        formatters={
            'sti_score': '{:,.3f}'.format,
            'standard_noise_percent': '{:,.1f}%'.format, 'simple_noise_percent': '{:,.1f}%'.format,
            'standard_reverb_percent': '{:,.1f}%'.format, 'simple_reverb_percent': '{:,.1f}%'.format,
            'avg_snr': '{:,.1f}'.format, 'theoretical_snr': '{:,.1f}'.format,
            'filename': lambda x: (x[:35] + '...') if len(x) > 38 else x
        },
        index=False
    ))
    print("=" * 160)

    # --- 2. Save Report to CSV ---
    report_filename = 'sti_batch_analysis_report.csv'
    try:
        df.to_csv(report_filename, index=False, float_format='%.3f')
        print(f"\n✅ Summary report saved to '{report_filename}'")
    except Exception as e:
        print(f"\n❌ Could not save report to CSV: {e}")

    # --- 3. Generate New Comparison Plots ---
    print("\nGenerating model comparison plots...")
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Prepare data for plotting ---
    df_perc = df.melt(id_vars=['theoretical_rt60', 'theoretical_snr'],
                      value_vars=['standard_noise_percent', 'simple_noise_percent', 'standard_reverb_percent', 'simple_reverb_percent'],
                      var_name='Metric', value_name='Percentage')
    df_perc[['Model', 'Damage Type', '_']] = df_perc['Metric'].str.split('_', expand=True)
    df_perc['Damage Type'] = df_perc['Damage Type'].str.capitalize()

    df_abs = df.melt(id_vars=['theoretical_rt60', 'theoretical_snr'],
                     value_vars=['standard_noise_loss', 'simple_noise_loss', 'standard_reverb_loss', 'simple_reverb_loss'],
                     var_name='Metric', value_name='Absolute Loss')
    df_abs[['Model', 'Damage Type', '_']] = df_abs['Metric'].str.split('_', expand=True)
    df_abs['Damage Type'] = df_abs['Damage Type'].str.capitalize()


    # --- Figure 1: Damage Percentage vs. RT60 (2x2 Layout) ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(20, 18), constrained_layout=True, sharey=True)
    fig1.suptitle('Model Comparison: Damage Percentage vs. RT60', fontsize=22)

    # Plot 1.1 (Top-Left): Standard Model, Noise %
    ax = axes1[0, 0]
    data = df_perc[(df_perc['Model'] == 'standard') & (df_perc['Damage Type'] == 'Noise')]
    sns.lineplot(data=data, x='theoretical_rt60', y='Percentage', hue='theoretical_snr', palette='coolwarm', marker='o', ax=ax)
    ax.set_title('Standard Model: Noise Damage', fontsize=16)
    ax.set_xlabel('Theoretical RT60 (s)', fontsize=12)
    ax.set_ylabel('Damage (%)', fontsize=12)
    ax.legend(title='SNR (dB)')

    # Plot 1.2 (Top-Right): Standard Model, Reverb %
    ax = axes1[0, 1]
    data = df_perc[(df_perc['Model'] == 'standard') & (df_perc['Damage Type'] == 'Reverb')]
    sns.lineplot(data=data, x='theoretical_rt60', y='Percentage', hue='theoretical_snr', palette='coolwarm', marker='o', ax=ax)
    ax.set_title('Standard Model: Reverb Damage', fontsize=16)
    ax.set_xlabel('Theoretical RT60 (s)', fontsize=12)
    ax.set_ylabel('') # Shared Y-axis
    ax.legend(title='SNR (dB)')

    # Plot 1.3 (Bottom-Left): Simplified Model, Noise %
    ax = axes1[1, 0]
    data = df_perc[(df_perc['Model'] == 'simple') & (df_perc['Damage Type'] == 'Noise')]
    sns.lineplot(data=data, x='theoretical_rt60', y='Percentage', hue='theoretical_snr', palette='coolwarm', marker='o', ax=ax)
    ax.set_title('Simplified Model: Noise Damage', fontsize=16)
    ax.set_xlabel('Theoretical RT60 (s)', fontsize=12)
    ax.set_ylabel('Damage (%)', fontsize=12)
    ax.legend(title='SNR (dB)')

    # Plot 1.4 (Bottom-Right): Simplified Model, Reverb %
    ax = axes1[1, 1]
    data = df_perc[(df_perc['Model'] == 'simple') & (df_perc['Damage Type'] == 'Reverb')]
    sns.lineplot(data=data, x='theoretical_rt60', y='Percentage', hue='theoretical_snr', palette='coolwarm', marker='o', ax=ax)
    ax.set_title('Simplified Model: Reverb Damage', fontsize=16)
    ax.set_xlabel('Theoretical RT60 (s)', fontsize=12)
    ax.set_ylabel('') # Shared Y-axis
    ax.legend(title='SNR (dB)')

    for ax in axes1.flat:
        ax.grid(True, which="both", ls="--")
        ax.set_ylim(-5, 105)


    # --- Figure 2: Absolute Damage vs. RT60 (2x2 Layout) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(20, 18), constrained_layout=True, sharey=True)
    fig2.suptitle('Model Comparison: Absolute STI Loss vs. RT60', fontsize=22)

    # Plot 2.1 (Top-Left): Standard Model, Noise Loss
    ax = axes2[0, 0]
    data = df_abs[(df_abs['Model'] == 'standard') & (df_abs['Damage Type'] == 'Noise')]
    sns.lineplot(data=data, x='theoretical_rt60', y='Absolute Loss', hue='theoretical_snr', palette='coolwarm', marker='o', ax=ax)
    ax.set_title('Standard Model: Noise Loss', fontsize=16)
    ax.set_xlabel('Theoretical RT60 (s)', fontsize=12)
    ax.set_ylabel('Absolute STI Loss (Points)', fontsize=12)
    ax.legend(title='SNR (dB)')

    # Plot 2.2 (Top-Right): Standard Model, Reverb Loss
    ax = axes2[0, 1]
    data = df_abs[(df_abs['Model'] == 'standard') & (df_abs['Damage Type'] == 'Reverb')]
    sns.lineplot(data=data, x='theoretical_rt60', y='Absolute Loss', hue='theoretical_snr', palette='coolwarm', marker='o', ax=ax)
    ax.set_title('Standard Model: Reverb Loss', fontsize=16)
    ax.set_xlabel('Theoretical RT60 (s)', fontsize=12)
    ax.set_ylabel('') # Shared Y-axis
    ax.legend(title='SNR (dB)')

    # Plot 2.3 (Bottom-Left): Simplified Model, Noise Loss
    ax = axes2[1, 0]
    data = df_abs[(df_abs['Model'] == 'simple') & (df_abs['Damage Type'] == 'Noise')]
    sns.lineplot(data=data, x='theoretical_rt60', y='Absolute Loss', hue='theoretical_snr', palette='coolwarm', marker='o', ax=ax)
    ax.set_title('Simplified Model: Noise Loss', fontsize=16)
    ax.set_xlabel('Theoretical RT60 (s)', fontsize=12)
    ax.set_ylabel('Absolute STI Loss (Points)', fontsize=12)
    ax.legend(title='SNR (dB)')

    # Plot 2.4 (Bottom-Right): Simplified Model, Reverb Loss
    ax = axes2[1, 1]
    data = df_abs[(df_abs['Model'] == 'simple') & (df_abs['Damage Type'] == 'Reverb')]
    sns.lineplot(data=data, x='theoretical_rt60', y='Absolute Loss', hue='theoretical_snr', palette='coolwarm', marker='o', ax=ax)
    ax.set_title('Simplified Model: Reverb Loss', fontsize=16)
    ax.set_xlabel('Theoretical RT60 (s)', fontsize=12)
    ax.set_ylabel('') # Shared Y-axis
    ax.legend(title='SNR (dB)')

    for ax in axes2.flat:
        ax.grid(True, which="both", ls="--")
        ax.set_ylim(bottom=-0.05)

    plt.show()


if __name__ == '__main__':
    recordings_folder = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\small dataset\dataset"
    reference_audio = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\STIPA ref.wav"

    if not os.path.isdir(recordings_folder):
        print(f"Error: Directory not found at '{recordings_folder}'")
    elif not os.path.exists(reference_audio):
        print(f"Error: Reference file not found at '{reference_audio}'")
    else:
        run_batch_analysis(recordings_folder, reference_audio)
