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
    # Default values if parsing fails
    theoretical_rt60 = np.nan
    theoretical_snr = np.nan

    # Extract RT60 (e.g., '0.17s')
    rt60_match = re.search(r'(\d+\.\d+)s', filename)
    if rt60_match:
        theoretical_rt60 = float(rt60_match.group(1))

    # Extract SNR (e.g., 'SNR-10dB' or 'SNR10dB')
    snr_match = re.search(r'SNR(-?\d+)', filename)
    if snr_match:
        theoretical_snr = float(snr_match.group(1))

    return theoretical_rt60, theoretical_snr


def run_batch_analysis(root_dir, ref_file):
    """
    Analyzes all .wav files in a directory, collects results,
    and generates summary plots for debugging.
    """
    # Find all .wav files to be analyzed
    degraded_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            # Ensure we only grab wav files and exclude the reference file itself
            if f.endswith('.wav') and 'ref' not in f.lower():
                degraded_files.append(os.path.join(dirpath, f))

    if not degraded_files:
        print(f"No .wav files found in '{root_dir}'. Exiting.")
        return

    print(f"Found {len(degraded_files)} files to analyze. Starting batch processing...")

    # Run analysis on each file and collect the results
    all_results = []
    for deg_file in tqdm(degraded_files, desc="Analyzing Files"):
        # For batch processing, we assume files are pre-aligned to save time.
        # Set show_plots=False to prevent a plot window for every single file.
        results = analyze_sti(ref_file, deg_file, align_the_signals=False, show_plots=False)

        # Parse filename for theoretical values and add them to the results
        theo_rt60, theo_snr = parse_filename(results['filename'])
        results['theoretical_rt60'] = theo_rt60
        results['theoretical_snr'] = theo_snr

        all_results.append(results)

    # Convert results to a pandas DataFrame for easier handling
    df = pd.DataFrame(all_results)

    # --- Create new columns for enhanced debugging ---
    df['snr_error'] = df['avg_snr'] - df['theoretical_snr']

    def get_primary_issue(row):
        noise = row['noise_damage_percent']
        reverb = row['reverb_damage_percent']
        if noise > 65: return "Noise"
        if reverb > 65: return "Reverb"
        if noise > 40 and reverb > 40: return "Mixed"
        if noise > reverb: return "Mainly Noise"
        return "Mainly Reverb"

    df['primary_issue'] = df.apply(get_primary_issue, axis=1)

    # --- 1. Print Summary Report ---
    print("\n" + "=" * 120)
    print("📊 Batch STI Analysis Summary Report 📊".center(120))
    print("=" * 120)

    display_cols = [
        'filename', 'sti_score', 'primary_issue', 'noise_damage_percent', 'reverb_damage_percent',
        'avg_snr', 'theoretical_snr', 'snr_error', 'theoretical_rt60'
    ]
    display_df = df[[col for col in display_cols if col in df.columns]].copy()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 150)
    print(display_df.to_string(
        formatters={
            'sti_score': '{:,.3f}'.format,
            'noise_damage_percent': '{:,.1f}%'.format,
            'reverb_damage_percent': '{:,.1f}%'.format,
            'avg_snr': '{:,.1f}'.format,
            'theoretical_snr': '{:,.1f}'.format,
            'snr_error': '{:,.1f}'.format,
            'theoretical_rt60': '{:,.2f}'.format,
            'filename': lambda x: (x[:45] + '...') if len(x) > 48 else x
        },
        index=False
    ))
    print("=" * 120)

    # --- 3. Save Report to CSV ---
    report_filename = 'sti_batch_analysis_report.csv'
    try:
        df.to_csv(report_filename, index=False, float_format='%.3f')
        print(f"\n✅ Summary report saved to '{report_filename}'")
    except Exception as e:
        print(f"\n❌ Could not save report to CSV: {e}")

    # --- 4. Generate Debugging Plots ---
    print("\nGenerating summary and debugging plots...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    fig.suptitle('STI Batch Analysis Dashboard', fontsize=22)

    # Plot 1: STI vs. Theoretical RT60
    ax1 = axes[0, 0]
    sns.scatterplot(data=df, x='theoretical_rt60', y='sti_score', hue='theoretical_snr', palette='coolwarm_r', ax=ax1,
                    s=70, legend='auto')
    ax1.set_title('1. STI Score vs. Theoretical RT60', fontsize=16)
    ax1.set_xlabel('Theoretical RT60 (s) from Filename', fontsize=12)
    ax1.set_ylabel('Measured STI Score', fontsize=12)
    ax1.grid(True, which="both", ls="--")

    # Plot 2: STI vs. Theoretical SNR
    ax2 = axes[0, 1]
    sns.scatterplot(data=df, x='theoretical_snr', y='sti_score', hue='theoretical_rt60', palette='viridis', ax=ax2,
                    s=70, legend='auto')
    ax2.set_title('2. STI Score vs. Theoretical SNR', fontsize=16)
    ax2.set_xlabel('Theoretical SNR (dB) from Filename', fontsize=12)
    ax2.set_ylabel('Measured STI Score', fontsize=12)
    ax2.grid(True, which="both", ls="--")

    # Plot 3: Damage Attribution Map
    ax3 = axes[1, 0]
    sns.scatterplot(data=df, x='noise_damage_percent', y='reverb_damage_percent', hue='sti_score', palette='inferno',
                    ax=ax3, s=70)
    ax3.set_title('3. Damage Attribution Map', fontsize=16)
    ax3.set_xlabel('Noise Damage (%)', fontsize=12)
    ax3.set_ylabel('Reverberation Damage (%)', fontsize=12)
    ax3.grid(True, which="both", ls="--")

    # Plot 4: Measured SNR vs. Theoretical SNR (DEBUG PLOT)
    ax4 = axes[1, 1]
    sns.scatterplot(data=df, x='theoretical_snr', y='avg_snr', hue='snr_error', palette='bwr', ax=ax4, s=70)
    lims = [
        min(df['theoretical_snr'].min() - 2, df['avg_snr'].min() - 2),
        max(df['theoretical_snr'].max() + 2, df['avg_snr'].max() + 2),
    ]
    ax4.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal Correlation (y=x)')
    ax4.set_title('4. DEBUG: Measured vs. Theoretical SNR', fontsize=16)
    ax4.set_xlabel('Theoretical SNR (dB) from Filename', fontsize=12)
    ax4.set_ylabel('Measured Average SNR (dB)', fontsize=12)
    ax4.legend()
    ax4.grid(True, which="both", ls="--")
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)

    # --- Figure 2: Detailed Reverb Damage Analysis ---
    fig2, ax5 = plt.subplots(figsize=(16, 10))
    fig2.suptitle('Detailed Analysis: Reverb Damage vs. Theoretical RT60', fontsize=18)

    sns.scatterplot(data=df, x='theoretical_rt60', y='reverb_damage_percent', hue='theoretical_snr',
                    style='primary_issue', palette='coolwarm_r', ax=ax5, s=80)
    ax5.set_xlabel('Theoretical RT60 (s) from Filename', fontsize=12)
    ax5.set_ylabel('Measured Reverb Damage (%)', fontsize=12)
    ax5.grid(True, which="both", ls="--")
    ax5.legend(title='Theoretical SNR (dB)')

    # Add STIPA score labels to each point for enhanced debugging
    for _, row in df.iterrows():
        if pd.notna(row['theoretical_rt60']) and pd.notna(row['reverb_damage_percent']):
            ax5.text(x=row['theoretical_rt60'],
                     y=row['reverb_damage_percent'] + 1.2,  # Small vertical offset
                     s=f"{row['sti_score']:.2f}",
                     fontdict={'size': 8, 'color': 'dimgray'},
                     ha='center')

    plt.show()


if __name__ == '__main__':
    # --- CONFIGURE BATCH ANALYSIS HERE ---

    # The folder containing all your degraded recordings.
    # This script will search recursively through all subfolders.
    recordings_folder = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\small dataset\dataset"

    # The clean, original STIPA signal
    reference_audio = r"C:\Users\itama\OneDrive\מסמכים\GitHub\STIPA-python\STIPA-recs\STIPA recs simulations\STIPA ref.wav"

    if not os.path.isdir(recordings_folder):
        print(f"Error: Directory not found at '{recordings_folder}'")
    elif not os.path.exists(reference_audio):
        print(f"Error: Reference file not found at '{reference_audio}'")
    else:
        run_batch_analysis(recordings_folder, reference_audio)