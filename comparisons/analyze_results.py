
import pandas as pd
import matplotlib.pyplot as plt
import os

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data():
    try:
        pas_prop = pd.read_csv(os.path.join(RESULTS_DIR, "pas_propofol.csv"))
        ana_prop = pd.read_csv(os.path.join(RESULTS_DIR, "anasim_propofol.csv"))
        pas_nore = pd.read_csv(os.path.join(RESULTS_DIR, "pas_norepinephrine.csv"))
        ana_nore = pd.read_csv(os.path.join(RESULTS_DIR, "anasim_norepinephrine.csv"))
        return pas_prop, ana_prop, pas_nore, ana_nore
    except FileNotFoundError as e:
        print(f"Error loading files from {RESULTS_DIR}: {e}")
        return None, None, None, None

def plot_propofol_comparison(pas_df, ana_df):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Cp
    axs[0].plot(pas_df["Time"]/60, pas_df["Cp"], label="PAS", linestyle="--")
    axs[0].plot(ana_df["Time"]/60, ana_df["Cp"], label="AnaSim", alpha=0.7)
    axs[0].set_ylabel("Propofol Cp (ug/mL)")
    axs[0].legend()
    axs[0].set_title("Propofol Comparison")
    
    # BIS
    axs[1].plot(pas_df["Time"]/60, pas_df["BIS"], label="PAS", linestyle="--")
    axs[1].plot(ana_df["Time"]/60, ana_df["BIS"], label="AnaSim", alpha=0.7)
    axs[1].set_ylabel("BIS")
    axs[1].legend()
    
    # MAP
    axs[2].plot(pas_df["Time"]/60, pas_df["MAP"], label="PAS", linestyle="--")
    axs[2].plot(ana_df["Time"]/60, ana_df["MAP"], label="AnaSim", alpha=0.7)
    axs[2].set_ylabel("MAP (mmHg)")
    axs[2].set_xlabel("Time (min)")
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "propofol_comparison.png"))
    print(f"Saved {os.path.join(RESULTS_DIR, 'propofol_comparison.png')}")
    plt.close()

def plot_norepi_comparison(pas_df, ana_df):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Cp
    axs[0].plot(pas_df["Time"]/60, pas_df["Cp"], label="PAS", linestyle="--")
    axs[0].plot(ana_df["Time"]/60, ana_df["Cp"], label="AnaSim", alpha=0.7)
    axs[0].set_ylabel("Norepinephrine Cp (ng/mL)")
    axs[0].legend()
    axs[0].set_title("Norepinephrine Comparison")
    
    # MAP
    axs[1].plot(pas_df["Time"]/60, pas_df["MAP"], label="PAS", linestyle="--")
    axs[1].plot(ana_df["Time"]/60, ana_df["MAP"], label="AnaSim", alpha=0.7)
    axs[1].set_ylabel("MAP (mmHg)")
    axs[1].set_xlabel("Time (min)")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "norepi_comparison.png"))
    print(f"Saved {os.path.join(RESULTS_DIR, 'norepi_comparison.png')}")
    plt.close()

def generate_report(pas_prop, ana_prop, pas_nore, ana_nore):
    md = "# AnaSim vs PAS Comparison Results\n\n"
    
    md += "## Propofol Scenario\n"
    md += f"- PAS Peak Cp: {pas_prop['Cp'].max():.2f} ug/mL\n"
    md += f"- AnaSim Peak Cp: {ana_prop['Cp'].max():.2f} ug/mL\n"
    md += f"- PAS Min BIS: {pas_prop['BIS'].min():.1f}\n"
    md += f"- AnaSim Min BIS: {ana_prop['BIS'].min():.1f}\n\n"
    
    md += "## Norepinephrine Scenario\n"
    md += f"- PAS Peak Cp: {pas_nore['Cp'].max():.2f} ng/mL\n"
    md += f"- AnaSim Peak Cp: {ana_nore['Cp'].max():.2f} ng/mL\n"
    md += f"- PAS Peak MAP: {pas_nore['MAP'].max():.1f} mmHg\n"
    md += f"- AnaSim Peak MAP: {ana_nore['MAP'].max():.1f} mmHg\n" # Corrected from res_norepi to ana_nore
    
    with open(os.path.join(RESULTS_DIR, "results.md"), "w") as f:
        f.write(md)
    print(f"Saved {os.path.join(RESULTS_DIR, 'results.md')}")

if __name__ == "__main__":
    pas_prop, ana_prop, pas_nore, ana_nore = load_data()
    if pas_prop is not None:
        plot_propofol_comparison(pas_prop, ana_prop)
        plot_norepi_comparison(pas_nore, ana_nore)
        generate_report(pas_prop, ana_prop, pas_nore, ana_nore)
