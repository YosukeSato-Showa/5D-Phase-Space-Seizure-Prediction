import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- 1. Style & Configuration for Nature Communications ---
plt.style.use('seaborn-v0_8-white')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'axes.linewidth': 1.5,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'legend.frameon': False
})

def plot_time_ablation(data_file, output_name):
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        sys.exit(1)
        
    df = pd.read_csv(data_file)
    
    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=300)
    
    # Define Nature-friendly colors
    color_auc = '#D62728'  # Deep Red
    color_patients = '#1F77B4'  # Deep Blue
    
    # Secondary Axis: Number of Patients Bar Plot (Plotted first to stay in background)
    ax2 = ax1.twinx()
    bars = ax2.bar(df['Threshold_Min'], df['Patients'], width=0.6, 
                   color=color_patients, alpha=0.25, edgecolor='none', label='Retained Patients')
    ax2.set_ylabel('Number of Unseen Patients', color=color_patients, fontweight='bold')
    ax2.set_ylim(0, 16)
    ax2.tick_params(axis='y', labelcolor=color_patients)
    
    # Primary Axis: ROC-AUC Line Plot
    line = ax1.plot(df['Threshold_Min'], df['Mean_AUC'], color=color_auc, marker='o', 
                    markersize=9, linewidth=3, label='Mean ROC-AUC')
    
    # Shaded region above chance level
    ax1.fill_between(df['Threshold_Min'], 0.5, df['Mean_AUC'], color=color_auc, alpha=0.1)
    
    # Chance level reference line
    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    
    # Golden Window / Phase Transition Marker (3.0 min)
    ax1.axvline(3.0, color='k', linestyle=':', linewidth=2)
    ax1.text(3.3, 0.55, 'Deterministic Phase Transition\n(3.0 min)', color='k', 
             fontsize=12, ha='left', va='bottom', fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    
    # Format Primary Axis
    ax1.set_xlim(15.5, 0.5)  # Reversed X-axis (Approaching clinical onset)
    ax1.set_ylim(0.45, 1.05)
    ax1.set_xlabel('Preictal Threshold (Minutes prior to seizure onset)', fontweight='bold')
    ax1.set_ylabel('Prediction Accuracy (ROC-AUC)', color=color_auc, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_auc)
    
    ax1.set_title('Time-to-seizure ablation isolates a 3-minute deterministic phase transition', 
                  loc='left', fontweight='bold', pad=20)
    
    # Remove unnecessary spines for clean look
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['left'].set_color(color_auc)
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['right'].set_color(color_patients)
    
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}")

if __name__ == "__main__":
    plot_time_ablation('Data_Fig3_Time_Ablation.csv', 'Nature_Fig3_Time_Ablation_Trajectory.pdf')