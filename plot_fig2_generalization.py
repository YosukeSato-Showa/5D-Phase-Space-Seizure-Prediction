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
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.frameon': False
})

def plot_box_strip(data_file, title, output_name, color):
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Skipping {title}.")
        return
    df = pd.read_csv(data_file)
    
    fig, ax = plt.subplots(figsize=(4.5, 6), dpi=300)
    
    # Boxplot
    sns.boxplot(y='ROC_AUC', data=df, color=color, width=0.4, 
                boxprops={'alpha': 0.5, 'edgecolor': 'k', 'linewidth': 1.5},
                medianprops={'color': 'k', 'linewidth': 2}, ax=ax)
    
    # Strip plot for individual variance
    sns.stripplot(y='ROC_AUC', data=df, color=color, size=8, alpha=0.8, 
                  edgecolor='k', linewidth=1, jitter=True, ax=ax)
    
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    ax.set_ylim(0.25, 1.05)
    ax.set_ylabel('Prediction Accuracy (ROC-AUC)', fontweight='bold')
    ax.set_title(title, loc='left', fontweight='bold', pad=15)
    
    # Remove top and right spines
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}")

def plot_fig2b_kde(data_file, output_name):
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Skipping Fig 2b.")
        return
    df = pd.read_csv(data_file)
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    
    # Select top 4 patients for clear visualization of spatial fragmentation
    top_patients = df['Patient_ID'].value_counts().index[:4]
    df_subset = df[df['Patient_ID'].isin(top_patients)]
    
    palette = sns.color_palette("Set2", n_colors=len(top_patients))
    
    sns.scatterplot(data=df_subset, x='tau_SVM_ms', y='tau_TE_ms', hue='Patient_ID', 
                    palette=palette, alpha=0.6, edgecolor='k', s=40, ax=ax)
    sns.kdeplot(data=df_subset, x='tau_SVM_ms', y='tau_TE_ms', hue='Patient_ID', 
                palette=palette, fill=False, linewidths=1.5, alpha=0.8, ax=ax)
    
    ax.set_xlabel(r'Spatial Classification Time ($\tau_{SVM}$) [ms]', fontweight='bold')
    ax.set_ylabel(r'Information Transfer Delay ($\tau_{TE}$) [ms]', fontweight='bold')
    ax.set_title('b  Baseline state space fragmentation', loc='left', fontweight='bold', pad=15)
    
    sns.despine(top=True, right=True)
    plt.legend(title='Patient ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}")

def plot_learning_curve(data_file, output_name):
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Skipping Fig 2d.")
        return
    df = pd.read_csv(data_file)
    
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    
    ax.plot(df['Epoch'], df['Train AUC (Source Patients)'], label='Train (Source)', 
            color='#1F77B4', linewidth=2.5, marker='o', markersize=5)
    ax.plot(df['Epoch'], df['Test AUC (Unseen Patient)'], label='Test (Target)', 
            color='#D62728', linewidth=2.5, marker='s', markersize=5)
    
    ax.fill_between(df['Epoch'], df['Train AUC (Source Patients)'], df['Test AUC (Unseen Patient)'], 
                    color='gray', alpha=0.15, label='Generalization Gap')
    
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, zorder=0)
    ax.set_ylim(0.35, 1.05)
    ax.set_xticks(range(0, 21, 5))
    ax.set_xlabel('Training Epochs', fontweight='bold')
    ax.set_ylabel('ROC-AUC', fontweight='bold')
    ax.set_title('d  Overfitting to phenotypic idiosyncrasies', loc='left', fontweight='bold', pad=15)
    
    ax.legend(loc='center right')
    sns.despine(top=True, right=True)
    
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_name}")

if __name__ == "__main__":
    # Ensure consistent output file names matching the manuscript
    plot_box_strip('Data_Fig2a_Static_2D.csv', 'a  Static 2D Baseline', 'Nature_Fig2a_Static_2D.pdf', '#E24A33')
    plot_fig2b_kde('Data_Fig2b_Baseline_Tracking.csv', 'Nature_Fig2b_Fragmentation.pdf')
    plot_box_strip('Data_Fig2c_Unconditioned_5D.csv', 'c  Unconditioned 5D', 'Nature_Fig2c_Unconditioned_5D.pdf', '#348ABD')
    plot_learning_curve('Data_Fig2d_Overfitting.csv', 'Nature_Fig2d_Overfitting.pdf')
    plot_box_strip('Data_Fig2e_Conditioned_5D_7.5min.csv', 'e  Conditioned 5D (7.5 min)', 'Nature_Fig2e_Conditioned_7.5min.pdf', '#988ED5')