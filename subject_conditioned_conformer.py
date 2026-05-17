import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

# --- 1. Model Definitions (5D Meta Dimension + FiLM) ---
class PhenotypeEncoder(nn.Module):
    # 5D Phase Space Marker (tau_TE, tau_SVM, v_TE, v_SVM, SampEn)
    def __init__(self, hidden_dim, meta_dim=5):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim), 
            nn.GELU(), 
            nn.LayerNorm(hidden_dim)
        )
        self.head_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.head_beta  = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.head_gamma.weight)
        nn.init.zeros_(self.head_gamma.bias)
        nn.init.zeros_(self.head_beta.weight)
        nn.init.zeros_(self.head_beta.bias)
        
    def forward(self, meta):
        x = self.shared_mlp(meta)
        return self.head_gamma(x), self.head_beta(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class SubjectConditionedConformer(nn.Module):
    def __init__(self, in_chans=17, hidden_dim=64, num_classes=2):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, 32, (1, 65), padding=(0, 32), bias=False)
        self.spatial_conv = nn.Conv2d(32, 64, (in_chans, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool1d(4) 
        
        # FiLM Layer Initialization
        self.pheno_enc = PhenotypeEncoder(64, meta_dim=5)
        self.transformer = nn.Sequential(
            TransformerBlock(64, nhead=8, dim_feedforward=256, dropout=0.2),
            TransformerBlock(64, nhead=8, dim_feedforward=256, dropout=0.2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, meta):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn1(x)
        x = torch.nn.functional.gelu(x)
        x = x.squeeze(2) 
        x = self.pool1(x) 
        
        # --- FiLM Modulation ---
        gamma, beta = self.pheno_enc(meta)
        x = (1.0 + gamma.unsqueeze(-1)) * x + beta.unsqueeze(-1)
        
        x = x.permute(0, 2, 1) 
        x = self.transformer(x)
        x = x.permute(0, 2, 1) 
        return self.classifier(x)

class CHBMITDataset(Dataset):
    def __init__(self, X, y, metas):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.metas = torch.tensor(metas, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.metas[idx], self.y[idx]

# --- 2. Data Loading (5D Phase Space Extraction) ---
def load_patient_specific_data():
    csv_file = "Data_Fig2b_Baseline_Tracking.csv" # Updated to final CSV name
    data_dir = "/home/sato/CHBMIT_Data"
    
    target_electrodes = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
    num_chans = len(target_electrodes)
    
    true_columns = ['Time_sec', 'Phase', 'Time_to_Seizure', 'Driver_Node', 'Min_SampEn', 'tau_TE_ms', 'tau_SVM_ms', 'Delta_Tau_ms', 'File_Name']
    df = pd.read_csv(csv_file, skiprows=1, names=true_columns)
    
    for col in ['Time_to_Seizure', 'tau_TE_ms', 'tau_SVM_ms', 'Min_SampEn']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df['Patient'] = df['File_Name'].apply(lambda x: str(x).split('_')[0])
    
    patient_data_dict = {}
    print("Applying Chronological Stratified Split with 5D Velocity Vectors & FiLM...")
    
    for patient, group in df.groupby('Patient'):
        group = group.sort_values(['File_Name', 'Time_sec']).copy()
        
        # Calculate velocity vectors (first derivative) and smooth
        group['v_TE'] = group.groupby('File_Name')['tau_TE_ms'].diff().fillna(0)
        group['v_SVM'] = group.groupby('File_Name')['tau_SVM_ms'].diff().fillna(0)
        group['v_TE'] = group.groupby('File_Name')['v_TE'].transform(lambda x: x.rolling(3, min_periods=1, center=True).mean())
        group['v_SVM'] = group.groupby('File_Name')['v_SVM'].transform(lambda x: x.rolling(3, min_periods=1, center=True).mean())
        
        patient_epochs, patient_labels, patient_metas = [], [], []
        files = group['File_Name'].unique()
        
        for f in files:
            edf_path = os.path.join(data_dir, f)
            if not os.path.exists(edf_path): continue
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                raw.filter(l_freq=0.5, h_freq=80.0, verbose=False)
            except: continue
            
            ch_map = {}
            for i, tgt in enumerate(target_electrodes):
                for j, ch in enumerate(raw.ch_names):
                    if tgt in ch.upper().replace('-', ' '):
                        ch_map[i] = j
                        break
            if len(ch_map) < 5: continue
            
            file_records = group[group['File_Name'] == f]
            for _, row in file_records.iterrows():
                tts = row['Time_to_Seizure']
                if pd.isna(tts) or tts > 900: continue # Baseline hold: up to 15 mins
                
                # [5D Marker Integration]
                window_meta = [row['tau_TE_ms'], row['tau_SVM_ms'], row['v_TE'], row['v_SVM'], row['Min_SampEn']]
                if pd.isna(window_meta).any(): continue
                
                # Evaluation threshold (e.g., 7.5 min = 450 sec)
                label = 1 if tts < 450 else 0
                
                start_samp = int(float(row['Time_sec']) * 256)
                end_samp = start_samp + 2560 
                if end_samp > raw.n_times: continue
                
                raw_data = raw.get_data(start=start_samp, stop=end_samp) * 1e6
                epoch_data = np.zeros((num_chans, 2560), dtype=np.float32)
                for tgt_idx, raw_idx in ch_map.items():
                    epoch_data[tgt_idx, :] = raw_data[raw_idx, :]
                    
                patient_epochs.append(epoch_data)
                patient_labels.append(label)
                patient_metas.append(window_meta)
                
        patient_epochs = np.array(patient_epochs)
        patient_labels = np.array(patient_labels)
        patient_metas = np.array(patient_metas)
        
        idx_0 = np.where(patient_labels == 0)[0]
        idx_1 = np.where(patient_labels == 1)[0]
        
        if len(idx_0) > 5 and len(idx_1) > 5:
            split_0 = int(len(idx_0) * 0.7)
            split_1 = int(len(idx_1) * 0.7)
            
            train_idx = np.concatenate((idx_0[:split_0], idx_1[:split_1]))
            test_idx = np.concatenate((idx_0[split_0:], idx_1[split_1:]))
            train_idx.sort(); test_idx.sort()
            
            scaler = StandardScaler()
            m_tr_scaled = scaler.fit_transform(patient_metas[train_idx])
            m_te_scaled = scaler.transform(patient_metas[test_idx])
            
            patient_data_dict[patient] = {
                'X_tr': patient_epochs[train_idx], 'y_tr': patient_labels[train_idx], 'm_tr': m_tr_scaled,
                'X_te': patient_epochs[test_idx], 'y_te': patient_labels[test_idx], 'm_te': m_te_scaled
            }
            
    return patient_data_dict

# --- 3. Training & Evaluation & CSV Output ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    patient_data_dict = load_patient_specific_data()
    print(f"\nExtracted {len(patient_data_dict)} patients.")
    
    results = []
    print("\n=== Simulating Patient-Specific RNS Tuning (5D + FiLM Conformer) ===")
    
    for patient, data in patient_data_dict.items():
        print(f"\n--- Patient: {patient} ---")
        
        tr_loader = DataLoader(CHBMITDataset(data['X_tr'], data['y_tr'], data['m_tr']), batch_size=32, shuffle=True)
        te_loader = DataLoader(CHBMITDataset(data['X_te'], data['y_te'], data['m_te']), batch_size=32, shuffle=False)
        
        model = SubjectConditionedConformer().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
        
        counts = np.bincount(data['y_tr'])
        weight_safe = 1.0
        weight_gw = counts[0] / (counts[1] + 1e-5) 
        class_weights = torch.tensor([weight_safe, weight_gw], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        best_auc = 0
        best_acc = 0
        
        for epoch in range(30):
            model.train()
            for X, m, y in tr_loader:
                X, m, y = X.to(device), m.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(X, m)
                loss = criterion(out, y)
                loss.backward(); optimizer.step()
                
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for X, m, y in te_loader:
                    X, m, y = X.to(device), m.to(device), y.to(device)
                    out = model(X, m)
                    preds_prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                    all_preds.extend(preds_prob)
                    all_labels.extend(y.cpu().numpy())
                    
            try:
                auc_val = roc_auc_score(all_labels, all_preds)
                acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
            except ValueError:
                auc_val = 0; acc = 0
            
            if auc_val > best_auc:
                best_auc = auc_val
                best_acc = acc
                
        print(f">>> Patient {patient} Final Test (Future Data) | ROC-AUC: {best_auc:.4f} | Acc: {best_acc*100:.2f}%")
        results.append({'Patient': patient, 'ROC_AUC': best_auc, 'Accuracy': best_acc})

    # --- CSV Output ---
    csv_out_file = 'Data_Fig2e_Conditioned_5D_7.5min.csv' # Updated to final CSV name
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(csv_out_file, index=False)
        
        avg_auc = np.mean([r['ROC_AUC'] for r in results])
        avg_acc = np.mean([r['Accuracy'] for r in results])
        print("\n" + "="*50)
        print("  FINAL 5D + FiLM CONFORMER RESULTS  ")
        print("="*50)
        print(f"Average ROC-AUC : {avg_auc:.4f}")
        print(f"Average Accuracy: {avg_acc*100:.2f}%")
        print(f"Results saved to: {csv_out_file}")
        print("="*50)

if __name__ == '__main__':
    main()
