#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:39:57 2026

@author: liusiqi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:19:43 2026

@author: liusiqi
"""

"""
Step 1: Import Libraries and Load Data
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

# Load data - adjust separator and encoding as needed
# Common separators: '\t' (TSV), ',' (CSV), ';' (semicolon)
df = pd.read_csv('/Users/liusiqi/Downloads/tprdb-mothership-clone/PUBLIC/RUC17/Tables/P01_P5.fd', sep='\t')
df.columns = df.columns.str.strip() 

# Ensure correct time order
df = df.sort_values('Id').reset_index(drop=True)

print(f"Total fixations: {len(df)}")
print(f"Columns: {df.columns.tolist()}")



"""
Step 2: Define Window Size and Parameters
"""
# Window size in number of fixations (can be 5, 7, or 9)
WINDOW_SIZE = 7              # Can be 5, 7, or 9 (5 is recommended for translation data)
HALF_WIN = WINDOW_SIZE // 2   # 2 for 5, 3 for 7, 4 for 9
DOMINANT_THRESHOLD = (WINDOW_SIZE // 2) + 2   # 5→4, 7→5, 9→6



participants = df['StudySession'].unique()
n_participants = len(participants)
print(f"Number of participants: {n_participants}")
if n_participants > 1:
    print("   (Multi‑participant mode)")
else:
    print("   (Single‑participant mode)")


"""
Step 3: Helper functions
"""
def compute_baselines(df_region):
    if len(df_region) < 2:
        return None
    dur_median = df_region['Dur'].median()
    dur_mean = df_region['Dur'].mean()
    dur_std = df_region['Dur'].std()
    cv = dur_std / dur_mean if dur_mean > 0 else 0
    sacc_amps = []
    for i in range(len(df_region) - 1):
        if df_region.iloc[i+1]['Win'] == df_region.iloc[i]['Win']:
            dx = df_region.iloc[i+1]['X'] - df_region.iloc[i]['X']
            dy = df_region.iloc[i+1]['Y'] - df_region.iloc[i]['Y']
            sacc_amps.append(np.hypot(dx, dy))
    sacc_median = np.median(sacc_amps) if sacc_amps else None
    word_jumps = []
    for i in range(len(df_region) - 1):
        if df_region.iloc[i+1]['Win'] != df_region.iloc[i]['Win']:
            continue
        if df_region.iloc[i]['Win'] == 1:
            id_diff = df_region.iloc[i+1]['STid'] - df_region.iloc[i]['STid']
        else:
            id_diff = df_region.iloc[i+1]['TTid'] - df_region.iloc[i]['TTid']
        if id_diff > 0:
            word_jumps.append(id_diff)
    word_jump_median = np.median(word_jumps) if word_jumps else 1.0
    return {
        'BASE_DUR': dur_median,
        'BASE_CV': cv,
        'BASE_SACC': sacc_median,
        'BASE_WORD_JUMP': word_jump_median
    }


def extract_window_features(window_df, region):
    if len(window_df) < 4:
        return None
    window_df = window_df.sort_values('Id').reset_index(drop=True)
    n_saccades = len(window_df) - 1
    win_dur = window_df['Dur'].median()
    dur_mean = window_df['Dur'].mean()
    dur_std = window_df['Dur'].std()
    win_cv = dur_std / dur_mean if dur_mean > 0 else 0
    sacc_amps = []
    for i in range(len(window_df) - 1):
        dx = window_df.iloc[i+1]['X'] - window_df.iloc[i]['X']
        dy = window_df.iloc[i+1]['Y'] - window_df.iloc[i]['Y']
        sacc_amps.append(np.hypot(dx, dy))
    win_sacc = np.median(sacc_amps) if sacc_amps else 0
    word_net = 0
    for i in range(n_saccades):
        f1 = window_df.iloc[i]
        f2 = window_df.iloc[i+1]
        if region == 1:
            id_diff = f2['STid'] - f1['STid']
        else:
            id_diff = f2['TTid'] - f1['TTid']
        word_net += id_diff
    regcount = 0
    for i in range(n_saccades):
        f1 = window_df.iloc[i]
        f2 = window_df.iloc[i+1]
        if region == 1:
            id_diff = f2['STid'] - f1['STid']
        else:
            id_diff = f2['TTid'] - f1['TTid']
        dx = f2['X'] - f1['X']
        dy = f2['Y'] - f1['Y']
        amp = np.hypot(dx, dy)
        if id_diff < 0 and amp > 26:
            regcount += 1
    return {
        'WIN_DUR': win_dur,
        'WIN_CV': win_cv,
        'WIN_SACC': win_sacc,
        'WIN_WORD_NET': word_net,
        'REGCOUNT': regcount,
        'N_SACCADES': n_saccades
    }


def apply_decision_tree(features, baselines, region):
    if features is None or baselines is None:
        return 'O'
    region_name = 'ST' if region == 1 else 'TT'
    BASE_DUR = baselines.get('BASE_DUR', 200)
    BASE_SACC = baselines.get('BASE_SACC', 80)
    BASE_WORD_JUMP = baselines.get('BASE_WORD_JUMP', 1.0)
    BASE_CV = baselines.get('BASE_CV', 0.3)
    n = features.get('N_SACCADES', 4)
    if (features['WIN_DUR'] < 0.75 * BASE_DUR and
        features['WIN_SACC'] > 1.5 * BASE_SACC and
        features['WIN_WORD_NET'] > 0.8 * n * BASE_WORD_JUMP):
        return 'S'
    if (features['REGCOUNT'] >= 2 and
        features['WIN_WORD_NET'] < 0.6 * n * BASE_WORD_JUMP):
        return 'R'
    if (features['WIN_WORD_NET'] < 0.3 * n * BASE_WORD_JUMP and
        features['WIN_SACC'] < BASE_SACC):
        return 'R'
    if (features['WIN_WORD_NET'] > 0.3 * n * BASE_WORD_JUMP and
        features['WIN_CV'] < 1.3 * BASE_CV):
        return 'L'
    return 'O'


def merge_labels(group):
    group = group.sort_values('Id').reset_index(drop=True)
    group['GP_final'] = ''
    if len(group) == 0:
        return group
    current_label = None
    current_start_idx = 0
    for i in range(len(group)):
        label = group.loc[i, 'GPlabel_N']
        if label == '?' or label == 'X':
            if current_label is not None:
                for j in range(current_start_idx, i):
                    if j == current_start_idx:
                        group.loc[j, 'GP_final'] = current_label.upper()
                    else:
                        group.loc[j, 'GP_final'] = current_label.lower()
                current_label = None
            group.loc[i, 'GP_final'] = label
            continue
        if current_label is None:
            current_label = label
            current_start_idx = i
        elif label != current_label:
            for j in range(current_start_idx, i):
                if j == current_start_idx:
                    group.loc[j, 'GP_final'] = current_label.upper()
                else:
                    group.loc[j, 'GP_final'] = current_label.lower()
            current_label = label
            current_start_idx = i
    if current_label is not None:
        for j in range(current_start_idx, len(group)):
            if j == current_start_idx:
                group.loc[j, 'GP_final'] = current_label.upper()
            else:
                group.loc[j, 'GP_final'] = current_label.lower()
    group['GP_final'] = group['GP_final'].replace('', '?')
    return group


def process_participant(df_p):
    # IFI pre‑segmentation
    df_p = df_p.sort_values('Id').reset_index(drop=True)
    df_p['IFI'] = df_p['Time'] - df_p['End'].shift(1)
    df_p['IFI'] = df_p['IFI'].clip(lower=0)
    threshold_ifi = df_p['IFI'].dropna().quantile(0.99)
    print(f"  IFI threshold: {threshold_ifi:.2f} ms")
    df_p['chunk_id'] = 0
    current_chunk = 0
    for i in range(1, len(df_p)):
        if df_p.loc[i, 'IFI'] > threshold_ifi:
            current_chunk += 1
        df_p.loc[i, 'chunk_id'] = current_chunk

    # Baselines per region
    baselines = {}
    for region, region_name in [(1, 'ST'), (2, 'TT')]:
        region_data = df_p[df_p['Win'] == region]
        baselines[region_name] = compute_baselines(region_data) if len(region_data) >= 2 else None
    
    # ---- Print and save baselines ----
    print("\n=== Baselines for this participant ===")
    for region_name, b in baselines.items():
        if b is not None:
            print(f"  {region_name}:")
            for k, v in b.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {region_name}: insufficient data")

    # Save to file (overwrites each run)
    with open('baselines_summary.txt', 'w') as f:
        for region_name, b in baselines.items():
            if b is not None:
                f.write(f"{region_name}: {b}\n")
            else:
                f.write(f"{region_name}: insufficient data\n")
    # Sliding window classification
    df_p['GPlabel_N'] = '?'
    total_centers = 0
    mixed_count = 0
    no_baseline_count = 0
    no_features_count = 0

    for chunk_id in df_p['chunk_id'].unique():
        chunk_data = df_p[df_p['chunk_id'] == chunk_id]
        if len(chunk_data) < WINDOW_SIZE:
            continue
        for center_idx in range(HALF_WIN, len(chunk_data) - HALF_WIN):
            total_centers += 1
            window = chunk_data.iloc[center_idx - HALF_WIN : center_idx + HALF_WIN + 1]

            # Region dominance check using dynamic threshold
            region_counts = window['Win'].value_counts()
            if len(region_counts) == 1:
                region = region_counts.index[0]
                filtered_window = window
            else:
                max_count = region_counts.max()
                if max_count >= DOMINANT_THRESHOLD:
                    region = region_counts.idxmax()
                    filtered_window = window[window['Win'] == region].copy()
                else:
                    center_row_idx = window.index[HALF_WIN]
                    df_p.loc[center_row_idx, 'GPlabel_N'] = 'X'
                    mixed_count += 1
                    continue

            region_name = 'ST' if region == 1 else 'TT'
            baselines_region = baselines.get(region_name)
            if baselines_region is None:
                no_baseline_count += 1
                continue
            features = extract_window_features(filtered_window, region)
            if features is None:
                no_features_count += 1
                continue
            label = apply_decision_tree(features, baselines_region, region)
            center_row_idx = window.index[HALF_WIN]
            df_p.loc[center_row_idx, 'GPlabel_N'] = label

    print(f"  Total center candidates: {total_centers}")
    print(f"    - Mixed region (skipped as 'X'): {mixed_count} ({mixed_count/total_centers*100:.1f}%)")
    print(f"    - No baseline: {no_baseline_count} ({no_baseline_count/total_centers*100:.1f}%)")
    print(f"    - Feature extraction failed: {no_features_count} ({no_features_count/total_centers*100:.1f}%)")
    print(f"    - Successfully labelled: {total_centers - mixed_count - no_baseline_count - no_features_count} ({ (total_centers - mixed_count - no_baseline_count - no_features_count)/total_centers*100:.1f}%)")

    df_p = merge_labels(df_p)
    return df_p

# ============================================================================
# STEP 4: Apply processing to all participants
# ============================================================================

if n_participants == 1:
    print("Processing single participant...")
    df_final = process_participant(df)
else:
    print("Processing multiple participants...")
    all_results = []
    for i, pid in enumerate(participants, 1):
        print(f"  [{i}/{n_participants}] {pid}")
        df_p = df[df['StudySession'] == pid].copy()
        df_p = process_participant(df_p)
        all_results.append(df_p)
    df_final = pd.concat(all_results, ignore_index=True)


# ============================================================================
# STEP 5: Summary and save
# ============================================================================

print("\n" + "="*50)
print("Final label distribution (after merging, including 'X'):")
print(df_final['GP_final'].value_counts())

print("\nSample of labelled fixations:")
print(df_final[['Id', 'StudySession', 'GPlabel_N', 'GP_final', 'chunk_id']].head(20))

df_final.to_csv('labeled_gaze_path2_Win7_T1.csv', index=False)
print("\nResults saved to 'labeled_gaze_path.csv'.")


