#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
an excerpt from TPR-DB_Tables.ipynb
The orginial way of classifying gaze path
"""

## Label GazePath

def label_fixations(df, PUB, 
                    line_break_dx_threshold=730, 
                    max_stid_increment=6,
                    linear_dx=300, 
                    linear_dy=150,
                    refix_dx=26, 
                    refix_dy=150,
                    linear_chain=2, 
                    refix_chain=2,
                    max_refix_skip=2,
                    verbose=False):
    """
    Label fixations as Linear (L/l), Re-fixation (R/r), or Scattered (S/s).
    First fixation in a pattern gets uppercase, subsequent ones get lowercase.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Sequential fixation data with columns: X, Y, STid, TTid, Win, Time, End, PUB, TL (optional)
    line_break_dx_threshold : int
        Threshold for detecting line breaks (default: 730)
    max_stid_increment : int
        Max STid/TTid increment for line breaks (default: 6)
    linear_dx, linear_dy : int
        Max horizontal/vertical distance for linear movement (default: 300, 150)
    refix_dx, refix_dy : int
        Max distance for re-fixation (default: 26, 150)
    linear_chain : int
        Min consecutive pairs for linear classification (default: 2)
    refix_chain : int
        Min consecutive pairs for re-fixation classification (default: 2)
    max_refix_skip : int
        Max small regressions allowed in linear chain (default: 2)
    verbose : bool
        Print debug information (default: False)
    
    Returns:
    --------
    pd.DataFrame
        Input DataFrame with added 'GPlabel' column (L/l/R/r/S/s)
    """
    
    df['GPlabel'] = 'S'  # Default to scattered
    
    if len(df) < 2:
        return df
    
    # Determine reading direction (LTR=1, RTL=-1)
    rtl_langs = {'ar', 'he', 'fa', 'ur'}
    tl_code = str(df.iloc[0].get('TL', '')).lower()
    dir_fac = -1 if tl_code in rtl_langs else 1
    
    def log(msg):
        if verbose:
            print(msg)
    
    def is_line_break(f1, f2):
        """Check if saccade represents a line break."""
        dx = (f2.X - f1.X) * dir_fac
        
        # Check word ID progression
        if f1.Win == 1:
            id_ok = 1 <= (int(f2.STid) - int(f1.STid)) <= max_stid_increment
        elif f1.Win == 2:
            id_ok = 1 <= (int(f2.TTid) - int(f1.TTid)) <= max_stid_increment
        else:
            id_ok = False
            
        return dx < -line_break_dx_threshold and id_ok
    
    def is_linear_saccade(f1, f2):
        """Check if saccade is linear (forward movement)."""
        dx = (f2.X - f1.X) * dir_fac
        dy = f2.Y - f1.Y
        
        # Calculate word distance
        if f2.Win == 1:
            dist = int(f2.STid) - int(f1.STid)
        elif f2.Win == 2:
            dist = int(f2.TTid) - int(f1.TTid)
        else:
            dist = 0
        
        # Check criteria
        id_dist_ok = 1 <= dist < 5
        spatial_ok = refix_dx < dx <= linear_dx and abs(dy) <= linear_dy
        line_break = is_line_break(f1, f2)
        
        result = spatial_ok or line_break or id_dist_ok
        
        log(f"\t\tLinear:\tdst={dist}, spc:{spatial_ok} lb:{line_break} dx={dx:.0f}, dy={dy:.0f} \t→ {result}")
        
        return result
    
    def is_refixation_saccade(f1, f2):
        """Check if saccade is a re-fixation (small movement)."""
        dx = abs(f2.X - f1.X)
        dy = abs(f2.Y - f1.Y)
        
        # Calculate word distance
        if f2.Win == 1:
            dist = abs(int(f2.STid) - int(f1.STid))
        elif f2.Win == 2:
            dist = abs(int(f2.TTid) - int(f1.TTid))
        else:
            dist = 0
        
        # Check criteria
        spatial_ok = dx <= refix_dx and dy <= refix_dy
        id_ok = dist < 2
        
        result = spatial_ok or id_ok
        
        log(f"\t\tRefix:\tdst={dist}, spc:{spatial_ok} dx={dx:.0f}, dy={dy:.0f} \t→ {result}")
        
        return result
    
    def is_valid_transition(f1, f2):
        """Check if transition between fixations is valid (same window, no gap)."""
        same_window = f2.Win == f1.Win
        no_gap = (f2.Time - f1.End) <= PUB
        return same_window and no_gap
    
    def label_pattern(start_idx, end_idx, pattern_type):
        """Label a pattern with uppercase start and lowercase continuation."""
        indices = df.index.tolist()
        df.at[indices[start_idx], 'GPlabel'] = pattern_type.upper()
        for k in range(start_idx + 1, end_idx + 1):
            if k < len(indices):
                df.at[indices[k], 'GPlabel'] = pattern_type.lower()
    
    # Main processing loop
    indices = df.index.tolist()
    i = 0
    
    while i < len(indices) - 1:
            
        f1 = df.loc[indices[i]]
        f2 = df.loc[indices[i + 1]]

        if(verbose) :
            wordIds = ''
            if(f2.Win == 1) : wordIds = f"{f1.STid}->{f2.STid}" 
            if(f2.Win == 2) : wordIds = f"{f1.TTid}->{f2.TTid}" 
    
            log(f"\nFixation {i}, Word={wordIds} Win={f1.Win}/{f2.Win}, Time={f1.Time}/{f2.Time}")
        
        # Skip invalid transitions
        if not is_valid_transition(f1, f2):
            i += 1
            continue
        
        # Try to detect pattern start
        is_linear_start = is_linear_saccade(f1, f2)
        is_refix_start = is_refixation_saccade(f1, f2)
        
        if not (is_linear_start or is_refix_start):
            i += 1
            continue
        
        # Extend pattern
        linear_count = 1 if is_linear_start else 0
        refix_count = 1 if is_refix_start else 0
        linear_sub_count = linear_count
        refix_sub_count = refix_count
        scatter_skip_count = 0
        chain_end = i + 1
        
        j = i + 1
        f3 = f2
        while j < len(indices) - 1:
            f4 = df.loc[indices[j + 1]]

            if(f3.Win == 1) : wordIds = f"{f3.STid}->{f4.STid}" 
            if(f3.Win == 2) : wordIds = f"{f3.TTid}->{f4.TTid}" 
       
            log(f"\tExtending {j}: Word={wordIds} Time={f1.Time}/{f2.Time} linear={linear_count}, refix={refix_count} sub={refix_sub_count}")
            
            # Stop if invalid transition
            if not is_valid_transition(f3, f4):
                break
            
            # Check if pattern continues
            is_linear_count = is_linear_saccade(f3, f4)
            is_refix_count = is_refixation_saccade(f1, f4)
            
            stopChain = True

            if is_linear_count and refix_sub_count < refix_chain:
                stopChain = False 
                f3 = f4
                linear_count += 1
                linear_sub_count += 1
                chain_end = j + 1
                if linear_sub_count >= linear_chain:
                    scatter_skip_count = 0
                    refix_count = 0
                    refix_sub_count = 0
                    f1 = f4
                    
            elif is_refix_count : #and refix_count > 0 and refix_sub_count < refix_chain:
                stopChain = False 
                f3 = f4
                linear_sub_count = 0
                if(refix_count > 0) :refix_count += 1
                refix_sub_count += 1
                
            if(stopChain):
                scatter_skip_count += 1
                log(f"\t\tScter:\tskip={scatter_skip_count}")

                if scatter_skip_count > 1:
                    break
                    
            j += 1
        
        # Label if chain is long enough
        if linear_count >= linear_chain:
            log(f"\t→ Linear chain: {linear_count} fixations:{i} -> {chain_end}")
            label_pattern(i, chain_end, 'L')
            i = chain_end + 1
        elif refix_count >= refix_chain:
            log(f"\t→ Refixation chain: {refix_count} fixations: {i} -> {i+refix_count}")
            label_pattern(i, i + refix_count, 'R')
            i += refix_count + 1
        else:
            i += 1
    
    return df