import numpy as np

def calculate_spectral_entropy(mzs, ints, max_peaks=64, int_floor=0.01, power=0.5, prec_mz=None, mask_precursor=True):
    """
    Cleans peaks and computes Shannon spectral entropy S(P) = -sum(w * ln(w)).
    Returns cleaned mzs, weights (sum=1), and entropy.
    """
    if len(mzs) == 0:
        return np.array([]), np.array([]), 0.0
    
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    
    # Precursor masking if requested (|m/z - prec_mz| > 1.5 Da)
    if mask_precursor and prec_mz is not None:
        mask = np.abs(mzs - prec_mz) > 1.5
        mzs = mzs[mask]
        ints = ints[mask]
        if len(mzs) == 0:
            return np.array([]), np.array([]), 0.0

    # Intensity floor (relative to max peak)
    max_i = np.max(ints)
    if max_i <= 0:
        return np.array([]), np.array([]), 0.0
    floor_mask = ints >= (int_floor * max_i)
    mzs = mzs[floor_mask]
    ints = ints[floor_mask]
    if len(mzs) == 0:
        return np.array([]), np.array([]), 0.0
        
    # Top max_peaks
    if len(mzs) > max_peaks:
        top_idx = np.argsort(ints)[-max_peaks:]
        mzs = mzs[top_idx]
        ints = ints[top_idx]
        # Sort by m/z
        order = np.argsort(mzs)
        mzs = mzs[order]
        ints = ints[order]
    else:
        order = np.argsort(mzs)
        mzs = mzs[order]
        ints = ints[order]
        
    # Power transform and normalize to probability distribution
    w = np.power(ints, power)
    w_sum = np.sum(w)
    if w_sum <= 0:
        return np.array([]), np.array([]), 0.0
    w = w / w_sum
    
    # Entropy: S = -sum(w * ln(w))
    entropy = -float(np.sum(w * np.log(w + 1e-15)))
    return mzs, w, entropy

def spectral_entropy_similarity(mzs1, ints1, mzs2, ints2, prec_mz, mz_tol_ppm=15.0, max_peaks=64, int_floor=0.01, power=0.5):
    """
    Computes exact spectral entropy similarity between query (1) and ref (2) with precursor masking.
    Returns:
        entropy_sim: float in [0, 1]
        n_matched: int
        has_high_mz: bool
        max_matched_mz: float
        cos_masked: float (precursor-masked cosine for comparison)
    """
    m1, w1, s1 = calculate_spectral_entropy(mzs1, ints1, max_peaks, int_floor, power, prec_mz, mask_precursor=True)
    m2, w2, s2 = calculate_spectral_entropy(mzs2, ints2, max_peaks, int_floor, power, prec_mz, mask_precursor=True)
    
    if len(m1) == 0 or len(m2) == 0:
        return {
            "entropy_sim": 0.0,
            "cos_masked": 0.0,
            "n_matched": 0,
            "has_high_mz": False,
            "max_matched_mz": 0.0
        }
        
    # Two-pointer alignment within +-mz_tol_ppm
    half_prec = 0.5 * prec_mz if prec_mz is not None else 0.0
    j_start = 0
    n2 = len(m2)
    
    matched_q = []
    matched_r = []
    matched_mzs = []
    
    # For cosine
    dot_masked = 0.0
    norm1 = np.linalg.norm(w1)
    norm2 = np.linalg.norm(w2)
    
    # Keep track of matched indices in 2
    matched_j_set = set()
    matches = [] # list of (i, j, m1, w1, w2)
    
    for i, (mz_val, val1) in enumerate(zip(m1, w1)):
        tol = mz_val * mz_tol_ppm * 1e-6
        best_diff = tol
        best_j = -1
        for j in range(j_start, n2):
            diff = abs(mz_val - m2[j])
            if m2[j] < mz_val - tol:
                j_start = j
                continue
            if m2[j] > mz_val + tol:
                break
            if diff <= best_diff:
                best_diff = diff
                best_j = j
                
        if best_j != -1 and best_j not in matched_j_set:
            matched_j_set.add(best_j)
            matches.append((i, best_j, mz_val, val1, w2[best_j]))
            matched_mzs.append(mz_val)
            dot_masked += val1 * w2[best_j]
            
    n_matched = len(matches)
    has_high_mz = any(m > half_prec for m in matched_mzs)
    max_matched_mz = max(matched_mzs) if matched_mzs else 0.0
    cos_masked = float(dot_masked / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0
    
    # Now construct the mixture spectrum M = (P + Q) / 2
    # For matched pairs: weight in sum is w1_i + w2_j
    # For unmatched in P: weight is w1_i
    # For unmatched in Q: weight is w2_j
    # Sum of all weights is sum(w1) + sum(w2) = 1.0 + 1.0 = 2.0
    # Thus normalized mixture weights are W_k / 2.0.
    
    matched_i_set = {m[0] for m in matches}
    mixture_weights = []
    
    for _, _, _, v1, v2 in matches:
        mixture_weights.append((v1 + v2) / 2.0)
        
    for i, v1 in enumerate(w1):
        if i not in matched_i_set:
            mixture_weights.append(v1 / 2.0)
            
    for j, v2 in enumerate(w2):
        if j not in matched_j_set:
            mixture_weights.append(v2 / 2.0)
            
    mixture_weights = np.array(mixture_weights, dtype=np.float64)
    # Mixture entropy S(M)
    s_mix = -float(np.sum(mixture_weights * np.log(mixture_weights + 1e-15)))
    
    # Entropy distance D = 2 * S(M) - S(P) - S(Q)
    # Since mixture_weights is normalized (sum = 1), 2*S(M) - S(P) - S(Q) is exactly 2 * JSD(P, Q)
    entropy_dist = 2.0 * s_mix - s1 - s2
    entropy_dist = max(0.0, entropy_dist)
    
    # Entropy similarity = 1 - entropy_dist / ln(4)
    ln4 = np.log(4.0)
    entropy_sim = float(max(0.0, min(1.0, 1.0 - (entropy_dist / ln4))))
    
    return {
        "entropy_sim": entropy_sim,
        "cos_masked": cos_masked,
        "n_matched": n_matched,
        "has_high_mz": has_high_mz,
        "max_matched_mz": max_matched_mz,
        "s1": s1,
        "s2": s2,
        "s_mix": s_mix
    }

if __name__ == "__main__":
    # Test identical spectra
    mzs = np.array([100.0, 150.0, 200.0, 250.0])
    ints = np.array([10.0, 50.0, 100.0, 30.0])
    res_id = spectral_entropy_similarity(mzs, ints, mzs, ints, prec_mz=300.0)
    print("Identical test:", res_id)
    assert abs(res_id["entropy_sim"] - 1.0) < 1e-5, f"Expected 1.0, got {res_id['entropy_sim']}"
    
    # Test completely disjoint spectra
    mzs_d = np.array([105.0, 155.0, 205.0, 255.0])
    res_dis = spectral_entropy_similarity(mzs, ints, mzs_d, ints, prec_mz=300.0)
    print("Disjoint test:", res_dis)
    assert abs(res_dis["entropy_sim"] - 0.0) < 1e-5, f"Expected 0.0, got {res_dis['entropy_sim']}"
    
    # Test partially overlapping
    mzs_part = np.array([100.0, 150.0, 210.0, 260.0])
    res_part = spectral_entropy_similarity(mzs, ints, mzs_part, ints, prec_mz=300.0)
    print("Partial test:", res_part)
    print("All tests passed successfully!")
