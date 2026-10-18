import json
import os
import sys

def main():
    json_path = r"E:\ENVEDA_casmi\research\c3\20261018-iso-rank\stage_a_result.json"
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=== STAGE A RESULT SIGNAL AUDIT ===")
    
    # 1. Forward trace in frozen selection (Scaffold Half A)
    forward_trace = data.get("frozen", {}).get("forward_trace", [])
    print(f"\n--- Forward Selection on Scaffold Half A (n_pool=49) ---")
    for step_info in forward_trace:
        step = step_info.get("step")
        tried = step_info.get("tried", {})
        sorted_tried = sorted(tried.items(), key=lambda x: x[1], reverse=True)
        print(f"Step {step}:")
        for sig, val in sorted_tried:
            print(f"   {sig:12s}: MRR@20 = {val:.4f}")

    # 2. Grid delta on Half A for m_ice
    grid_delta = data.get("frozen", {}).get("grid_delta_A", {})
    print(f"\n--- Half A Delta MRR@25 by m_ice ---")
    for m, d in grid_delta.items():
        print(f"   m_ice = {m:4s}: Delta = {d:+.6f}")

    # 3. Pseudo_C3 Half B evaluation
    pc3_b = data.get("pseudo_c3_B", {})
    print(f"\n--- pseudo_c3 Half B (n={pc3_b.get('n')}) ---")
    print(f"   Overall Delta     : {pc3_b.get('delta'):+.6f}")
    print(f"   Fire-only Delta   : {pc3_b.get('delta_fire_only'):+.6f}")
    print(f"   Fire rate         : {pc3_b.get('fire_rate')*100:.2f}%")
    print(f"   Up/Down count     : {pc3_b.get('up')} up / {pc3_b.get('down')} down")

    # 4. C1/C2 breakdown
    c12 = data.get("c12", {})
    print(f"\n--- C1/C2 Cohorts Breakdown ---")
    for cohort, configs in c12.items():
        print(f"\nCohort: {cohort}")
        for cfg_name, res in configs.items():
            if "B/frozen" in cfg_name or "all/frozen" in cfg_name:
                print(f"   {cfg_name:25s}: n={res.get('n')}, fire_rate={res.get('fire_rate')*100:.2f}%, "
                      f"down={res.get('down')}, Delta={res.get('delta'):+.6f}, "
                      f"Fire-only Delta={res.get('delta_wc_fire_only'):+.6f}")

    # 5. Composite results
    comp = data.get("composite", {}).get("frozen_frag_only", {})
    print(f"\n--- Leaderboard Composite (frozen_frag_only) ---")
    for pi, vals in comp.items():
        print(f"   pi = {pi}: Worst case = {vals.get('worst_case'):+.6f}, Actual = {vals.get('actual'):+.6f}")

if __name__ == "__main__":
    main()
