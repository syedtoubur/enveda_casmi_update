"""Patch E:/casmi/lines/exploit-b/build/casmi_post.py with ICE_ROUTE_MARGIN."""
from pathlib import Path

P = Path("E:/casmi/lines/exploit-b/build/casmi_post.py")
content = P.read_text(encoding="utf-8")

# 1. Add ICE_ROUTE_MARGIN to CFG
old_cfg = "    FUSION_WINDOW=None,      # fused shortlist size before the simulator rerank (None = FUSE_N; E006 uses 40)"
new_cfg = """    FUSION_WINDOW=None,      # fused shortlist size before the simulator rerank (None = FUSE_N; E006 uses 40)
    ICE_ROUTE_MARGIN=None,   # relative margin (s1 - s2)/s1 threshold to skip ICE fallback (None = off)"""
assert old_cfg in content, "old_cfg not found"
content = content.replace(old_cfg, new_cfg, 1)

# 2. Add ice_route_skip helper after metric_fusion
old_fn_end = "    return [representatives[key] for key in ordered], [scores[key] for key in ordered], collapsed\n"
new_fn = """    return [representatives[key] for key in ordered], [scores[key] for key in ordered], collapsed


def ice_route_skip(scores, margin):
    \"\"\"True iff top-1 vs top-2 relative margin (s1 - s2)/s1 is at least `margin`.\"\"\"
    if margin is None:
        return False
    if not scores or len(scores) < 2:
        return False
    s1, s2 = float(scores[0]), float(scores[1])
    if s1 <= 0.0:
        return False
    return ((s1 - s2) / s1) >= float(margin)
"""
assert old_fn_end in content, "old_fn_end not found"
content = content.replace(old_fn_end, new_fn, 1)

# 3. Add skip logic to _metric_overlay
old_fallback = """            elif c['ICE_FALLBACK'] and ice and not gl:
                order = ice_rerank(cands, keys, scores, [form(s) for s in cands], ice,
                                   lam=c['ICE_LAM'], top_n=len(cands))
                cands = [cands[i] for i in order]
                stats['ice_fallback'] = stats.get('ice_fallback', 0) + 1"""

new_fallback = """            elif c['ICE_FALLBACK'] and ice and not gl:
                if ice_route_skip(scores, c.get('ICE_ROUTE_MARGIN')):
                    stats['ice_route_skipped'] = stats.get('ice_route_skipped', 0) + 1
                else:
                    order = ice_rerank(cands, keys, scores, [form(s) for s in cands], ice,
                                       lam=c['ICE_LAM'], top_n=len(cands))
                    cands = [cands[i] for i in order]
                    stats['ice_fallback'] = stats.get('ice_fallback', 0) + 1"""

assert old_fallback in content, "old_fallback not found"
content = content.replace(old_fallback, new_fallback, 1)

P.write_text(content, encoding="utf-8")
print("Successfully patched E:/casmi/lines/exploit-b/build/casmi_post.py")
