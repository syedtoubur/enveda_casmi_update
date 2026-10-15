"""Create E:/casmi/lines/exploit-b/build/test_ice_route.py."""
from pathlib import Path

content = '''"""Unit tests for ice_route_skip and ICE_ROUTE_MARGIN in casmi_post.py.

usage: python build/test_ice_route.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import casmi_post as cp


def test_ice_route_skip_edge_cases():
    # None margin always returns False
    assert cp.ice_route_skip([0.5, 0.2], None) is False
    assert cp.ice_route_skip([], None) is False
    
    # Empty or single score returns False
    assert cp.ice_route_skip([], 0.15) is False
    assert cp.ice_route_skip([0.5], 0.15) is False
    
    # Non-positive top score returns False
    assert cp.ice_route_skip([0.0, 0.0], 0.15) is False
    assert cp.ice_route_skip([-0.1, -0.2], 0.15) is False


def test_ice_route_skip_margins():
    # s1 = 0.25, s2 = 0.20 -> rel_margin = 0.05 / 0.25 = 0.20
    assert cp.ice_route_skip([0.25, 0.20], 0.15) is True
    assert cp.ice_route_skip([0.25, 0.20], 0.25) is False
    assert cp.ice_route_skip([0.25, 0.20], 0.20) is True  # boundary exact

    # s1 = 0.40, s2 = 0.35 -> rel_margin = 0.05 / 0.40 = 0.125
    assert cp.ice_route_skip([0.40, 0.35], 0.1521) is False
    assert cp.ice_route_skip([0.40, 0.30], 0.1521) is True  # 0.10 / 0.40 = 0.25 >= 0.1521


def test_no_op_cfg_default():
    assert cp.CFG['ICE_ROUTE_MARGIN'] is None


if __name__ == '__main__':
    test_ice_route_skip_edge_cases()
    test_ice_route_skip_margins()
    test_no_op_cfg_default()
    print("ALL UNIT TESTS PASSED.")
'''

Path("E:/casmi/lines/exploit-b/build/test_ice_route.py").write_text(content, encoding="utf-8")
print("Wrote E:/casmi/lines/exploit-b/build/test_ice_route.py")
