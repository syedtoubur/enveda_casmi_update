"""Update ice_route_skip in E:/casmi/lines/exploit-b/build/casmi_post.py to handle float tolerance."""
from pathlib import Path

P = Path("E:/casmi/lines/exploit-b/build/casmi_post.py")
content = P.read_text(encoding="utf-8")

old_return = "    return ((s1 - s2) / s1) >= float(margin)"
new_return = "    return ((s1 - s2) / s1) >= float(margin) - 1e-9"
assert old_return in content
content = content.replace(old_return, new_return, 1)
P.write_text(content, encoding="utf-8")
print("Updated casmi_post.py with float tolerance")
