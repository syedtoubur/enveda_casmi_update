import os

bashrc_path = os.path.expanduser("~/.bashrc")
with open(bashrc_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "export PATH=" in line and "C:\\Users" in line:
        continue
    if "export PATH=" in line and ".local/bin" in line:
        continue
    new_lines.append(line)

# Ensure proper exports at the end
if "export OAUTHLIB_RELAX_TOKEN_SCOPE=1\n" not in new_lines:
    new_lines.append("export OAUTHLIB_RELAX_TOKEN_SCOPE=1\n")
new_lines.append('export PATH="$HOME/.local/bin:$PATH"\n')

with open(bashrc_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Updated ~/.bashrc successfully.")
