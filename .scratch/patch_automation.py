import os

path = '/home/tayeb/.local/lib/python3.12/site-packages/colab_cli/commands/automation.py'
with open(path, 'r') as f:
    text = f.read()

target = """            if not data.get("success"):
                uri = data.get("unauthorized_redirect_uri")
                typer.echo(
                    f"\\n[colab] REQUIRED: Google Drive Authorization needed.\\nPlease visit:\\n\\n{uri}\\n"
                )
                state.history.log_event(s.name, "drive_auth_needed", {"uri": uri})
                pass"""

replacement = """            if not data.get("success"):
                uri = data.get("unauthorized_redirect_uri")
                typer.echo(
                    f"\\n[colab] REQUIRED: Google Drive Authorization needed.\\nPlease visit:\\n\\n{uri}\\n"
                )
                state.history.log_event(s.name, "drive_auth_needed", {"uri": uri})
                with open("/home/tayeb/drive_url.txt", "w") as _uf:
                    _uf.write(uri)
                input("\\n[colab] Action required: Visit the link above in your browser to approve, then press Enter to continue...")"""

if target in text:
    with open(path, 'w') as f:
        f.write(text.replace(target, replacement))
    print('PATCH_SUCCESS')
else:
    print('TARGET_NOT_FOUND')
