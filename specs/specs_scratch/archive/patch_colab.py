target = '/home/tayeb/.local/lib/python3.12/site-packages/colab_cli/commands/automation.py'
with open(target, 'r', encoding='utf-8') as f:
    content = f.read()

bad = """                state.history.log_event(s.name, "drive_auth_needed", {"uri": uri})
pass

            typer.echo("[colab] Authorizing VM...")"""

good = """                state.history.log_event(s.name, "drive_auth_needed", {"uri": uri})
                pass

            typer.echo("[colab] Authorizing VM...")"""

if bad in content:
    content = content.replace(bad, good)
    with open(target, 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS: Fixed indentation!")
else:
    print("Pattern not found")
