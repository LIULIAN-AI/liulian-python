# Breakpoint bridge (VS Code ↔ PyCharm)

This workspace includes a shared breakpoint bridge file:

- `.debug/breakpoints.bridge.json`

Use `tools/breakpoint_bridge.py` to sync breakpoints between VS Code and PyCharm.

## Important constraints

- There is no stable official cross-IDE breakpoint API.
- VS Code sync works by reading/writing VS Code internal `state.vscdb` (`debug.breakpoint` key).
- PyCharm sync works by reading/writing `.idea/workspace.xml` (`XDebuggerManager`).
- Close the target IDE before importing breakpoints into it (to avoid overwrite by live IDE state).

## Commands

From project root:

```bash
python tools/breakpoint_bridge.py export-vscode
python tools/breakpoint_bridge.py import-pycharm
```

After you debug in PyCharm and want to sync back:

```bash
python tools/breakpoint_bridge.py export-pycharm
python tools/breakpoint_bridge.py import-vscode
```

Optional flags:

- `--bridge <path>`: custom bridge file path
- `--workspace-root <path>`: workspace root (default `.`)
- `export-vscode --db-path <.../state.vscdb>`: explicit VS Code DB
- `import-vscode --db-path <.../state.vscdb>`: explicit VS Code DB
- `list-vscode-dbs`: list candidate VS Code DBs (`*` means likely match)
- `export-pycharm --workspace-xml <.../.idea/workspace.xml>`
- `import-pycharm --workspace-xml <.../.idea/workspace.xml>`

If auto-detection fails, run:

```bash
python tools/breakpoint_bridge.py list-vscode-dbs
```

Then use the chosen DB explicitly:

```bash
python tools/breakpoint_bridge.py export-vscode --db-path /home/.../state.vscdb
python tools/breakpoint_bridge.py import-vscode --db-path /home/.../state.vscdb
```

## Recommended workflow

1. Set breakpoints in VS Code.
2. Run `export-vscode` to write bridge file.
3. Run `import-pycharm`.
4. Debug in PyCharm.
5. Remove/adjust breakpoints in PyCharm.
6. Run `export-pycharm` then `import-vscode`.

## Bridge format

```json
{
  "version": 1,
  "updated_at": "2026-02-17T00:00:00Z",
  "breakpoints": [
    {
      "path": "liulian/runtime/trainer.py",
      "line": 165,
      "enabled": true
    }
  ]
}
```
