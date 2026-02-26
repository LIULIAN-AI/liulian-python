#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _norm_rel_path(path: str) -> str:
    return path.replace('\\', '/').lstrip('./')


def _read_bridge(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            'version': 1,
            'updated_at': _now_iso(),
            'breakpoints': [],
        }
    return json.loads(path.read_text(encoding='utf-8'))


def _write_bridge(path: Path, payload: Dict[str, Any]) -> None:
    payload['updated_at'] = _now_iso()
    payload['version'] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def _dedupe_breakpoints(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        path = _norm_rel_path(str(item.get('path', '')))
        line = int(item.get('line', 0))
        enabled = bool(item.get('enabled', True))
        if not path or line <= 0:
            continue
        key = (path, line)
        if key in seen:
            continue
        seen.add(key)
        out.append({'path': path, 'line': line, 'enabled': enabled})
    out.sort(key=lambda x: (x['path'], x['line']))
    return out


def _backup_file(path: Path) -> Path:
    stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    bak = path.with_suffix(path.suffix + f'.bak.{stamp}')
    bak.write_bytes(path.read_bytes())
    return bak


def _ensure_xdebug_breakpoints_root(root: ET.Element) -> ET.Element:
    comp = root.find("./component[@name='XDebuggerManager']")
    if comp is None:
        comp = ET.SubElement(root, 'component', {'name': 'XDebuggerManager'})
    manager = comp.find('./breakpoint-manager')
    if manager is None:
        manager = ET.SubElement(comp, 'breakpoint-manager')
    bps = manager.find('./breakpoints')
    if bps is None:
        bps = ET.SubElement(manager, 'breakpoints')
    return bps


def _parse_pycharm_workspace(workspace_xml: Path) -> Tuple[List[Dict[str, Any]], int]:
    tree = ET.parse(workspace_xml)
    root = tree.getroot()
    bps_root = _ensure_xdebug_breakpoints_root(root)

    result: List[Dict[str, Any]] = []
    max_stamp = 0
    for node in list(bps_root):
        if node.tag != 'line-breakpoint':
            continue
        if node.attrib.get('type') != 'python-line':
            continue

        url_node = node.find('./url')
        line_node = node.find('./line')
        ts_node = node.find("./option[@name='timeStamp']")
        if url_node is None or line_node is None or not url_node.text or not line_node.text:
            continue

        url = url_node.text.strip()
        if not url.startswith('file://$PROJECT_DIR$/'):
            continue

        rel_path = url.replace('file://$PROJECT_DIR$/', '', 1)
        try:
            zero_based = int(line_node.text.strip())
        except ValueError:
            continue
        enabled = node.attrib.get('enabled', 'true').lower() == 'true'
        result.append({'path': _norm_rel_path(rel_path), 'line': zero_based + 1, 'enabled': enabled})

        if ts_node is not None:
            try:
                max_stamp = max(max_stamp, int(ts_node.attrib.get('value', '0')))
            except ValueError:
                pass

    return _dedupe_breakpoints(result), max_stamp


def export_pycharm(workspace_xml: Path, bridge_path: Path, merge: bool) -> None:
    current = _read_bridge(bridge_path)
    py_bps, _ = _parse_pycharm_workspace(workspace_xml)

    if merge:
        merged = _dedupe_breakpoints(list(current.get('breakpoints', [])) + py_bps)
    else:
        merged = py_bps

    current['breakpoints'] = merged
    _write_bridge(bridge_path, current)
    print(f'Exported {len(py_bps)} PyCharm breakpoints to {bridge_path}')


def import_pycharm(workspace_xml: Path, bridge_path: Path, replace_python: bool) -> None:
    payload = _read_bridge(bridge_path)
    bridge_bps = _dedupe_breakpoints(list(payload.get('breakpoints', [])))

    tree = ET.parse(workspace_xml)
    root = tree.getroot()
    bps_root = _ensure_xdebug_breakpoints_root(root)

    _, max_stamp = _parse_pycharm_workspace(workspace_xml)
    next_stamp = max_stamp + 1

    kept: List[ET.Element] = []
    for node in list(bps_root):
        if node.tag == 'line-breakpoint' and node.attrib.get('type') == 'python-line' and replace_python:
            continue
        kept.append(node)

    for node in list(bps_root):
        bps_root.remove(node)
    for node in kept:
        bps_root.append(node)

    for bp in bridge_bps:
        lb = ET.SubElement(
            bps_root,
            'line-breakpoint',
            {
                'enabled': 'true' if bp.get('enabled', True) else 'false',
                'suspend': 'THREAD',
                'type': 'python-line',
            },
        )
        url = ET.SubElement(lb, 'url')
        url.text = f"file://$PROJECT_DIR$/{_norm_rel_path(bp['path'])}"
        line = ET.SubElement(lb, 'line')
        line.text = str(int(bp['line']) - 1)
        ET.SubElement(lb, 'option', {'name': 'timeStamp', 'value': str(next_stamp)})
        next_stamp += 1

    _backup_file(workspace_xml)
    tree.write(workspace_xml, encoding='UTF-8', xml_declaration=False)
    print(f'Imported {len(bridge_bps)} breakpoints into {workspace_xml}')


def _find_vscode_db(workspace_root: Path) -> Optional[Path]:
    storage = Path.home() / '.config' / 'Code' / 'User' / 'workspaceStorage'
    if not storage.exists():
        return None

    root_str = str(workspace_root.resolve())
    for db in storage.glob('*/state.vscdb'):
        try:
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute("SELECT value FROM ItemTable WHERE key='debug.breakpoint'")
            row = cur.fetchone()
            con.close()
            if not row or row[0] is None:
                continue
            if root_str in row[0] or workspace_root.name in row[0]:
                return db
        except Exception:
            continue
    return None


def _list_vscode_db_candidates(workspace_root: Path) -> List[Tuple[Path, int, bool]]:
    storage = Path.home() / '.config' / 'Code' / 'User' / 'workspaceStorage'
    if not storage.exists():
        return []

    root_str = str(workspace_root.resolve())
    results: List[Tuple[Path, int, bool]] = []
    for db in storage.glob('*/state.vscdb'):
        try:
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute("SELECT value FROM ItemTable WHERE key='debug.breakpoint'")
            row = cur.fetchone()
            con.close()

            count = 0
            contains_workspace = False
            if row and row[0]:
                raw = json.loads(row[0])
                count = len(raw) if isinstance(raw, list) else 0
                text = row[0]
                contains_workspace = (root_str in text) or (workspace_root.name in text)
            results.append((db, count, contains_workspace))
        except Exception:
            continue

    results.sort(key=lambda x: (not x[2], -x[1], str(x[0])))
    return results


def _read_vscode_breakpoints(db_path: Path, workspace_root: Path) -> List[Dict[str, Any]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT value FROM ItemTable WHERE key='debug.breakpoint'")
    row = cur.fetchone()
    con.close()
    if not row or row[0] is None:
        return []

    raw = json.loads(row[0])
    out: List[Dict[str, Any]] = []
    root_str = str(workspace_root.resolve())

    for bp in raw:
        uri = bp.get('uri', {})
        fs_path = uri.get('fsPath')
        if not fs_path:
            continue
        fs_path = str(fs_path)
        if not fs_path.startswith(root_str):
            continue
        rel = os.path.relpath(fs_path, root_str)
        line = int(bp.get('lineNumber', 1))
        enabled = bool(bp.get('enabled', True))
        out.append({'path': _norm_rel_path(rel), 'line': line, 'enabled': enabled})

    return _dedupe_breakpoints(out)


def _write_vscode_breakpoints(
    db_path: Path,
    workspace_root: Path,
    bridge_bps: List[Dict[str, Any]],
    replace_for_workspace: bool,
) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT value FROM ItemTable WHERE key='debug.breakpoint'")
    row = cur.fetchone()

    if row and row[0]:
        current = json.loads(row[0])
    else:
        current = []

    root_str = str(workspace_root.resolve())

    if replace_for_workspace:
        kept = []
        for bp in current:
            uri = bp.get('uri', {})
            fs_path = str(uri.get('fsPath', ''))
            if not fs_path.startswith(root_str):
                kept.append(bp)
        current = kept

    existing_ids = {str(bp.get('id', '')) for bp in current}

    def _next_id() -> str:
        n = 1
        while True:
            candidate = f'bridge-{int(time.time())}-{n}'
            if candidate not in existing_ids:
                existing_ids.add(candidate)
                return candidate
            n += 1

    for bp in bridge_bps:
        abs_path = str((workspace_root / bp['path']).resolve())
        current.append(
            {
                'id': _next_id(),
                'enabled': bool(bp.get('enabled', True)),
                'uri': {
                    '$mid': 1,
                    'fsPath': abs_path,
                    'external': Path(abs_path).as_uri(),
                    'path': abs_path,
                    'scheme': 'file',
                },
                'lineNumber': int(bp['line']),
                'column': 1,
                'condition': None,
                'hitCondition': None,
                'logMessage': None,
                'mode': None,
                'modeLabel': '',
                'adapterData': None,
                'triggeredBy': None,
            }
        )

    cur.execute('INSERT OR REPLACE INTO ItemTable(key,value) VALUES(?,?)', ('debug.breakpoint', json.dumps(current)))
    con.commit()
    con.close()


def export_vscode(workspace_root: Path, bridge_path: Path, db_path: Optional[Path], merge: bool) -> None:
    current = _read_bridge(bridge_path)
    db = db_path or _find_vscode_db(workspace_root)
    if db is None:
        raise RuntimeError('Could not find VS Code state.vscdb for this workspace.')

    vs_bps = _read_vscode_breakpoints(db, workspace_root)
    if merge:
        merged = _dedupe_breakpoints(list(current.get('breakpoints', [])) + vs_bps)
    else:
        merged = vs_bps

    current['breakpoints'] = merged
    _write_bridge(bridge_path, current)
    print(f'Exported {len(vs_bps)} VS Code breakpoints from {db} to {bridge_path}')


def import_vscode(workspace_root: Path, bridge_path: Path, db_path: Optional[Path], replace_for_workspace: bool) -> None:
    payload = _read_bridge(bridge_path)
    bridge_bps = _dedupe_breakpoints(list(payload.get('breakpoints', [])))

    db = db_path or _find_vscode_db(workspace_root)
    if db is None:
        raise RuntimeError('Could not find VS Code state.vscdb for this workspace.')

    _write_vscode_breakpoints(db, workspace_root, bridge_bps, replace_for_workspace)
    print(f'Imported {len(bridge_bps)} breakpoints from {bridge_path} into {db}')


def clear_bridge(bridge_path: Path) -> None:
    payload = _read_bridge(bridge_path)
    payload['breakpoints'] = []
    _write_bridge(bridge_path, payload)
    print(f'Cleared bridge file: {bridge_path}')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Breakpoint bridge between VS Code and PyCharm.')
    p.add_argument('--bridge', default='.debug/breakpoints.bridge.json', help='Path to shared bridge JSON file.')
    p.add_argument('--workspace-root', default='.', help='Workspace root path for relative file paths.')

    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('export-pycharm', help='Read PyCharm breakpoints into bridge file.')
    s.add_argument('--workspace-xml', default='.idea/workspace.xml')
    s.add_argument('--merge', action='store_true')

    s = sub.add_parser('import-pycharm', help='Write bridge breakpoints into PyCharm workspace.xml.')
    s.add_argument('--workspace-xml', default='.idea/workspace.xml')
    s.add_argument('--replace-python', action='store_true', default=True)

    s = sub.add_parser('export-vscode', help='Read VS Code breakpoints into bridge file.')
    s.add_argument('--db-path', default=None, help='Optional explicit VS Code state.vscdb path.')
    s.add_argument('--merge', action='store_true')

    s = sub.add_parser('import-vscode', help='Write bridge breakpoints into VS Code state.vscdb.')
    s.add_argument('--db-path', default=None, help='Optional explicit VS Code state.vscdb path.')
    s.add_argument('--replace-workspace', action='store_true', default=True)

    sub.add_parser('list-vscode-dbs', help='List candidate VS Code state.vscdb files.')

    sub.add_parser('clear', help='Clear all bridge breakpoints.')
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    bridge = Path(args.bridge).resolve()
    workspace_root = Path(args.workspace_root).resolve()

    try:
        if args.cmd == 'export-pycharm':
            export_pycharm(Path(args.workspace_xml).resolve(), bridge, bool(args.merge))
        elif args.cmd == 'import-pycharm':
            import_pycharm(Path(args.workspace_xml).resolve(), bridge, bool(args.replace_python))
        elif args.cmd == 'export-vscode':
            db = Path(args.db_path).resolve() if args.db_path else None
            export_vscode(workspace_root, bridge, db, bool(args.merge))
        elif args.cmd == 'import-vscode':
            db = Path(args.db_path).resolve() if args.db_path else None
            import_vscode(workspace_root, bridge, db, bool(args.replace_workspace))
        elif args.cmd == 'clear':
            clear_bridge(bridge)
        elif args.cmd == 'list-vscode-dbs':
            rows = _list_vscode_db_candidates(workspace_root)
            if not rows:
                print('No VS Code workspaceStorage databases found.')
            else:
                for db, count, match in rows:
                    marker = '*' if match else ' '
                    print(f'[{marker}] breakpoints={count:3d}  {db}')
        else:
            parser.print_help()
            return 2
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
