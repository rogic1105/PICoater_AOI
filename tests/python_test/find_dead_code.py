"""Dead code candidate finder for AniloxRoll.Monitor.

策略：
1. 掃描所有 .cs 檔抽出 method definitions（private/internal/public/static）
2. 對每個方法名 grep 全 codebase，計算引用次數
3. 引用次數 = 1（只有定義本身）→ dead 候選
4. 排除明顯的 framework hooks（Form1_Load, OnPaint, Designer-generated 等）

輸出：候選清單 + 簡單統計，不自動刪除。

用法：
    python tests/python_test/find_dead_code.py [src_root]
預設 src_root = src_dotnet/AniloxRoll.Monitor
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

# WinForms / .NET 框架自動呼叫的方法（不應視為 dead）
FRAMEWORK_HOOKS = {
    "InitializeComponent", "Dispose", "Main", "Equals", "GetHashCode", "ToString",
    "Finalize", "MemberwiseClone",
    # WinForms event handler 命名（Designer 註冊）
    "OnPaint", "OnLoad", "OnClosing", "OnClosed", "OnResize", "OnShown",
    "OnKeyDown", "OnKeyUp", "OnKeyPress", "OnMouseDown", "OnMouseUp", "OnMouseMove",
}

# 排除的目錄
EXCLUDE_DIRS = {"bin", "obj", "build", "Properties", ".vs", "TestRunner"}

# Method 定義 regex（簡化版，不支援泛型限制）
METHOD_PATTERN = re.compile(
    r"^\s*(?:public|private|protected|internal)\s+"
    r"(?:static\s+|async\s+|virtual\s+|override\s+|sealed\s+|new\s+)*"
    r"(?:[\w<>?\[\],\s]+?)\s+"
    r"([A-Z]\w*)\s*\("
)

# Property 定義 regex
PROPERTY_PATTERN = re.compile(
    r"^\s*(?:public|private|protected|internal)\s+"
    r"(?:static\s+|virtual\s+|override\s+)*"
    r"(?:[\w<>?\[\],\s]+?)\s+"
    r"([A-Z]\w*)\s*\{\s*(?:get|set)"
)


def iter_cs_files(root: Path):
    for p in root.rglob("*.cs"):
        if any(d in p.parts for d in EXCLUDE_DIRS):
            continue
        if p.name.endswith(".Designer.cs"):
            continue  # designer 檔不掃定義（其方法本來就由 designer 內部呼叫）
        yield p


def extract_definitions(path: Path):
    """回傳 (method_name, line_no) list。"""
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                m = METHOD_PATTERN.match(line)
                if m:
                    name = m.group(1)
                    if name in FRAMEWORK_HOOKS:
                        continue
                    out.append((name, i))
                p = PROPERTY_PATTERN.match(line)
                if p:
                    out.append((p.group(1), i))
    except UnicodeDecodeError:
        pass
    return out


def count_references(name: str, all_text: str):
    """全 codebase 出現次數（不分大小寫邊界考慮）。"""
    pattern = r"\b" + re.escape(name) + r"\b"
    return len(re.findall(pattern, all_text))


def main():
    src_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src_dotnet/AniloxRoll.Monitor")
    if not src_root.exists():
        print(f"ERROR: {src_root} not found")
        return 1

    files = list(iter_cs_files(src_root))
    print(f"# Scanning {len(files)} .cs files under {src_root}")

    # Step 1: 收集所有定義
    defs_by_file = {}
    for f in files:
        ds = extract_definitions(f)
        if ds:
            defs_by_file[f] = ds

    # Step 2: 載入全 codebase 文字（包括 Designer.cs 因為它會引用我們的方法）
    # 不止主程式，也掃 tests 因為測試會引用 internal
    all_files = []
    for root_dir in [Path("src_dotnet"), Path("tests/dotnet_test")]:
        if root_dir.exists():
            for p in root_dir.rglob("*.cs"):
                if not any(d in p.parts for d in {"bin", "obj", "build", ".vs"}):
                    all_files.append(p)

    all_text_parts = []
    for f in all_files:
        try:
            all_text_parts.append(f.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            try:
                all_text_parts.append(f.read_text(encoding="utf-16"))
            except Exception:
                pass
    all_text = "\n".join(all_text_parts)

    # Step 3: 找出候選
    print(f"\n# Dead code candidates (引用次數 <= 1，扣掉定義本身 = 無呼叫者)\n")
    candidates = []
    for f, defs in defs_by_file.items():
        for name, lineno in defs:
            cnt = count_references(name, all_text)
            if cnt <= 1:
                candidates.append((f, lineno, name, cnt))

    # 依檔案分組輸出
    by_file = defaultdict(list)
    for f, ln, n, c in candidates:
        by_file[f].append((ln, n, c))

    for f in sorted(by_file.keys()):
        print(f"\n## {f.as_posix()}")
        for ln, name, cnt in sorted(by_file[f]):
            print(f"  L{ln:>5d}  {name}  (refs={cnt})")

    print(f"\n# Total: {len(candidates)} candidates across {len(by_file)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
