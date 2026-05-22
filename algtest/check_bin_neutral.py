"""讀 .bin 曲線檔做統計，驗證中性化（無 clamp）改動是否生效。

.bin 格式（CameraFrameSaver.SaveCurveBinFromArray）:
    magic(4) "MCBF" + version(4)=1 + scale_factor(4f) + array_length(4) + float32[]

預期：
- 舊（clamp+u8）：max <= 255.0，常見 ≈ 255 飽和
- 新（無 clamp）：max 可超過 255，峰值較銳利
"""

import struct
import sys
from pathlib import Path


def read_bin(path: Path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"MCBF":
            raise ValueError(f"bad magic in {path}: {magic!r}")
        version, scale, n = struct.unpack("<ifi", f.read(12))
        data = struct.unpack(f"<{n}f", f.read(n * 4))
        return version, scale, list(data)


def stat(arr):
    n = len(arr)
    s = sum(arr)
    mx = max(arr)
    mn = min(arr)
    mean = s / n
    over_255 = sum(1 for v in arr if v > 255.0)
    over_300 = sum(1 for v in arr if v > 300.0)
    return dict(n=n, min=mn, max=mx, mean=mean, over_255=over_255, over_300=over_300)


def report(path: Path):
    v, scale, data = read_bin(path)
    s = stat(data)
    flag = ""
    if s["max"] <= 255.0001:
        flag = "  <- max <=255 (looks CLAMPED, old)"
    elif s["max"] > 255.0:
        flag = "  <- max >255 (NEUTRAL, new)"
    print(f"{path.name:55s}  len={s['n']:>5d}  "
          f"max={s['max']:>9.3f}  mean={s['mean']:>7.3f}  "
          f"#>255={s['over_255']:>4d}  #>300={s['over_300']:>4d}{flag}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = [Path(p) for p in sys.argv[1:]]
    else:
        # 預設：列出最新一批
        root = Path(r"D:\Anilox\Captures")
        files = sorted(root.rglob("*.bin"), key=lambda p: p.stat().st_mtime, reverse=True)[:24]
    for p in files:
        try:
            report(p)
        except Exception as e:
            print(f"{p}: ERROR {e}")
