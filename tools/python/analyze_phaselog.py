#!/usr/bin/env python3
# 多相機相位 log 分析（phaselog-yyyyMMdd.csv）。
# 欄位：wallclock,cam,seq,ticks,freqHz
# 按「時間斷層(>5s，新 grab session)」切段，每段算：相位 median/stdev、漂移率 ppm、掉幀。
# 對齊用「tick 就近配對」（同板相機共用板載時鐘 epoch）；offset 用 mod 幀週期置中（避免半幀邊界翻幀）。
# 用法：python analyze_phaselog.py <phaselog.csv>
import sys, bisect, statistics as st
from datetime import datetime

def parse(path):
    rows = []
    for ln in open(path):
        p = ln.strip().split(',')
        if len(p) < 5: continue
        try: rows.append((datetime.strptime(p[0], "%H:%M:%S.%f"), int(p[1]), int(p[3])))
        except: continue
    return rows

def segment(rows):
    """按 >5s 時間斷層切段（= 新 grab session）。回傳 [[(t,cam,tick)...], ...]"""
    segs = []; cur = []; last = None
    for t, cam, tk in rows:
        if last and (t - last).total_seconds() > 5 and cur:
            segs.append(cur); cur = []
        cur.append((t, cam, tk)); last = t
    if cur: segs.append(cur)
    return segs

def frame_period(c1):
    d = sorted(c1[i+1][1]-c1[i][1] for i in range(len(c1)-1) if 0 < c1[i+1][1]-c1[i][1] < 5e9)
    return d[len(d)//2] if d else 0

def analyze(seg):
    c1 = [(t, tk) for t, cam, tk in seg if cam == 1]
    c2 = sorted(tk for _, cam, tk in seg if cam == 2)
    if len(c1) < 3 or len(c2) < 3: return None
    dur = (seg[-1][0] - seg[0][0]).total_seconds()
    fp = frame_period(c1)
    drops = sum(1 for i in range(len(c1)-1) if fp and (c1[i+1][1]-c1[i][1]) > 1.5*fp)
    pts = []
    for t, tk in c1:
        i = bisect.bisect_left(c2, tk); cand = [c2[j] for j in (i-1, i) if 0 <= j < len(c2)]
        if not cand: continue
        n = min(cand, key=lambda x: abs(tk-x)); ph = (tk - n) / 125000.0  # ms (假設 125MHz)
        if fp and abs(ph) > fp/125000/2: continue
        pts.append((t, ph))
    if not pts: return None
    ys = [p for _, p in pts]
    drift = None
    if len(pts) > 4 and dur > 10:
        t0 = pts[0][0]; xs = [(t-t0).total_seconds() for t, _ in pts]
        mx = sum(xs)/len(xs); my = sum(ys)/len(ys); den = sum((x-mx)**2 for x in xs)
        if den > 0: drift = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/den*1000  # us/s
    return dict(t0=seg[0][0], dur=dur, n1=len(c1), n2=len(c2), fp_ms=fp/125000 if fp else 0,
                drops=drops, med=st.median(ys), std=st.pstdev(ys), mn=min(ys), mx=max(ys), drift=drift)

def main():
    rows = parse(sys.argv[1])
    if not rows: print("無資料"); return
    print(f"共 {len(rows)} 行，{rows[0][0].strftime('%H:%M:%S')}~{rows[-1][0].strftime('%H:%M:%S')}\n")
    hdr = f"{'起始':>8} {'秒':>5} {'幀/cam1':>7} {'幀週ms':>6} {'掉幀':>4} {'相位med_ms':>10} {'相位std_ms':>10} {'相位範圍ms':>16} {'漂移us/s':>9}"
    print(hdr); print("-"*len(hdr))
    for seg in segment(rows):
        r = analyze(seg)
        if not r: continue
        rng = f"[{r['mn']:.3f},{r['mx']:.3f}]"
        dr = f"{r['drift']:.3f}" if r['drift'] is not None else "-"
        print(f"{r['t0'].strftime('%H:%M:%S'):>8} {r['dur']:5.0f} {r['n1']:7d} {r['fp_ms']:6.0f} {r['drops']:4d} {r['med']:10.4f} {r['std']:10.4f} {rng:>16} {dr:>9}")

if __name__ == "__main__": main()
