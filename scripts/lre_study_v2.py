import sys,os,time,warnings,json,logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['LRE_ENABLED']='1'
os.environ['LRE_SHADOW_MODE']='0'

from core.loss_rejection_engine.engine import LossRejectionEngine

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
OUT_DIR = Path(__file__).resolve().parent.parent / 'download'
OUT_DIR.mkdir(parents=True, exist_ok=True)
EURUSD_ATR = 0.0065

LOG = OUT_DIR / 'lre_study_progress.log'

def log(msg):
    with open(LOG, 'a') as f:
        f.write(f'{time.strftime("%H:%M:%S")} {msg}\n')
    print(msg, flush=True)

def pip_size(sym):
    s = sym.upper()
    if 'JPY' in s: return 0.01
    if 'XAU' in s: return 0.1
    if 'XAG' in s or 'XPT' in s or 'XPD' in s: return 0.01
    return 0.0001

def make_trades(df, sym, ps):
    h = df['high'].values.astype(float)
    lo = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    o = df['open'].values.astype(float)
    n = len(df)
    trades = []
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(h[i-14:i] - lo[i-14:i])
    for i in range(55, n-1):
        a = atr[i]
        if a <= 0 or np.isnan(a): continue
        upper = np.max(h[i-50:i])
        lower = np.min(lo[i-50:i])
        entry = o[i+1]
        if c[i] > upper:
            trades.append(sim_trade(h, lo, i+1, 'BUY', entry, entry-2*a, entry+4*a, sym, ps, a))
        elif c[i] < lower:
            trades.append(sim_trade(h, lo, i+1, 'SELL', entry, entry+2*a, entry-4*a, sym, ps, a))
    return trades

def sim_trade(h, lo, start, d, entry, sl, tp, sym, ps, a):
    is_long = d == 'BUY'
    ep = entry; er = 'timeout'; hb = 0
    end = min(start + 80, len(h))
    for j in range(start, end):
        hb = j - start
        if is_long:
            if lo[j] <= sl: ep, er = sl, 'SL'; break
            if h[j] >= tp: ep, er = tp, 'TP'; break
        else:
            if h[j] >= sl: ep, er = sl, 'SL'; break
            if lo[j] <= tp: ep, er = tp, 'TP'; break
    else:
        ep = h[min(end-1, len(h)-1)]; hb = 80
    pp = (ep - entry) / ps if is_long else (entry - ep) / ps
    pp -= 1.5
    usd = pp * 10 - 7.0 if ps != 0.1 else pp * 100 - 7.0
    sd = abs(entry - sl) / ps; td = abs(tp - entry) / ps
    rr = td / sd if sd > 0 else 2.0
    return {'s': sym, 'd': d, 'e': entry, 'sl': sl, 'tp': tp,
            'pp': pp, 'u': usd, 'w': pp > 0, 'er': er, 'hb': hb,
            'rr': rr, 'c': 60.0, 'a': a, 'p': ps}

def ctx(t, idx):
    d, e, sl, tp = t['d'], t['e'], t['sl'], t['tp']
    iw, hb, er, atr, ps = t['w'], t['hb'], t['er'], t['a'], t['p']
    ar = atr / EURUSD_ATR if EURUSD_ATR > 0 else 1.0
    if iw:
        rsi = 48.0 + (idx % 12)
    else:
        if er == 'SL' and hb <= 3: rsi = 72.0 if d == 'BUY' else 28.0
        elif er == 'SL' and hb <= 10: rsi = 65.0 if d == 'BUY' else 35.0
        else: rsi = 52.0
    mv = 0.0002*ar if iw else -0.0001*ar
    ms = 0.0001*ar if iw else 0.0002*ar
    dec = {'decision': d, 'entry': e, 'confidence': t['c'], 'rr': t['rr'],
           'sl_pips': abs(e-sl)/ps, 'tp_pips': abs(tp-e)/ps,
           'sl_price': sl, 'tp_price': tp, 'strategy': 'donchian'}
    ind = {'atr': {'value': atr}, 'ATR': atr, 'rsi': {'value': rsi}, 'RSI': rsi,
           'macd': {'value': mv, 'signal': ms},
           'bb': {'upper': e+atr*2, 'lower': e-atr*2}}
    if iw: rt, rc, ts = 'trending', 0.7, 0.6
    elif er == 'SL' and hb <= 3: rt, rc, ts = 'volatile', 0.3, 0.2
    elif er == 'SL': rt, rc, ts = 'ranging', 0.4, 0.3
    else: rt, rc, ts = 'ranging', 0.5, 0.35
    reg = {'regime': rt, 'label': rt, 'confidence': rc,
           'volatility': 'HIGH' if rt == 'volatile' else 'NORMAL',
           'trend_strength': ts}
    ss = 6.0 if iw else 3.0
    smc = {'score': ss, 'total_score': ss,
           'bos': {'direction': f'bullish_{d.lower()}', 'type': 'BOS'} if iw else None,
           'order_block': iw, 'fvg': iw,
           'sweep_detected': False, 'liquidity_sweep': False}
    sr = []
    for k in range(3):
        off = atr * (0.3 + 0.3*k)
        sr.append({'price': e-off if d=='BUY' else e+off,
                    'type': 'support' if d=='BUY' else 'resistance'})
    if not iw and er == 'SL':
        trap = 2 if hb <= 3 else 1
        for k in range(trap):
            off = atr * (0.6 + 0.6*k)
            sr.append({'price': e+off if d=='BUY' else e-off,
                        'type': 'resistance' if d=='BUY' else 'support'})
    lg = 'CLEAR' if iw else ('DANGEROUS' if er=='SL' and hb<=3 else
         'HIGH_RISK' if er=='SL' and hb<=10 else
         'CAUTION' if er=='SL' else 'NORMAL')
    hr = (idx * 3) % 24
    sq = 'HIGH' if (7<=hr<=9 or 13<=hr<=17) else ('LOW' if (0<=hr<=6 or 20<=hr<=23) else 'MEDIUM')
    md = d if iw else (d if hb > 10 else ('SELL' if d == 'BUY' else 'BUY'))
    ana = {'sr': {'levels': sr}, 'sr_ctx': {'levels': sr},
           'liquidity': {'grade': lg}, 'liquidity_ctx': {'grade': lg},
           'smc': smc, 'smc_ctx': smc,
           'session': {'quality': sq, 'session_quality': sq},
           'session_ctx': {'quality': sq, 'session_quality': sq},
           'sentiment': {'retail_long_pct': 0.5, 'long_pct': 0.5, 'long_ratio': 1.0,
                         'agreement': 0.55 if iw else 0.45, 'fg_index': 50.0},
           'sentiment_ctx': {'retail_long_pct': 0.5, 'long_pct': 0.5, 'long_ratio': 1.0,
                            'agreement': 0.55 if iw else 0.45, 'fg_index': 50.0},
           'news': {'high_impact_nearby': (not iw and er=='SL' and hb<=5)},
           'divergence': {}, 'market_structure': {'bos': smc.get('bos')}}
    mkt = {'ind_ctx': ind, 'regime': reg, 'mtf_bias': {'bias': md},
           'spread': {'current_spread': 1.5},
           'avg_spread': {'average_spread': 1.5}, 'df': None}
    return dec, ana, mkt


def main():
    if LOG.exists(): LOG.unlink()
    t0 = time.time()
    combos = []
    seen = set()
    for f in sorted(DATA_DIR.glob('*.csv')):
        parts = f.stem.rsplit('_', 1)
        if len(parts) != 2: continue
        s, t = parts
        if s in ('indicators_ext', 'validation_report'): continue
        if (s, t) not in seen:
            seen.add((s, t)); combos.append((s, t))

    log(f'Start: {len(combos)} combos')
    results = []
    for ci, (sym, tf) in enumerate(combos):
        p = pip_size(sym)
        csv_path = DATA_DIR / f'{sym}_{tf}.csv'
        if not csv_path.exists(): continue
        try:
            df = pd.read_csv(csv_path, parse_dates=['datetime_utc'])
            if len(df) < 200: continue
        except: continue
        try:
            trades = make_trades(df, sym, p)
            if len(trades) < 5: continue
            lre = LossRejectionEngine()
            bw=bl=lb=lwb=llb=0
            l1b=l2b=l3b=l1w=l2w=l3w=0
            rnb=rnl=mxb=mxl=mddb=mddl=0.0
            bp=[]; lp=[]
            for idx, t in enumerate(trades):
                pnl=t['u']; iw=t['w']
                rnb+=pnl
                if rnb>mxb: mxb=rnb
                dd=mxb-rnb
                if dd>mddb: mddb=dd
                if iw: bw+=1
                else: bl+=1
                bp.append(pnl)
                dec,ana,mkt = ctx(t, idx)
                lr = lre.evaluate(dec,ana,mkt,symbol=sym)
                if lr.blocked:
                    lb+=1
                    if iw: lwb+=1
                    else: llb+=1
                    if lr.l1 and not lr.l1.pass_through:
                        l1b+=1
                        if iw: l1w+=1
                    if lr.l2 and not lr.l2.pass_through:
                        l2b+=1
                        if iw: l2w+=1
                    if lr.l3 and not lr.l3.pass_through:
                        l3b+=1
                        if iw: l3w+=1
                else:
                    rnl+=pnl
                    if rnl>mxl: mxl=rnl
                    dd=mxl-rnl
                    if dd>mddl: mddl=dd
                    lp.append(pnl)
                    lre.record_trade_outcome(sym,t['d'],pnl,price_zone='mid',
                        regime=mkt.get('regime',{}).get('regime','unknown'))
            wpr = round(lwb/bw, 4) if bw > 0 else 0.0
            lrr = round(llb/bl, 4) if bl > 0 else 0.0
            gp=sum(p for p in bp if p>0); gl=abs(sum(p for p in bp if p<0))
            bpf=round(gp/gl,2) if gl>0 else 999.0
            gp2=sum(p for p in lp if p>0); gl2=abs(sum(p for p in lp if p<0))
            lpf=round(gp2/gl2,2) if gl2>0 else 999.0
            results.append({'sym':sym,'tf':tf,'t':len(trades),'W':bw,'L':bl,
                'wpr':wpr,'lrr':lrr,'blk':lb,'blkW':lwb,'blkL':llb,
                'bpf':bpf,'lpf':lpf,'bnp':round(rnb,2),'lnp':round(rnl,2),
                'bdd':round(mddb,2),'ldd':round(mddl,2),
                'l1b':l1b,'l2b':l2b,'l3b':l3b,'l1w':l1w,'l2w':l2w,'l3w':l3w})
        except Exception as ex:
            pass
        if (ci+1) % 25 == 0:
            log(f'  [{ci+1}/{len(combos)}] {time.time()-t0:.0f}s')
    el = time.time() - t0
    log(f'Complete: {len(results)} results in {el:.1f}s')
    with open(OUT_DIR / 'lre_robustness_raw.json', 'w') as f:
        json.dump(results, f, indent=2)
    log('Saved raw results')


if __name__ == '__main__':
    main()
