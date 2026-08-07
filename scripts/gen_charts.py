import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = '/home/z/my-project/forex-agent/download'
CHARTS = OUT + '/lre_charts'
CHARTS.mkdir(exist_ok=True)

with open(OUT + '/lre_robustness_report.json') as f:
    report = json.load(f)
R = report['all_results']
agg = report['aggregate']
meta = report['meta']
v = report['verdict']
f = report['failures']

print('Charts...', flush=True)

# Chart 1: WPR histogram
wprs = [r['wpr'] for r in R]
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
bins = np.arange(0, 1.05, 0.05)
ax.hist(wprs, bins=bins, color=(0.54, 0.45, 0.15), edgecolor='white', alpha=0.85)
ax.axvline(0.80, color=(0.725, 0.11, 0.11), linestyle='--', linewidth=2, label='Target (80%)')
ax.axvline(np.mean(wprs), color='navy', linestyle=':', linewidth=1.5, label=f'Mean ({np.mean(wprs):.1%})')
ax.set_xlabel('Winner Preservation Rate', fontsize=10)
ax.set_ylabel('Number of Combinations', fontsize=10)
ax.legend(fontsize=9)
ax.set_title('WPR Distribution Across All 144 Symbol/Timeframe Combinations', fontsize=11)
plt.savefig(CHARTS + '/wpr_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print('  wpr_distribution.png', flush=True)

# Chart 2: WPR by Timeframe
tf_data = report['timeframe']
labels = list(tf_data.keys())
means = [tf_data[t]['wpr']['mean'] for t in labels]
meds = [tf_data[t]['wpr']['median'] for t in labels]
stds = [tf_data[t]['wpr']['std'] for t in labels]
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
x = np.arange(len(labels))
ax.bar(x, means, 0.5, yerr=stds, color=[(0.54,0.45,0.15),(0.85,0.47,0.02),(0.725,0.11,0.11)], edgecolor='white', alpha=0.85, capsize=3)
ax.plot(x, meds, 'D', color='navy', markersize=8, label='Median', zorder=5)
ax.axhline(0.80, color=(0.725,0.11,0.11), linestyle='--', linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('WPR', fontsize=10)
ax.set_title('WPR by Timeframe', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
plt.savefig(CHARTS + '/wpr_by_timeframe.png', dpi=200, bbox_inches='tight')
plt.close()
print('  wpr_by_timeframe.png', flush=True)

# Chart 3: WPR by Category
cat_data = report['category']
cats = sorted(cat_data.keys(), key=lambda c: cat_data[c]['wpr']['mean'])
vals = [cat_data[c]['wpr']['mean'] for c in cats]
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
colors = [(0.27,0.51,0.35) if v >= 0.8 else ((0.85,0.47,0.02) if v >= 0.5 else (0.725,0.11,0.11)) for v in vals]
ax.barh(names, vals, color=ncolors, edgecolor='white', height=0.6)
ax.axvline(0.80, color=(0.725,0.11,0.11), linestyle='--', linewidth=1.5)
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontsize=8)
ax.set_xlabel('WPR', fontsize=10)
ax.set_title('WPR by Symbol Category', fontsize=11)
ax.set_xlim(0, 1.1)
plt.savefig(CHARTS + '/wpr_by_category.png', dpi=200, bbox_inches='tight')
plt.close()
print('  wpr_by_category.png', flush=True)

# Chart 4: LRR by Timeframe
labels = list(tf_data.keys())
means = [tf_data[t]['lrr']['mean'] for t in labels]
stds = [tf_data[t]['lrr']['std'] for t in labels]
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
nx = np.arange(len(labels))
ax.bar(x, means, 0.5, yerr=stds, color=[(0.27,0.51,0.35),(0.54,0.45,0.15),(0.85,0.47,0.02)], edgecolor='white', alpha=0.85, capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('LRR', fontsize=10)
ax.set_title('LRR by Timeframe', fontsize=11)
ax.set_ylim(0, 1.1)
plt.savefig(CHARTS + '/lrr_by_timeframe.png', dpi=200, bbox_inches='tight')
plt.close()
print('  lrr_by_timeframe.png', flush=True)

# Chart 5: Precision/Recall pie
f = report['failures']
tp = f['tp_total']; fp = f['fp_total']
fn = sum(r['L']-r['blkL'] for r in R)
fw = sum(r['W'] for r in R)
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
data = {'True Pos': tp, 'False Pos': fp, 'False Neg': fn, 'True Neg': fw}
labels = list(data.keys())
sizes = [data[l] for l in labels]
clrs = [(0.27,0.51,0.35),(0.725,0.11,0.11),(0.85,0.47,0.02),(0.58,0.63,0.39)]
wedges = [0.01]*4
nwedges[1] = 0.05
nwedges[0] = 0.05
ax.pie(sizes, labels=labels, colors=clrs, wedgeprops=nwedges, autopct='%1.1f%%', textprops={'fontsize': 9}, startangle=90)
ax.set_title('Classification Breakdown (All 144 Combinations)', fontsize=11)
prec = tp/(tp+fp) if tp+fp > 0 else 0
rec = tp/(tp+fn) if tp+fn > 0 else 0
ax.text(0, -1.3, f'Precision: {prec:.1%}  |  Recall: {rec:.1%}', fontsize=10, ha='center')
plt.savefig(CHARTS + '/precision_recall.png', dpi=200, bbox_inches='tight')
plt.close()
print('  precision_recall.png', flush=True)
print(f'All charts saved to {CHARTS}/')