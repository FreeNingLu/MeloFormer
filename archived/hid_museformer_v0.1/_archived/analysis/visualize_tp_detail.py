#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T↔P 详细相关性可视化
"""

import json
import matplotlib.pyplot as plt
import numpy as np

with open('token_tp_detail_stats.json', 'r') as f:
    stats = json.load(f)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 按距离衰减
ax1 = axes[0, 0]
distances = list(range(1, 11))
tp = [stats['by_distance'][str(d)]['T_to_P'] for d in distances]
pt = [stats['by_distance'][str(d)]['P_to_T'] for d in distances]
tt = [stats['by_distance'][str(d)]['T_to_T'] for d in distances]
pp = [stats['by_distance'][str(d)]['P_to_P'] for d in distances]

ax1.plot(distances, tt, 'b-o', linewidth=2, markersize=8, label='T→T (同类型)')
ax1.plot(distances, pp, 'r-s', linewidth=2, markersize=8, label='P→P (同类型)')
ax1.plot(distances, tp, 'g-^', linewidth=2, markersize=8, label='T→P (跨类型)')
ax1.plot(distances, pt, 'm-v', linewidth=2, markersize=8, label='P→T (跨类型)')

ax1.set_xlabel('Note Distance (音符距离)', fontsize=12)
ax1.set_ylabel('NMI', fontsize=12)
ax1.set_title('Token Correlation by Note Distance', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(distances)
ax1.set_ylim(0, 0.65)

# 添加衰减百分比注释
ax1.annotate(f'T→P 衰减: {(1-tp[-1]/tp[0])*100:.0f}%',
             xy=(10, tp[-1]), xytext=(8, 0.15),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, color='green')

# 2. 同类型 vs 跨类型对比
ax2 = axes[0, 1]
x = np.arange(len(distances))
width = 0.2

bars1 = ax2.bar(x - 1.5*width, tt, width, label='T→T', color='#3498db')
bars2 = ax2.bar(x - 0.5*width, pp, width, label='P→P', color='#e74c3c')
bars3 = ax2.bar(x + 0.5*width, tp, width, label='T→P', color='#2ecc71')
bars4 = ax2.bar(x + 1.5*width, pt, width, label='P→T', color='#9b59b6')

ax2.set_xlabel('Note Distance', fontsize=12)
ax2.set_ylabel('NMI', fontsize=12)
ax2.set_title('Same-Type vs Cross-Type Correlation', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(distances)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

# 3. T→P 相对于 T→T 的比例
ax3 = axes[1, 0]
ratio = [tp[i] / tt[i] * 100 for i in range(len(distances))]
colors = ['#e74c3c' if r < 40 else '#f39c12' if r < 50 else '#2ecc71' for r in ratio]
bars = ax3.bar(distances, ratio, color=colors)

ax3.set_xlabel('Note Distance', fontsize=12)
ax3.set_ylabel('T→P / T→T (%)', fontsize=12)
ax3.set_title('Cross-Type as % of Same-Type (T→P / T→T)', fontsize=14, fontweight='bold')
ax3.axhline(y=50, color='orange', linestyle='--', label='50% threshold')
ax3.axhline(y=40, color='red', linestyle='--', label='40% threshold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(distances)

for bar, r in zip(bars, ratio):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{r:.0f}%', ha='center', fontsize=9)

# 4. 结论和建议
ax4 = axes[1, 1]
ax4.axis('off')

conclusion = """
T→P, P→T Cross-Type Token Correlation Analysis
================================================

Key Findings:
1. Cross-type (T→P, P→T) correlation is MUCH LOWER than same-type
   - T→T: 0.59 → 0.34 (distance 1→10)
   - P→P: 0.45 → 0.30 (distance 1→10)
   - T→P: 0.23 → 0.19 (distance 1→10)

2. T→P is only 38-55% of T→T correlation
   - This means cross-type attention is LESS important

3. Cross-type correlation decays slowly (-18.8% over 10 notes)
   - T→T decays faster (-42.3%)
   - T→P has more "baseline" noise

4. Same bar vs Cross bar (adjacent notes):
   - T→P same bar: 0.25
   - T→P cross bar: 0.28 (actually higher!)
   - P→T cross bar: 0.17 (lower)

Optimization Recommendations:
=============================
┌─────────────────────────────────────────────────────┐
│ For SAME instrument, DIFFERENT bars:                │
│   • T→T, P→P: KEEP full connection (high NMI)      │
│   • T→P, P→T: Can be SPARSE (low NMI ~0.2)         │
│                                                     │
│ For CROSS instrument:                               │
│   • T→P, P→T: Can be REMOVED (very low benefit)    │
│                                                     │
│ Within SAME note:                                   │
│   • T↔P: MUST connect (structural necessity)       │
└─────────────────────────────────────────────────────┘

Estimated Additional Savings: ~15-20%
(on top of existing bar-level optimization)
"""

ax4.text(0.02, 0.98, conclusion, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('token_tp_detail_analysis.png', dpi=150, bbox_inches='tight')
print('Saved: token_tp_detail_analysis.png')

plt.show()
