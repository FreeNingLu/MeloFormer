#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token 跨类型相关性可视化 (按小节距离)
"""

import json
import matplotlib.pyplot as plt
import numpy as np

with open('token_cross_type_by_bar.json', 'r') as f:
    stats = json.load(f)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 同类型按距离衰减
ax1 = axes[0, 0]
offsets = list(range(1, 17))
token_types = ['T', 'P', 'D', 'V']
colors = {'T': '#3498db', 'P': '#e74c3c', 'D': '#2ecc71', 'V': '#9b59b6'}
markers = {'T': 'o', 'P': 's', 'D': '^', 'V': 'v'}

for t in token_types:
    key = f'{t}→{t}'
    values = [stats['by_offset'][str(o)].get(key, 0) for o in offsets]
    ax1.plot(offsets, values, f'-{markers[t]}', color=colors[t],
             linewidth=2, markersize=6, label=f'{t}→{t}')

ax1.set_xlabel('Bar Offset (小节距离)', fontsize=12)
ax1.set_ylabel('NMI', fontsize=12)
ax1.set_title('Same-Type Token Correlation by Bar Distance', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xticks([1, 4, 8, 12, 16])
ax1.set_ylim(0, 0.25)

# 标记周期性
for x in [4, 8, 16]:
    ax1.axvline(x=x, color='gray', linestyle='--', alpha=0.5)

# 2. 跨类型 T→X 对比
ax2 = axes[0, 1]
cross_pairs = ['T→P', 'T→D', 'T→V', 'P→D', 'P→V', 'D→V']
cross_colors = ['#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c', '#7f8c8d']

for pair, color in zip(cross_pairs[:3], cross_colors[:3]):  # T→X
    values = [stats['by_offset'][str(o)].get(pair, 0) for o in offsets]
    ax2.plot(offsets, values, '-o', color=color, linewidth=2, markersize=5, label=pair)

# 添加 T→T 作为参考
tt_values = [stats['by_offset'][str(o)].get('T→T', 0) for o in offsets]
ax2.plot(offsets, tt_values, '--', color='#3498db', linewidth=2, alpha=0.5, label='T→T (ref)')

ax2.set_xlabel('Bar Offset', fontsize=12)
ax2.set_ylabel('NMI', fontsize=12)
ax2.set_title('Cross-Type Correlation: T→X', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xticks([1, 4, 8, 12, 16])

# 3. 同小节内跨类型相关性
ax3 = axes[1, 0]
same_bar = stats['same_bar_cross_type']
pairs = sorted(same_bar.keys())
values = [same_bar[p] for p in pairs]
colors_bar = ['#e74c3c' if v > 0.1 else '#f39c12' if v > 0.08 else '#95a5a6' for v in values]

bars = ax3.bar(pairs, values, color=colors_bar)
ax3.set_ylabel('NMI', fontsize=12)
ax3.set_title('Same-Bar Cross-Type Correlation', fontsize=14, fontweight='bold')
ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='High (>0.1)')
ax3.axhline(y=0.08, color='orange', linestyle='--', alpha=0.5, label='Medium (>0.08)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3, axis='y')

for bar, v in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{v:.3f}', ha='center', fontsize=9)

# 4. 跨类型 / 同类型 比例
ax4 = axes[1, 1]
ax4.axis('off')

conclusion = """
Token Cross-Type Correlation Analysis (by Bar Distance)
========================================================

Key Findings:
1. Same-Type correlation is MUCH HIGHER than Cross-Type
   - T→T: 0.19 (avg offset 1-4)
   - P→P: 0.14
   - T→P: 0.095 (only 49% of T→T)
   - T→D: 0.069 (only 36% of T→T)
   - T→V: 0.066 (only 34% of T→T)

2. Cross-Type correlation is relatively FLAT across distances
   - T→P: 0.107 (offset=1) → 0.080 (offset=16)
   - Decay is slower than Same-Type

3. Periodicity in Same-Type (4-bar structure):
   - T→T peaks at offset=4, 8, 16
   - Cross-Type shows NO periodicity

4. Duration (D) and Velocity (V) correlations are lowest:
   - D→V: 0.056 (can be ignored)
   - P→V, T→V: ~0.05-0.07

Optimization Recommendations:
=============================
┌──────────────────────────────────────────────────────────┐
│ Token Type │ Connection Strategy                         │
├──────────────────────────────────────────────────────────┤
│ T→T, P→P   │ FULL connection (high NMI ~0.14-0.19)      │
│            │ Essential for rhythm/melody patterns        │
├──────────────────────────────────────────────────────────┤
│ T→P, P→T   │ SPARSE (~50%) for offset > 4               │
│            │ Keep full for same bar & near bars          │
├──────────────────────────────────────────────────────────┤
│ T→D, P→D   │ SPARSE (~30%) or same-bar only             │
│ T→V, P→V   │ Low correlation, sparse is safe            │
├──────────────────────────────────────────────────────────┤
│ D→V, V→D   │ MINIMAL or REMOVE                          │
│            │ Very low correlation (0.056)                │
└──────────────────────────────────────────────────────────┘

Estimated Additional Savings: ~25-35%
(by making low-correlation cross-type pairs sparse)
"""

ax4.text(0.02, 0.98, conclusion, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('token_cross_type_by_bar.png', dpi=150, bbox_inches='tight')
print('Saved: token_cross_type_by_bar.png')

plt.show()
