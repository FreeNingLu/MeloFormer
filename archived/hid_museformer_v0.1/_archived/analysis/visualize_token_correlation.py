#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token 相关性可视化
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
with open('token_correlation_stats.json', 'r') as f:
    stats = json.load(f)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
fig = plt.figure(figsize=(16, 12))

# 1. 同一音符内 Token 相关性
ax1 = fig.add_subplot(2, 2, 1)
same_note = stats['same_note']
pairs = list(same_note.keys())
values = [same_note[p]['mean'] for p in pairs]
colors = ['#e74c3c' if v > 0.2 else '#3498db' if v > 0.15 else '#95a5a6' for v in values]
bars = ax1.bar(pairs, values, color=colors)
ax1.set_ylabel('NMI')
ax1.set_title('Same Note: Token Correlation\n(T=Time, P=Pitch, D=Duration, V=Velocity)')
ax1.set_ylim(0, 0.35)
ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='High (>0.2)')
ax1.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Medium (>0.15)')
ax1.legend(loc='upper right')
for bar, v in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.2f}',
             ha='center', va='bottom', fontsize=9)

# 2. 相邻音符间 Token 相关性热力图
ax2 = fig.add_subplot(2, 2, 2)
token_types = ['T', 'P', 'D', 'V']
adjacent = stats['adjacent_note']
matrix = np.zeros((4, 4))
for i, t1 in enumerate(token_types):
    for j, t2 in enumerate(token_types):
        key = f'{t1}→{t2}'
        if key in adjacent:
            matrix[i, j] = adjacent[key]['mean']

im = ax2.imshow(matrix, cmap='RdYlBu_r', vmin=0, vmax=0.6)
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels([f'{t}\n(current)' for t in token_types])
ax2.set_yticklabels([f'{t}\n(prev)' for t in token_types])
ax2.set_title('Adjacent Note: Token Correlation\n(Previous → Current)')
plt.colorbar(im, ax=ax2, label='NMI')

# 添加数值标签
for i in range(4):
    for j in range(4):
        text = ax2.text(j, i, f'{matrix[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=10)

# 3. 同小节 vs 跨小节对比
ax3 = fig.add_subplot(2, 2, 3)
same_bar = stats['same_bar']
cross_bar = stats['cross_bar']

# 选择关键配对
key_pairs = ['T→T', 'P→P', 'D→D', 'V→V', 'T→P', 'P→T']
x = np.arange(len(key_pairs))
width = 0.35

same_bar_vals = [same_bar[k]['mean'] if k in same_bar else 0 for k in key_pairs]
cross_bar_vals = [cross_bar[k]['mean'] if k in cross_bar else 0 for k in key_pairs]

bars1 = ax3.bar(x - width/2, same_bar_vals, width, label='Same Bar', color='#2ecc71')
bars2 = ax3.bar(x + width/2, cross_bar_vals, width, label='Cross Bar', color='#e74c3c')

ax3.set_ylabel('NMI')
ax3.set_title('Same Bar vs Cross Bar Token Correlation')
ax3.set_xticks(x)
ax3.set_xticklabels(key_pairs)
ax3.legend()
ax3.set_ylim(0, 0.7)

# 添加差异百分比
for i, (s, c) in enumerate(zip(same_bar_vals, cross_bar_vals)):
    if c > 0:
        diff = (s - c) / c * 100
        color = 'green' if diff > 0 else 'red'
        ax3.text(i, max(s, c) + 0.02, f'{diff:+.0f}%', ha='center', fontsize=8, color=color)

# 4. 优化建议表
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

# 分析结果
analysis = """
Token Correlation Analysis Results
==================================

Key Findings:
1. T→T (Time to Time): NMI=0.56 (adjacent), 0.64 (same bar)
   → Strong rhythmic pattern, TIME tokens should see TIME tokens

2. P→P (Pitch to Pitch): NMI=0.47 (adjacent), 0.53 (same bar)
   → Strong melodic pattern, PITCH tokens should see PITCH tokens

3. T→P, P→T: NMI=0.23-0.26
   → Moderate cross-type correlation

4. V (Velocity): Low correlation with others (<0.15)
   → Velocity tokens less important for attention

Optimization Recommendations:
=============================
┌────────────────┬───────────────────────────────┐
│ Query Token    │ Key Tokens to Attend          │
├────────────────┼───────────────────────────────┤
│ T (Time)       │ All T tokens (high priority)  │
│                │ Chord tokens                  │
├────────────────┼───────────────────────────────┤
│ P (Pitch)      │ Same-note T (必须)            │
│                │ All P tokens (high priority)  │
│                │ Chord tokens                  │
├────────────────┼───────────────────────────────┤
│ D (Duration)   │ Same-note T, P                │
│                │ Previous D tokens             │
├────────────────┼───────────────────────────────┤
│ V (Velocity)   │ Same-note T, P                │
│                │ Can be sparse                 │
└────────────────┴───────────────────────────────┘

Potential Savings: ~20-30% attention reduction
(by making V tokens sparse, reducing cross-type attention)
"""

ax4.text(0.05, 0.95, analysis, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('token_correlation_analysis.png', dpi=150, bbox_inches='tight')
print('Saved: token_correlation_analysis.png')

# 图2: 详细热力图
fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (category, title) in enumerate([
    ('adjacent_note', 'Adjacent Note'),
    ('same_bar', 'Same Bar'),
    ('cross_bar', 'Cross Bar')
]):
    ax = axes[idx]
    data = stats[category]
    matrix = np.zeros((4, 4))
    for i, t1 in enumerate(token_types):
        for j, t2 in enumerate(token_types):
            key = f'{t1}→{t2}'
            if key in data:
                matrix[i, j] = data[key]['mean']

    im = ax.imshow(matrix, cmap='RdYlBu_r', vmin=0, vmax=0.65)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(token_types)
    ax.set_yticklabels(token_types)
    ax.set_xlabel('Current Token')
    ax.set_ylabel('Previous Token')
    ax.set_title(f'{title}')

    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{matrix[i, j]:.2f}',
                   ha='center', va='center', color='black', fontsize=10, fontweight='bold')

plt.colorbar(im, ax=axes, label='NMI', shrink=0.8)
plt.suptitle('Token Type Correlation Heatmaps', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('token_correlation_heatmaps.png', dpi=150, bbox_inches='tight')
print('Saved: token_correlation_heatmaps.png')

plt.show()
print('\nDone!')
