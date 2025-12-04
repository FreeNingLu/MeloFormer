#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相关性分析可视化
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
with open('correlation_stats.json', 'r') as f:
    stats = json.load(f)

# 准备数据
same_inst_offsets = sorted([int(k) for k in stats['same_inst'].keys()])
same_inst_nmi = [stats['same_inst'][str(o)]['mean'] for o in same_inst_offsets]
same_inst_std = [stats['same_inst'][str(o)]['std'] for o in same_inst_offsets]

cross_inst_offsets = sorted([int(k) for k in stats['cross_inst'].keys()])
cross_inst_nmi = [stats['cross_inst'][str(o)]['mean'] for o in cross_inst_offsets]
cross_inst_std = [stats['cross_inst'][str(o)]['std'] for o in cross_inst_offsets]

cross_same_bar_nmi = stats['cross_inst_same_bar']['mean']

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 图1: 主要对比图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1.1 同乐器 vs 跨乐器 NMI 对比
ax1 = axes[0, 0]
ax1.plot(same_inst_offsets, same_inst_nmi, 'b-o', label='Same Instrument', linewidth=2, markersize=4)
ax1.plot(cross_inst_offsets, cross_inst_nmi, 'r-s', label='Cross Instrument', linewidth=2, markersize=4)
ax1.axhline(y=cross_same_bar_nmi, color='g', linestyle='--', label=f'Cross Inst Same Bar ({cross_same_bar_nmi:.3f})')
ax1.fill_between(same_inst_offsets,
                  [m - s for m, s in zip(same_inst_nmi, same_inst_std)],
                  [m + s for m, s in zip(same_inst_nmi, same_inst_std)],
                  alpha=0.2, color='blue')
ax1.fill_between(cross_inst_offsets,
                  [m - s for m, s in zip(cross_inst_nmi, cross_inst_std)],
                  [m + s for m, s in zip(cross_inst_nmi, cross_inst_std)],
                  alpha=0.2, color='red')
ax1.set_xlabel('Bar Offset')
ax1.set_ylabel('Normalized Mutual Information (NMI)')
ax1.set_title('Correlation: Same vs Cross Instrument')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 33)
ax1.set_ylim(0.4, 0.8)

# 标记4的倍数
for x in [4, 8, 12, 16, 24, 32]:
    if x <= 32:
        ax1.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

# 1.2 同乐器周期性分析
ax2 = axes[0, 1]
ax2.bar(same_inst_offsets, same_inst_nmi, color='steelblue', alpha=0.7)
# 标记4的倍数
colors = ['red' if o % 4 == 0 else 'steelblue' for o in same_inst_offsets]
ax2.bar(same_inst_offsets, same_inst_nmi, color=colors, alpha=0.7)
ax2.set_xlabel('Bar Offset')
ax2.set_ylabel('NMI')
ax2.set_title('Same Instrument: 4-bar Periodicity (Red = 4x)')
ax2.grid(True, alpha=0.3, axis='y')

# 1.3 跨乐器衰减曲线
ax3 = axes[1, 0]
# 加入 offset=0 (同小节)
all_cross_offsets = [0] + cross_inst_offsets
all_cross_nmi = [cross_same_bar_nmi] + cross_inst_nmi
ax3.plot(all_cross_offsets, all_cross_nmi, 'r-o', linewidth=2, markersize=5)
ax3.fill_between(all_cross_offsets,
                  [all_cross_nmi[0]] + [m - s for m, s in zip(cross_inst_nmi, cross_inst_std)],
                  [all_cross_nmi[0]] + [m + s for m, s in zip(cross_inst_nmi, cross_inst_std)],
                  alpha=0.2, color='red')
ax3.axvspan(0, 2, alpha=0.2, color='green', label='Full Connection (offset≤2)')
ax3.axvline(x=4, color='orange', linestyle='--', label='Sparse Connection (offset=4)')
ax3.set_xlabel('Bar Offset')
ax3.set_ylabel('NMI')
ax3.set_title('Cross Instrument: Decay Curve')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.5, 33)

# 1.4 掩码设计建议
ax4 = axes[1, 1]
ax4.axis('off')

# 创建表格数据
table_data = [
    ['Connection Type', 'Rule', 'NMI Range', 'Justification'],
    ['Same Instrument', 'Full Connection', '0.57-0.71', 'High correlation at all offsets'],
    ['Cross Inst (offset=0)', 'Full Connection', '0.65', 'Harmonic coordination'],
    ['Cross Inst (offset≤2)', 'Full Connection', '0.59-0.60', 'Transition/continuation'],
    ['Cross Inst (offset=4)', 'Sparse', '0.58', 'Phrase boundary'],
    ['Cross Inst (offset>4)', 'No Connection', '<0.56', 'Low, no periodicity'],
]

table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                   loc='center', cellLoc='center',
                   colWidths=[0.25, 0.2, 0.15, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# 设置表头样式
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', fontweight='bold')

ax4.set_title('Mask Design Recommendations (Based on Correlation)', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=150, bbox_inches='tight')
print('Saved: correlation_analysis.png')

# 图2: 详细热力图
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# 2.1 NMI 差异热力图 (同乐器 - 跨乐器)
ax5 = axes2[0]
diff_nmi = [s - c for s, c in zip(same_inst_nmi, cross_inst_nmi)]
colors = ['green' if d > 0.05 else 'yellow' if d > 0 else 'red' for d in diff_nmi]
bars = ax5.bar(same_inst_offsets, diff_nmi, color=colors, alpha=0.7)
ax5.axhline(y=0, color='black', linewidth=1)
ax5.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
ax5.set_xlabel('Bar Offset')
ax5.set_ylabel('NMI Difference (Same - Cross)')
ax5.set_title('Same vs Cross Instrument NMI Difference')
ax5.grid(True, alpha=0.3, axis='y')

# 添加注释
ax5.annotate('Same Inst always higher', xy=(16, 0.06), fontsize=10, ha='center')

# 2.2 连接密度估算
ax6 = axes2[1]

# 假设一首曲子有 32 小节，4 个乐器
n_bars = 32
n_instruments = 4
tokens_per_bar = 20  # 假设每个乐器每小节 20 个 token

# 全连接
full_tokens = n_bars * n_instruments * tokens_per_bar
full_connections = full_tokens * (full_tokens + 1) // 2

# 当前掩码设计
# 同乐器: 全连接 (每个乐器内部)
same_inst_conn = n_instruments * (n_bars * tokens_per_bar) * (n_bars * tokens_per_bar + 1) // 2
# 跨乐器: offset<=2 全连接 + offset=4
cross_inst_near = n_instruments * (n_instruments - 1) * n_bars * 3 * tokens_per_bar * tokens_per_bar  # 约估
cross_inst_far = n_instruments * (n_instruments - 1) * (n_bars - 4) * tokens_per_bar * tokens_per_bar

sparse_connections = same_inst_conn + cross_inst_near + cross_inst_far

labels = ['Full Attention', 'Our FC-Attention']
connections = [full_connections, sparse_connections]
savings = [0, (1 - sparse_connections / full_connections) * 100]

bars = ax6.bar(labels, [c / 1e6 for c in connections], color=['gray', 'steelblue'])
ax6.set_ylabel('Connections (Millions)')
ax6.set_title(f'Connection Density Comparison\n(n_bars={n_bars}, n_inst={n_instruments})')

# 添加节省百分比标签
for i, (bar, saving) in enumerate(zip(bars, savings)):
    if saving > 0:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'-{saving:.1f}%', ha='center', fontsize=12, fontweight='bold', color='green')

ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('correlation_detail.png', dpi=150, bbox_inches='tight')
print('Saved: correlation_detail.png')

# 图3: 周期性分析
fig3, ax7 = plt.subplots(figsize=(12, 5))

# 计算每个 offset 相对于前一个的变化
same_inst_change = [0] + [same_inst_nmi[i] - same_inst_nmi[i-1] for i in range(1, len(same_inst_nmi))]

ax7.bar(same_inst_offsets, same_inst_change, color='steelblue', alpha=0.7)
ax7.axhline(y=0, color='black', linewidth=1)

# 标记4的倍数
for x in [4, 8, 12, 16, 24, 32]:
    ax7.axvline(x=x, color='red', linestyle='--', alpha=0.5)

ax7.set_xlabel('Bar Offset')
ax7.set_ylabel('NMI Change from Previous Offset')
ax7.set_title('Same Instrument: NMI Change Pattern (Red lines = 4x offsets)')
ax7.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_periodicity.png', dpi=150, bbox_inches='tight')
print('Saved: correlation_periodicity.png')

plt.show()
print('\nDone!')
