#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相似度分布图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_stats(path='similarity_stats.json'):
    with open(path, 'r') as f:
        return json.load(f)


def plot_similarity_distribution_by_offset(stats, output_path='distribution_by_offset.png'):
    """
    按 offset 分组的相似度分布图
    """
    same_inst = stats['same_inst']
    cross_diff = stats['cross_inst_diff_bar']

    # 聚合跨乐器
    cross_by_offset = {}
    for pair_str, offset_data in cross_diff.items():
        for offset_str, data in offset_data.items():
            offset = int(offset_str)
            if offset not in cross_by_offset:
                cross_by_offset[offset] = []
            cross_by_offset[offset].append(data['pitch_class']['mean'])

    # 选择关键 offset
    key_offsets = [1, 2, 4, 8, 16, 32]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, offset in enumerate(key_offsets):
        ax = axes[idx]

        # 同乐器数据
        same_val = same_inst.get(str(offset), {}).get('pitch_class', {}).get('mean', 0)
        same_std = same_inst.get(str(offset), {}).get('pitch_class', {}).get('std', 0)
        same_count = same_inst.get(str(offset), {}).get('pitch_class', {}).get('count', 0)

        # 跨乐器数据
        cross_vals = cross_by_offset.get(offset, [])
        cross_mean = np.mean(cross_vals) if cross_vals else 0
        cross_std = np.std(cross_vals) if cross_vals else 0

        # 绘制柱状图 + 误差棒
        x = [0, 1]
        means = [same_val, cross_mean]
        stds = [same_std, cross_std]
        colors = ['steelblue', 'coral']

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(['Same\nInstrument', 'Cross\nInstrument'], fontsize=11)
        ax.set_ylabel('Pitch Class Similarity', fontsize=11)
        ax.set_title(f'Offset = {offset} bars', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 0.7)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.02,
                   f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=10)

        # 添加差值
        diff = same_val - cross_mean
        ax.annotate(f'Δ = {diff:.3f}', xy=(0.5, 0.65), xycoords='axes fraction',
                   fontsize=12, ha='center', color='green' if diff > 0.15 else 'orange',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Similarity Distribution by Bar Offset\n(Error bars = Standard Deviation)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_similarity_curve_with_confidence(stats, output_path='distribution_curve_confidence.png'):
    """
    带置信区间的相似度曲线
    """
    same_inst = stats['same_inst']
    cross_diff = stats['cross_inst_diff_bar']

    # 聚合跨乐器
    cross_by_offset = {}
    for pair_str, offset_data in cross_diff.items():
        for offset_str, data in offset_data.items():
            offset = int(offset_str)
            if offset not in cross_by_offset:
                cross_by_offset[offset] = {'mean': [], 'std': []}
            cross_by_offset[offset]['mean'].append(data['pitch_class']['mean'])
            cross_by_offset[offset]['std'].append(data['pitch_class']['std'])

    offsets = list(range(1, 33))

    # 同乐器数据
    same_mean = []
    same_std = []
    for o in offsets:
        if str(o) in same_inst:
            same_mean.append(same_inst[str(o)]['pitch_class']['mean'])
            same_std.append(same_inst[str(o)]['pitch_class']['std'])
        else:
            same_mean.append(0)
            same_std.append(0)

    # 跨乐器数据
    cross_mean = []
    cross_std = []
    for o in offsets:
        if o in cross_by_offset:
            cross_mean.append(np.mean(cross_by_offset[o]['mean']))
            cross_std.append(np.mean(cross_by_offset[o]['std']))  # 平均标准差
        else:
            cross_mean.append(0)
            cross_std.append(0)

    same_mean = np.array(same_mean)
    same_std = np.array(same_std)
    cross_mean = np.array(cross_mean)
    cross_std = np.array(cross_std)

    fig, ax = plt.subplots(figsize=(16, 8))

    # 绘制置信区间
    ax.fill_between(offsets, same_mean - same_std, same_mean + same_std,
                   alpha=0.3, color='steelblue', label='Same Inst ±1σ')
    ax.fill_between(offsets, cross_mean - cross_std, cross_mean + cross_std,
                   alpha=0.3, color='coral', label='Cross Inst ±1σ')

    # 绘制均值曲线
    ax.plot(offsets, same_mean, 'b-o', linewidth=2, markersize=8, label='Same Instrument Mean')
    ax.plot(offsets, cross_mean, 'r-s', linewidth=2, markersize=8, label='Cross Instrument Mean')

    # 标记 4 的倍数
    for o in [4, 8, 12, 16, 24, 32]:
        ax.axvline(x=o, color='green', linestyle='--', alpha=0.5)

    ax.set_xlabel('Bar Offset', fontsize=14)
    ax.set_ylabel('Pitch Class Similarity', fontsize=14)
    ax.set_title('Similarity Curves with Confidence Intervals (±1 Std Dev)\nGreen dashed lines = multiples of 4 bars', fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 32.5)
    ax.set_ylim(0, 0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_periodicity_highlight(stats, output_path='distribution_periodicity.png'):
    """
    周期性高亮图：突出显示 4 的倍数位置
    """
    same_inst = stats['same_inst']

    offsets = list(range(1, 33))
    pitch_class = [same_inst.get(str(o), {}).get('pitch_class', {}).get('mean', 0) for o in offsets]

    # 区分 4 的倍数和其他
    is_multiple_of_4 = [o % 4 == 0 for o in offsets]
    is_1_or_2 = [o in [1, 2] for o in offsets]

    colors = []
    for i, o in enumerate(offsets):
        if o in [1, 2]:
            colors.append('purple')  # 短期相关
        elif o % 4 == 0:
            colors.append('green')   # 4 的倍数
        else:
            colors.append('lightgray')  # 其他

    fig, ax = plt.subplots(figsize=(16, 8))

    bars = ax.bar(offsets, pitch_class, color=colors, edgecolor='black', alpha=0.8)

    # 添加平均线
    avg_multiple_4 = np.mean([pitch_class[o-1] for o in [4, 8, 12, 16, 24, 32]])
    avg_others = np.mean([pitch_class[o-1] for o in offsets if o not in [1, 2, 4, 8, 12, 16, 24, 32]])

    ax.axhline(y=avg_multiple_4, color='green', linestyle='--', linewidth=2,
              label=f'Avg (4x): {avg_multiple_4:.4f}')
    ax.axhline(y=avg_others, color='gray', linestyle='--', linewidth=2,
              label=f'Avg (others): {avg_others:.4f}')

    ax.set_xlabel('Bar Offset', fontsize=14)
    ax.set_ylabel('Pitch Class Similarity', fontsize=14)
    ax.set_title('Same Instrument Similarity: Periodicity Analysis\nPurple = Short-term (1,2), Green = Multiples of 4, Gray = Others', fontsize=16)
    ax.set_xticks(offsets)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.65)

    # 添加数值标签
    for bar, val, o in zip(bars, pitch_class, offsets):
        if o in [1, 2, 4, 8, 12, 16, 24, 32]:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_same_vs_cross_scatter(stats, output_path='distribution_scatter.png'):
    """
    同乐器 vs 跨乐器散点图
    """
    same_inst = stats['same_inst']
    cross_diff = stats['cross_inst_diff_bar']

    # 聚合跨乐器
    cross_by_offset = {}
    for pair_str, offset_data in cross_diff.items():
        for offset_str, data in offset_data.items():
            offset = int(offset_str)
            if offset not in cross_by_offset:
                cross_by_offset[offset] = []
            cross_by_offset[offset].append(data['pitch_class']['mean'])

    offsets = list(range(1, 33))

    same_vals = [same_inst.get(str(o), {}).get('pitch_class', {}).get('mean', 0) for o in offsets]
    cross_vals = [np.mean(cross_by_offset.get(o, [0])) for o in offsets]

    fig, ax = plt.subplots(figsize=(10, 10))

    # 区分颜色
    colors = []
    markers = []
    for o in offsets:
        if o in [1, 2]:
            colors.append('purple')
        elif o % 4 == 0:
            colors.append('green')
        else:
            colors.append('gray')

    scatter = ax.scatter(cross_vals, same_vals, c=colors, s=150, edgecolors='black', alpha=0.8)

    # 添加标签
    for i, o in enumerate(offsets):
        ax.annotate(str(o), (cross_vals[i], same_vals[i]), fontsize=9,
                   ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

    # 对角线 (y = x)
    ax.plot([0, 0.6], [0, 0.6], 'k--', alpha=0.5, label='y = x (equal)')

    # 拟合线
    z = np.polyfit(cross_vals, same_vals, 1)
    p = np.poly1d(z)
    ax.plot([0, 0.35], [p(0), p(0.35)], 'r-', linewidth=2,
           label=f'Linear fit: y = {z[0]:.2f}x + {z[1]:.2f}')

    ax.set_xlabel('Cross Instrument Similarity', fontsize=14)
    ax.set_ylabel('Same Instrument Similarity', fontsize=14)
    ax.set_title('Same vs Cross Instrument Similarity by Offset\nGreen = 4x, Purple = 1,2, Gray = Others', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 0.35)
    ax.set_ylim(0.3, 0.6)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_difference_distribution(stats, output_path='distribution_difference.png'):
    """
    同乐器与跨乐器的差值分布
    """
    same_inst = stats['same_inst']
    cross_diff = stats['cross_inst_diff_bar']

    # 聚合跨乐器
    cross_by_offset = {}
    for pair_str, offset_data in cross_diff.items():
        for offset_str, data in offset_data.items():
            offset = int(offset_str)
            if offset not in cross_by_offset:
                cross_by_offset[offset] = []
            cross_by_offset[offset].append(data['pitch_class']['mean'])

    offsets = list(range(1, 33))

    same_vals = [same_inst.get(str(o), {}).get('pitch_class', {}).get('mean', 0) for o in offsets]
    cross_vals = [np.mean(cross_by_offset.get(o, [0])) for o in offsets]
    diff_vals = [s - c for s, c in zip(same_vals, cross_vals)]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 上图：差值柱状图
    ax1 = axes[0]
    colors = ['green' if d > 0.2 else 'orange' if d > 0.15 else 'red' for d in diff_vals]
    bars = ax1.bar(offsets, diff_vals, color=colors, edgecolor='black', alpha=0.8)

    ax1.axhline(y=np.mean(diff_vals), color='blue', linestyle='--', linewidth=2,
               label=f'Average difference: {np.mean(diff_vals):.4f}')
    ax1.axhline(y=0.2, color='green', linestyle=':', alpha=0.7, label='High threshold (0.20)')
    ax1.axhline(y=0.15, color='orange', linestyle=':', alpha=0.7, label='Medium threshold (0.15)')

    ax1.set_xlabel('Bar Offset', fontsize=12)
    ax1.set_ylabel('Similarity Difference\n(Same - Cross)', fontsize=12)
    ax1.set_title('Difference between Same and Cross Instrument Similarity\nGreen = High benefit, Orange = Medium, Red = Low', fontsize=14)
    ax1.set_xticks(offsets)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 下图：累积分布
    ax2 = axes[1]
    sorted_diff = np.sort(diff_vals)
    cumulative = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)

    ax2.plot(sorted_diff, cumulative, 'b-o', linewidth=2, markersize=8)
    ax2.axvline(x=np.mean(diff_vals), color='red', linestyle='--', label=f'Mean: {np.mean(diff_vals):.4f}')
    ax2.axvline(x=np.median(diff_vals), color='green', linestyle='--', label=f'Median: {np.median(diff_vals):.4f}')

    ax2.set_xlabel('Similarity Difference (Same - Cross)', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Cumulative Distribution of Similarity Difference', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comprehensive_distribution(stats, output_path='distribution_comprehensive.png'):
    """
    综合分布图：一图展示所有关键信息
    """
    same_inst = stats['same_inst']
    cross_diff = stats['cross_inst_diff_bar']

    # 聚合跨乐器
    cross_by_offset = {}
    for pair_str, offset_data in cross_diff.items():
        for offset_str, data in offset_data.items():
            offset = int(offset_str)
            if offset not in cross_by_offset:
                cross_by_offset[offset] = {'pc': [], 'rhythm': []}
            cross_by_offset[offset]['pc'].append(data['pitch_class']['mean'])
            cross_by_offset[offset]['rhythm'].append(data['rhythm']['mean'])

    offsets = list(range(1, 33))

    # 数据
    same_pc = [same_inst.get(str(o), {}).get('pitch_class', {}).get('mean', 0) for o in offsets]
    same_rhythm = [same_inst.get(str(o), {}).get('rhythm', {}).get('mean', 0) for o in offsets]
    cross_pc = [np.mean(cross_by_offset.get(o, {'pc': [0]})['pc']) for o in offsets]
    cross_rhythm = [np.mean(cross_by_offset.get(o, {'rhythm': [0]})['rhythm']) for o in offsets]

    fig = plt.figure(figsize=(20, 16))

    # 创建 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. 主曲线图 (占据左上 2x2)
    ax_main = fig.add_subplot(gs[0:2, 0:2])

    ax_main.fill_between(offsets, same_pc, alpha=0.3, color='steelblue')
    ax_main.fill_between(offsets, cross_pc, alpha=0.3, color='coral')
    ax_main.plot(offsets, same_pc, 'b-o', linewidth=2.5, markersize=10, label='Same Inst - Pitch Class')
    ax_main.plot(offsets, cross_pc, 'r-s', linewidth=2.5, markersize=10, label='Cross Inst - Pitch Class')
    ax_main.plot(offsets, same_rhythm, 'b--^', linewidth=1.5, markersize=6, alpha=0.7, label='Same Inst - Rhythm')
    ax_main.plot(offsets, cross_rhythm, 'r--v', linewidth=1.5, markersize=6, alpha=0.7, label='Cross Inst - Rhythm')

    # 标记关键位置
    for o in [1, 2, 4, 8, 12, 16, 24, 32]:
        ax_main.axvline(x=o, color='green' if o % 4 == 0 else 'purple',
                       linestyle='--', alpha=0.4, linewidth=1.5)

    ax_main.set_xlabel('Bar Offset', fontsize=14)
    ax_main.set_ylabel('Jaccard Similarity', fontsize=14)
    ax_main.set_title('Similarity Distribution: Same vs Cross Instrument\n(Purple = offset 1,2; Green = multiples of 4)', fontsize=16)
    ax_main.legend(fontsize=11, loc='upper right')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(0.5, 32.5)
    ax_main.set_ylim(0, 0.8)

    # 2. 差值图 (右上)
    ax_diff = fig.add_subplot(gs[0, 2])
    diff_pc = [s - c for s, c in zip(same_pc, cross_pc)]
    colors = ['green' if o in [1, 2, 4, 8, 12, 16, 24, 32] else 'lightgray' for o in offsets]
    ax_diff.bar(offsets, diff_pc, color=colors, edgecolor='black', alpha=0.8)
    ax_diff.axhline(y=np.mean(diff_pc), color='red', linestyle='--')
    ax_diff.set_xlabel('Bar Offset', fontsize=11)
    ax_diff.set_ylabel('Difference', fontsize=11)
    ax_diff.set_title('Same - Cross\n(Pitch Class)', fontsize=12)
    ax_diff.set_xticks([1, 4, 8, 16, 32])
    ax_diff.grid(True, alpha=0.3, axis='y')

    # 3. 箱线图对比 (右中)
    ax_box = fig.add_subplot(gs[1, 2])
    box_data = [same_pc, cross_pc]
    bp = ax_box.boxplot(box_data, labels=['Same Inst', 'Cross Inst'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    ax_box.set_ylabel('Pitch Class Similarity', fontsize=11)
    ax_box.set_title('Distribution Comparison', fontsize=12)
    ax_box.grid(True, alpha=0.3, axis='y')

    # 4. 关键 offset 柱状图 (左下)
    ax_key = fig.add_subplot(gs[2, 0])
    key_offsets = [1, 2, 4, 8, 16, 32]
    key_same = [same_pc[o-1] for o in key_offsets]
    key_cross = [cross_pc[o-1] for o in key_offsets]

    x = np.arange(len(key_offsets))
    width = 0.35
    ax_key.bar(x - width/2, key_same, width, label='Same', color='steelblue', edgecolor='black')
    ax_key.bar(x + width/2, key_cross, width, label='Cross', color='coral', edgecolor='black')
    ax_key.set_xticks(x)
    ax_key.set_xticklabels(key_offsets)
    ax_key.set_xlabel('Key Offsets', fontsize=11)
    ax_key.set_ylabel('Similarity', fontsize=11)
    ax_key.set_title('Key Offset Comparison', fontsize=12)
    ax_key.legend(fontsize=9)
    ax_key.grid(True, alpha=0.3, axis='y')

    # 5. 统计摘要 (中下)
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.axis('off')

    stats_text = f"""
    ┌─────────────────────────────────────┐
    │     Statistical Summary             │
    ├─────────────────────────────────────┤
    │ Same Instrument:                    │
    │   Mean:   {np.mean(same_pc):.4f}                     │
    │   Std:    {np.std(same_pc):.4f}                     │
    │   Max:    {np.max(same_pc):.4f} (offset={same_pc.index(max(same_pc))+1})           │
    │   Min:    {np.min(same_pc):.4f} (offset={same_pc.index(min(same_pc))+1})           │
    ├─────────────────────────────────────┤
    │ Cross Instrument:                   │
    │   Mean:   {np.mean(cross_pc):.4f}                     │
    │   Std:    {np.std(cross_pc):.4f}                     │
    │   Max:    {np.max(cross_pc):.4f}                     │
    │   Min:    {np.min(cross_pc):.4f}                     │
    ├─────────────────────────────────────┤
    │ Difference (Same - Cross):          │
    │   Mean:   {np.mean(diff_pc):.4f}                     │
    │   Ratio:  {np.mean(same_pc)/np.mean(cross_pc):.2f}x                        │
    └─────────────────────────────────────┘
    """
    ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=11,
                 family='monospace', transform=ax_stats.transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 6. 掩码建议 (右下)
    ax_mask = fig.add_subplot(gs[2, 2])
    ax_mask.axis('off')

    mask_text = """
    ┌─────────────────────────────────────┐
    │   FC-Attention Mask Design          │
    ├─────────────────────────────────────┤
    │                                     │
    │ Same Instrument:                    │
    │   Keep: 1, 2, 4, 8, 12, 16, 24, 32  │
    │   (8 / 32 = 25%)                    │
    │                                     │
    │ Cross Instrument:                   │
    │   Keep: 1, 2, 4 only                │
    │   (3 / 32 = 9.4%)                   │
    │                                     │
    │ Same-bar Cross Inst:                │
    │   Always keep (full connection)     │
    │                                     │
    ├─────────────────────────────────────┤
    │ Expected sparsity gain: ~70%        │
    └─────────────────────────────────────┘
    """
    ax_mask.text(0.5, 0.5, mask_text, ha='center', va='center', fontsize=11,
                family='monospace', transform=ax_mask.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.suptitle('Comprehensive Similarity Distribution Analysis\nBased on 5,000 MIDI files', fontsize=18, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import os

    stats_path = '/Users/freeninglu/Desktop/MuseFormer/hid_museformer/analysis/similarity_stats.json'
    output_dir = '/Users/freeninglu/Desktop/MuseFormer/hid_museformer/analysis/'

    print("Loading statistics...")
    stats = load_stats(stats_path)

    print("\nGenerating distribution plots...")

    plot_similarity_distribution_by_offset(stats, os.path.join(output_dir, 'dist_by_offset.png'))
    plot_similarity_curve_with_confidence(stats, os.path.join(output_dir, 'dist_curve_confidence.png'))
    plot_periodicity_highlight(stats, os.path.join(output_dir, 'dist_periodicity.png'))
    plot_same_vs_cross_scatter(stats, os.path.join(output_dir, 'dist_scatter.png'))
    plot_difference_distribution(stats, os.path.join(output_dir, 'dist_difference.png'))
    plot_comprehensive_distribution(stats, os.path.join(output_dir, 'dist_comprehensive.png'))

    print("\nAll distribution plots generated!")


if __name__ == '__main__':
    main()
