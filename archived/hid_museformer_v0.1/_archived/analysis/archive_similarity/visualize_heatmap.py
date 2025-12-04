#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的相似度热力图
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


def plot_complete_heatmap(stats, output_path='complete_heatmap.png'):
    """
    完整的相似度热力图
    X轴: Bar Offset (1-32)
    Y轴: 连接类型 (同乐器 Pitch/PitchClass/Rhythm, 跨乐器 Pitch/PitchClass/Rhythm)
    """
    same_inst = stats['same_inst']
    cross_diff = stats['cross_inst_diff_bar']

    # 聚合跨乐器数据
    cross_by_offset = {}
    for pair_str, offset_data in cross_diff.items():
        for offset_str, data in offset_data.items():
            offset = int(offset_str)
            if offset not in cross_by_offset:
                cross_by_offset[offset] = {'pitch': [], 'pitch_class': [], 'rhythm': []}
            cross_by_offset[offset]['pitch'].append(data['pitch']['mean'])
            cross_by_offset[offset]['pitch_class'].append(data['pitch_class']['mean'])
            cross_by_offset[offset]['rhythm'].append(data['rhythm']['mean'])

    offsets = list(range(1, 33))

    # 构建数据矩阵 (6 rows x 32 cols)
    row_labels = [
        'Same Inst - Pitch',
        'Same Inst - Pitch Class',
        'Same Inst - Rhythm',
        'Cross Inst - Pitch',
        'Cross Inst - Pitch Class',
        'Cross Inst - Rhythm',
    ]

    data = np.zeros((6, 32))

    for j, o in enumerate(offsets):
        # 同乐器
        if str(o) in same_inst:
            data[0, j] = same_inst[str(o)]['pitch']['mean']
            data[1, j] = same_inst[str(o)]['pitch_class']['mean']
            data[2, j] = same_inst[str(o)]['rhythm']['mean']

        # 跨乐器
        if o in cross_by_offset:
            data[3, j] = np.mean(cross_by_offset[o]['pitch'])
            data[4, j] = np.mean(cross_by_offset[o]['pitch_class'])
            data[5, j] = np.mean(cross_by_offset[o]['rhythm'])

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(20, 8))

    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.7)

    # 设置坐标轴
    ax.set_xticks(range(32))
    ax.set_xticklabels(offsets, fontsize=10)
    ax.set_yticks(range(6))
    ax.set_yticklabels(row_labels, fontsize=12)

    # 添加分隔线
    ax.axhline(y=2.5, color='white', linewidth=3)

    # 标记 4 的倍数
    for o in [4, 8, 12, 16, 24, 32]:
        idx = o - 1
        ax.axvline(x=idx - 0.5, color='blue', linestyle='-', linewidth=1.5, alpha=0.5)
        ax.axvline(x=idx + 0.5, color='blue', linestyle='-', linewidth=1.5, alpha=0.5)

    # 添加数值标签
    for i in range(6):
        for j in range(32):
            val = data[i, j]
            if val > 0:
                color = 'white' if val > 0.4 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

    ax.set_xlabel('Bar Offset', fontsize=14)
    ax.set_title('Complete Similarity Heatmap: Same Instrument vs Cross Instrument\n(Blue boxes = multiples of 4 bars)', fontsize=16)

    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Jaccard Similarity', fontsize=12)

    # 添加注释
    ax.annotate('Same Instrument', xy=(-3, 1), fontsize=14, fontweight='bold', rotation=90, va='center')
    ax.annotate('Cross Instrument', xy=(-3, 4), fontsize=14, fontweight='bold', rotation=90, va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison_heatmap(stats, output_path='comparison_heatmap.png'):
    """
    同乐器 vs 跨乐器 对比热力图 (只看 Pitch Class)
    """
    same_inst = stats['same_inst']
    cross_diff = stats['cross_inst_diff_bar']

    # 聚合跨乐器数据
    cross_by_offset = {}
    for pair_str, offset_data in cross_diff.items():
        for offset_str, data in offset_data.items():
            offset = int(offset_str)
            if offset not in cross_by_offset:
                cross_by_offset[offset] = []
            cross_by_offset[offset].append(data['pitch_class']['mean'])

    offsets = list(range(1, 33))

    # 构建数据
    same_data = []
    cross_data = []
    diff_data = []

    for o in offsets:
        same_val = same_inst.get(str(o), {}).get('pitch_class', {}).get('mean', 0)
        cross_val = np.mean(cross_by_offset.get(o, [0]))
        same_data.append(same_val)
        cross_data.append(cross_val)
        diff_data.append(same_val - cross_val)

    # 绘制三行热力图
    fig, axes = plt.subplots(3, 1, figsize=(20, 10))

    # 1. Same Instrument
    ax1 = axes[0]
    data1 = np.array(same_data).reshape(1, -1)
    im1 = ax1.imshow(data1, aspect='auto', cmap='Blues', vmin=0, vmax=0.6)
    ax1.set_yticks([0])
    ax1.set_yticklabels(['Same Instrument'], fontsize=12)
    ax1.set_xticks(range(32))
    ax1.set_xticklabels(offsets, fontsize=9)
    ax1.set_title('Same Instrument - Pitch Class Similarity', fontsize=14)
    for j, val in enumerate(same_data):
        color = 'white' if val > 0.4 else 'black'
        ax1.text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)
    # 标记 4 的倍数
    for o in [4, 8, 12, 16, 24, 32]:
        ax1.axvline(x=o-1-0.5, color='red', linestyle='-', linewidth=2, alpha=0.7)
        ax1.axvline(x=o-1+0.5, color='red', linestyle='-', linewidth=2, alpha=0.7)
    plt.colorbar(im1, ax=ax1, shrink=0.3)

    # 2. Cross Instrument
    ax2 = axes[1]
    data2 = np.array(cross_data).reshape(1, -1)
    im2 = ax2.imshow(data2, aspect='auto', cmap='Oranges', vmin=0, vmax=0.6)
    ax2.set_yticks([0])
    ax2.set_yticklabels(['Cross Instrument'], fontsize=12)
    ax2.set_xticks(range(32))
    ax2.set_xticklabels(offsets, fontsize=9)
    ax2.set_title('Cross Instrument - Pitch Class Similarity', fontsize=14)
    for j, val in enumerate(cross_data):
        color = 'white' if val > 0.4 else 'black'
        ax2.text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)
    plt.colorbar(im2, ax=ax2, shrink=0.3)

    # 3. Difference (Same - Cross)
    ax3 = axes[2]
    data3 = np.array(diff_data).reshape(1, -1)
    im3 = ax3.imshow(data3, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.35)
    ax3.set_yticks([0])
    ax3.set_yticklabels(['Difference'], fontsize=12)
    ax3.set_xticks(range(32))
    ax3.set_xticklabels(offsets, fontsize=9)
    ax3.set_title('Difference (Same - Cross): Higher = More benefit from same-instrument attention', fontsize=14)
    for j, val in enumerate(diff_data):
        ax3.text(j, 0, f'{val:.2f}', ha='center', va='center', fontsize=9, color='black')
    plt.colorbar(im3, ax=ax3, shrink=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_instrument_pair_heatmap(stats, output_path='instrument_pair_heatmap.png'):
    """
    不同乐器对之间的同小节相似度热力图
    """
    cross_same = stats['cross_inst_same_bar']

    # GM 乐器分类
    def get_category(program):
        categories = {
            (0, 7): 'Piano',
            (8, 15): 'Chrom.Perc',
            (16, 23): 'Organ',
            (24, 31): 'Guitar',
            (32, 39): 'Bass',
            (40, 47): 'Strings',
            (48, 55): 'Ensemble',
            (56, 63): 'Brass',
            (64, 71): 'Reed',
            (72, 79): 'Pipe',
            (80, 87): 'SynthLead',
            (88, 95): 'SynthPad',
            (96, 103): 'SynthFX',
            (104, 111): 'Ethnic',
            (112, 119): 'Percussive',
            (120, 127): 'SoundFX',
        }
        for (start, end), name in categories.items():
            if start <= program <= end:
                return name
        return 'Unknown'

    # 聚合按类别
    category_pairs = {}
    for pair_str, data in cross_same.items():
        parts = pair_str.split('-')
        if len(parts) != 2:
            continue
        try:
            inst1, inst2 = int(parts[0]), int(parts[1])
        except:
            continue

        cat1 = get_category(inst1)
        cat2 = get_category(inst2)
        pair_key = tuple(sorted([cat1, cat2]))

        if pair_key not in category_pairs:
            category_pairs[pair_key] = {'pitch_class': [], 'rhythm': [], 'count': 0}

        category_pairs[pair_key]['pitch_class'].append(data['pitch_class']['mean'])
        category_pairs[pair_key]['rhythm'].append(data['rhythm']['mean'])
        category_pairs[pair_key]['count'] += data['pitch_class']['count']

    # 获取所有类别
    all_categories = sorted(set([c for pair in category_pairs.keys() for c in pair]))

    # 构建矩阵
    n = len(all_categories)
    matrix = np.zeros((n, n))

    for i, cat1 in enumerate(all_categories):
        for j, cat2 in enumerate(all_categories):
            pair_key = tuple(sorted([cat1, cat2]))
            if pair_key in category_pairs:
                matrix[i, j] = np.mean(category_pairs[pair_key]['pitch_class'])
                matrix[j, i] = matrix[i, j]

    # 绘制
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(matrix, cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(all_categories, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(all_categories, fontsize=10)

    # 添加数值
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if val > 0:
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    ax.set_title('Cross-Instrument Same-Bar Similarity (by Category)\nPitch Class Jaccard Similarity', fontsize=16)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pitch Class Similarity', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_mask_design_heatmap(stats, output_path='mask_design_heatmap.png'):
    """
    最终的掩码设计热力图
    展示哪些连接应该保留
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

    # 数据
    same_sim = [same_inst.get(str(o), {}).get('pitch_class', {}).get('mean', 0) for o in offsets]
    cross_sim = [np.mean(cross_by_offset.get(o, [0])) for o in offsets]

    # 掩码建议
    keep_same = [1, 2, 4, 8, 12, 16, 24, 32]
    keep_cross = [1, 2, 4]

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # 1. 同乐器相似度
    ax1 = axes[0, 0]
    colors1 = ['green' if o in keep_same else 'lightgray' for o in offsets]
    bars1 = ax1.bar(offsets, same_sim, color=colors1, edgecolor='black')
    ax1.set_xlabel('Bar Offset', fontsize=12)
    ax1.set_ylabel('Pitch Class Similarity', fontsize=12)
    ax1.set_title('Same Instrument Similarity\n(Green = Keep in FC-Attention)', fontsize=14)
    ax1.set_xticks(offsets)
    ax1.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    # 添加数值
    for bar, val in zip(bars1, same_sim):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}',
                ha='center', va='bottom', fontsize=8, rotation=90)

    # 2. 跨乐器相似度
    ax2 = axes[0, 1]
    colors2 = ['orange' if o in keep_cross else 'lightgray' for o in offsets]
    bars2 = ax2.bar(offsets, cross_sim, color=colors2, edgecolor='black')
    ax2.set_xlabel('Bar Offset', fontsize=12)
    ax2.set_ylabel('Pitch Class Similarity', fontsize=12)
    ax2.set_title('Cross Instrument Similarity\n(Orange = Keep in FC-Attention)', fontsize=14)
    ax2.set_xticks(offsets)
    ax2.axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, cross_sim):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}',
                ha='center', va='bottom', fontsize=8, rotation=90)

    # 3. 掩码设计矩阵
    ax3 = axes[1, 0]
    mask_matrix = np.zeros((2, 32))
    for o in keep_same:
        mask_matrix[0, o-1] = 1
    for o in keep_cross:
        mask_matrix[1, o-1] = 1

    im3 = ax3.imshow(mask_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax3.set_xticks(range(32))
    ax3.set_xticklabels(offsets, fontsize=8)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Same Instrument', 'Cross Instrument'], fontsize=12)
    ax3.set_xlabel('Bar Offset', fontsize=12)
    ax3.set_title('FC-Attention Mask Design\n(Green = Keep, Red = Remove)', fontsize=14)

    for i in range(2):
        for j in range(32):
            text = '✓' if mask_matrix[i, j] == 1 else ''
            ax3.text(j, i, text, ha='center', va='center', fontsize=10,
                    color='white' if mask_matrix[i, j] == 1 else 'lightgray')

    # 4. 统计摘要
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║           FC-Attention Mask Design Summary                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  Based on analysis of 5,000 MIDI files (450K total)          ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  SAME INSTRUMENT:                                            ║
    ║    • Average similarity: {np.mean(same_sim):.4f}                            ║
    ║    • Peak at offset=4: {same_sim[3]:.4f} (4-bar periodicity)           ║
    ║    • Keep offsets: 1, 2, 4, 8, 12, 16, 24, 32                ║
    ║    • Connections kept: 8 / 32 = 25%                          ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  CROSS INSTRUMENT:                                           ║
    ║    • Average similarity: {np.mean(cross_sim):.4f}                            ║
    ║    • No clear periodicity (flat curve)                       ║
    ║    • Keep offsets: 1, 2, 4 (short-term only)                 ║
    ║    • Connections kept: 3 / 32 = 9.4%                         ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  SAME-BAR CROSS INSTRUMENT:                                  ║
    ║    • Always keep (harmonic coordination)                     ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  ATTENTION REDUCTION:                                        ║
    ║    • Same inst long-range: 75% reduction                     ║
    ║    • Cross inst long-range: 90.6% reduction                  ║
    ║    • Overall sparsity improvement: significant               ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """

    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
            family='monospace', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import os

    stats_path = '/Users/freeninglu/Desktop/MuseFormer/hid_museformer/analysis/similarity_stats.json'
    output_dir = '/Users/freeninglu/Desktop/MuseFormer/hid_museformer/analysis/'

    print("Loading statistics...")
    stats = load_stats(stats_path)

    print("\nGenerating heatmaps...")

    plot_complete_heatmap(stats, os.path.join(output_dir, 'heatmap_complete.png'))
    plot_comparison_heatmap(stats, os.path.join(output_dir, 'heatmap_comparison.png'))
    plot_instrument_pair_heatmap(stats, os.path.join(output_dir, 'heatmap_instrument_pairs.png'))
    plot_mask_design_heatmap(stats, os.path.join(output_dir, 'heatmap_mask_design.png'))

    print("\nAll heatmaps generated!")


if __name__ == '__main__':
    main()
