#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨乐器相关性可视化分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无 GUI 模式

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_stats(path='similarity_stats.json'):
    with open(path, 'r') as f:
        return json.load(f)


def plot_same_instrument_similarity(stats, output_path='same_inst_similarity.png'):
    """
    图1: 同乐器不同小节的相似度曲线
    验证 MuseFormer 的 4 小节周期性发现
    """
    same_inst = stats['same_inst']

    # 提取数据
    offsets = sorted([int(k) for k in same_inst.keys() if int(k) <= 32])
    pitch_class = [same_inst[str(o)]['pitch_class']['mean'] for o in offsets]
    rhythm = [same_inst[str(o)]['rhythm']['mean'] for o in offsets]
    pitch = [same_inst[str(o)]['pitch']['mean'] for o in offsets]

    fig, ax = plt.subplots(figsize=(14, 8))

    # 绘制曲线
    ax.plot(offsets, pitch_class, 'b-o', linewidth=2, markersize=8, label='Pitch Class Similarity')
    ax.plot(offsets, rhythm, 'g-s', linewidth=2, markersize=8, label='Rhythm Similarity')
    ax.plot(offsets, pitch, 'r-^', linewidth=2, markersize=6, label='Pitch Similarity')

    # 标记 4 的倍数（MuseFormer 发现的周期性位置）
    key_offsets = [4, 8, 12, 16, 24, 32]
    for ko in key_offsets:
        if ko in offsets:
            idx = offsets.index(ko)
            ax.axvline(x=ko, color='orange', linestyle='--', alpha=0.5)
            ax.scatter([ko], [pitch_class[idx]], color='orange', s=200, zorder=5, edgecolors='black', linewidths=2)

    # 标记 offset=1,2（短期相关）
    for ko in [1, 2]:
        if ko in offsets:
            idx = offsets.index(ko)
            ax.scatter([ko], [pitch_class[idx]], color='purple', s=200, zorder=5, edgecolors='black', linewidths=2)

    ax.set_xlabel('Bar Offset (小节偏移)', fontsize=14)
    ax.set_ylabel('Jaccard Similarity', fontsize=14)
    ax.set_title('Same Instrument: Bar-to-Bar Similarity\n同乐器不同小节的相似度 (验证 MuseFormer 的发现)', fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 33)
    ax.set_ylim(0, 1)

    # 添加注释
    ax.annotate('Short-term\n(offset=1,2)', xy=(1.5, 0.5), fontsize=11,
                color='purple', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.annotate('4-bar periodicity\n(offset=4,8,12,16...)', xy=(10, 0.55), fontsize=11,
                color='orange', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")


def plot_cross_vs_same_instrument(stats, output_path='cross_vs_same_similarity.png'):
    """
    图2: 同乐器 vs 跨乐器的相似度对比
    关键发现：跨乐器相似度远低于同乐器
    """
    same_inst = stats['same_inst']
    cross_diff = stats['cross_inst_diff_bar']

    # 提取同乐器数据
    offsets = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32]
    same_pc = []
    for o in offsets:
        if str(o) in same_inst:
            same_pc.append(same_inst[str(o)]['pitch_class']['mean'])
        else:
            same_pc.append(0)

    # 聚合跨乐器数据（所有乐器对的平均）
    cross_by_offset = {}
    for pair_str, offset_data in cross_diff.items():
        for offset_str, data in offset_data.items():
            offset = int(offset_str)
            if offset not in cross_by_offset:
                cross_by_offset[offset] = []
            cross_by_offset[offset].append(data['pitch_class']['mean'])

    cross_pc = []
    for o in offsets:
        if o in cross_by_offset:
            cross_pc.append(np.mean(cross_by_offset[o]))
        else:
            cross_pc.append(0)

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(offsets))
    width = 0.35

    bars1 = ax.bar(x - width/2, same_pc, width, label='Same Instrument (同乐器)', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, cross_pc, width, label='Cross Instrument (跨乐器)', color='coral', edgecolor='black')

    # 添加数值标签
    for bar, val in zip(bars1, same_pc):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, color='steelblue')
    for bar, val in zip(bars2, cross_pc):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9, color='coral')

    ax.set_xlabel('Bar Offset (小节偏移)', fontsize=14)
    ax.set_ylabel('Pitch Class Similarity', fontsize=14)
    ax.set_title('Same Instrument vs Cross Instrument Similarity\n同乐器 vs 跨乐器相似度对比', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(offsets)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 0.7)

    # 添加差距注释
    avg_same = np.mean([v for v in same_pc if v > 0])
    avg_cross = np.mean([v for v in cross_pc if v > 0])
    ax.axhline(y=avg_same, color='steelblue', linestyle='--', alpha=0.7)
    ax.axhline(y=avg_cross, color='coral', linestyle='--', alpha=0.7)

    ax.annotate(f'Avg: {avg_same:.3f}', xy=(11, avg_same + 0.02), fontsize=11, color='steelblue')
    ax.annotate(f'Avg: {avg_cross:.3f}', xy=(11, avg_cross + 0.02), fontsize=11, color='coral')
    ax.annotate(f'Gap: {avg_same - avg_cross:.3f}\n({(avg_same/avg_cross - 1)*100:.0f}% higher)',
               xy=(11.5, (avg_same + avg_cross)/2), fontsize=12, color='black',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")


def plot_heatmap_same_inst(stats, output_path='same_inst_heatmap.png'):
    """
    图3: 同乐器相似度热力图（验证周期性）
    """
    same_inst = stats['same_inst']

    offsets = list(range(1, 33))
    metrics = ['pitch', 'pitch_class', 'rhythm']

    data = np.zeros((len(metrics), len(offsets)))
    for i, metric in enumerate(metrics):
        for j, o in enumerate(offsets):
            if str(o) in same_inst and metric in same_inst[str(o)]:
                data[i, j] = same_inst[str(o)][metric]['mean']

    fig, ax = plt.subplots(figsize=(16, 5))

    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.7)

    ax.set_xticks(range(len(offsets)))
    ax.set_xticklabels(offsets, fontsize=9)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(['Pitch', 'Pitch Class', 'Rhythm'], fontsize=12)

    # 标记 4 的倍数
    for o in [4, 8, 12, 16, 24, 32]:
        idx = o - 1
        ax.axvline(x=idx - 0.5, color='blue', linestyle='-', linewidth=2, alpha=0.7)
        ax.axvline(x=idx + 0.5, color='blue', linestyle='-', linewidth=2, alpha=0.7)

    ax.set_xlabel('Bar Offset (小节偏移)', fontsize=14)
    ax.set_title('Same Instrument Similarity Heatmap\n同乐器相似度热力图 (蓝框=4的倍数)', fontsize=16)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Jaccard Similarity', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")


def plot_attention_mask_recommendation(stats, output_path='attention_mask_recommendation.png'):
    """
    图4: FC-Attention 掩码设计建议
    """
    same_inst = stats['same_inst']
    cross_diff = stats['cross_inst_diff_bar']

    # 计算跨乐器平均
    cross_by_offset = {}
    for pair_str, offset_data in cross_diff.items():
        for offset_str, data in offset_data.items():
            offset = int(offset_str)
            if offset not in cross_by_offset:
                cross_by_offset[offset] = []
            cross_by_offset[offset].append(data['pitch_class']['mean'])

    offsets = list(range(1, 33))

    # 准备数据
    same_data = []
    cross_data = []
    for o in offsets:
        if str(o) in same_inst:
            same_data.append(same_inst[str(o)]['pitch_class']['mean'])
        else:
            same_data.append(0)
        if o in cross_by_offset:
            cross_data.append(np.mean(cross_by_offset[o]))
        else:
            cross_data.append(0)

    # 创建掩码建议矩阵
    # 行: query 乐器 (0=same, 1=cross)
    # 列: offset
    threshold_high = 0.40  # 高相似度阈值
    threshold_low = 0.30   # 低相似度阈值

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 上图：相似度曲线 + 阈值
    ax1 = axes[0]
    ax1.fill_between(offsets, same_data, alpha=0.3, color='steelblue', label='Same Instrument')
    ax1.fill_between(offsets, cross_data, alpha=0.3, color='coral', label='Cross Instrument')
    ax1.plot(offsets, same_data, 'b-o', linewidth=2, markersize=6)
    ax1.plot(offsets, cross_data, 'r-s', linewidth=2, markersize=6)

    ax1.axhline(y=threshold_high, color='green', linestyle='--', linewidth=2, label=f'High threshold ({threshold_high})')
    ax1.axhline(y=threshold_low, color='orange', linestyle='--', linewidth=2, label=f'Low threshold ({threshold_low})')

    # 标记保留的连接
    for o in [1, 2, 4, 8, 12, 16, 24, 32]:
        ax1.axvline(x=o, color='green', linestyle=':', alpha=0.5)

    ax1.set_xlabel('Bar Offset', fontsize=12)
    ax1.set_ylabel('Pitch Class Similarity', fontsize=12)
    ax1.set_title('Similarity Curves with Thresholds\n相似度曲线与阈值', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 32.5)
    ax1.set_ylim(0, 0.65)

    # 下图：建议的掩码矩阵
    ax2 = axes[1]

    # 构建掩码建议
    mask_same = np.zeros(len(offsets))
    mask_cross = np.zeros(len(offsets))

    # 同乐器：保留 1,2,4,8,12,16,24,32
    keep_same = [1, 2, 4, 8, 12, 16, 24, 32]
    for o in keep_same:
        mask_same[o-1] = 1

    # 跨乐器：只保留 1,2,4
    keep_cross = [1, 2, 4]
    for o in keep_cross:
        mask_cross[o-1] = 1

    mask_matrix = np.vstack([mask_same, mask_cross])

    im = ax2.imshow(mask_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    ax2.set_xticks(range(len(offsets)))
    ax2.set_xticklabels(offsets, fontsize=9)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Same Instrument\n(同乐器)', 'Cross Instrument\n(跨乐器)'], fontsize=11)

    ax2.set_xlabel('Bar Offset (小节偏移)', fontsize=12)
    ax2.set_title('Recommended FC-Attention Mask Design\n建议的 FC-Attention 掩码设计 (绿=保留, 红=删除)', fontsize=14)

    # 添加文字说明
    for i in range(2):
        for j in range(len(offsets)):
            text = '✓' if mask_matrix[i, j] == 1 else ''
            ax2.text(j, i, text, ha='center', va='center', fontsize=12, color='white' if mask_matrix[i, j] == 1 else 'gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")


def plot_periodicity_analysis(stats, output_path='periodicity_analysis.png'):
    """
    图5: 周期性分析（FFT 验证 4 小节周期）
    """
    same_inst = stats['same_inst']

    # 提取连续的相似度数据
    offsets = list(range(1, 33))
    similarity = []
    for o in offsets:
        if str(o) in same_inst:
            similarity.append(same_inst[str(o)]['pitch_class']['mean'])
        else:
            similarity.append(0)

    # 去均值
    similarity = np.array(similarity)
    similarity_centered = similarity - np.mean(similarity)

    # FFT
    fft_result = np.fft.fft(similarity_centered)
    fft_magnitude = np.abs(fft_result)[:len(offsets)//2]
    frequencies = np.fft.fftfreq(len(offsets), d=1)[:len(offsets)//2]
    periods = 1 / frequencies[1:]  # 排除 0 频率
    fft_magnitude_no_dc = fft_magnitude[1:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：原始相似度
    ax1 = axes[0]
    ax1.bar(offsets, similarity, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axhline(y=np.mean(similarity), color='red', linestyle='--', label=f'Mean: {np.mean(similarity):.3f}')

    # 标记 4 的倍数
    for o in [4, 8, 12, 16, 24, 32]:
        ax1.axvline(x=o, color='orange', linestyle='--', alpha=0.7)

    ax1.set_xlabel('Bar Offset', fontsize=12)
    ax1.set_ylabel('Pitch Class Similarity', fontsize=12)
    ax1.set_title('Same Instrument Similarity by Offset', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图：FFT 频谱
    ax2 = axes[1]
    ax2.bar(range(len(periods)), fft_magnitude_no_dc, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xticks(range(len(periods)))
    ax2.set_xticklabels([f'{p:.1f}' for p in periods], fontsize=8, rotation=45)
    ax2.set_xlabel('Period (bars)', fontsize=12)
    ax2.set_ylabel('FFT Magnitude', fontsize=12)
    ax2.set_title('Frequency Analysis (FFT)\n周期性分析', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    # 标记主要周期
    peak_idx = np.argmax(fft_magnitude_no_dc)
    ax2.annotate(f'Peak at period={periods[peak_idx]:.1f} bars',
                xy=(peak_idx, fft_magnitude_no_dc[peak_idx]),
                xytext=(peak_idx + 2, fft_magnitude_no_dc[peak_idx] * 1.1),
                fontsize=11, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")


def plot_summary_table(stats, output_path='summary_table.png'):
    """
    图6: 总结表格
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # 准备数据
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

    # 表格数据
    offsets = [1, 2, 4, 8, 12, 16, 24, 32]

    table_data = []
    for o in offsets:
        same_sim = same_inst.get(str(o), {}).get('pitch_class', {}).get('mean', 0)
        cross_sim = np.mean(cross_by_offset.get(o, [0]))
        ratio = same_sim / cross_sim if cross_sim > 0 else 0
        keep_cross = '✓' if o <= 4 else '✗'

        table_data.append([
            str(o),
            f'{same_sim:.4f}',
            f'{cross_sim:.4f}',
            f'{ratio:.2f}x',
            '✓',  # 同乐器都保留
            keep_cross,
        ])

    columns = ['Offset', 'Same Inst\nSimilarity', 'Cross Inst\nSimilarity', 'Ratio', 'Keep\nSame', 'Keep\nCross']

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.1, 0.15, 0.15, 0.12, 0.1, 0.1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # 设置表头样式
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # 高亮 4 的倍数行
    highlight_rows = [1, 3, 4, 5, 6, 7, 8]  # offset 1, 4, 8, 12, 16, 24, 32 对应的行
    for row in [3, 4, 5, 6, 7, 8]:  # 4, 8, 12, 16, 24, 32
        for col in range(len(columns)):
            table[(row, col)].set_facecolor('#FFF2CC')

    ax.set_title('FC-Attention Mask Design Summary\n基于统计分析的 FC-Attention 掩码设计建议',
                fontsize=16, fontweight='bold', pad=20)

    # 添加说明文字
    note_text = """
    Key Findings (关键发现):

    1. Same Instrument: offset=4,8,12,16,24,32 show significantly higher similarity (~0.50)
       同乐器: 4的倍数位置相似度显著更高 (验证 MuseFormer)

    2. Cross Instrument: similarity is consistently lower (~0.25) with NO clear periodicity
       跨乐器: 相似度普遍较低且无明显周期性

    3. Recommendation: Keep all fine-grained connections for same instrument,
       but only keep offset=1,2,4 for cross instrument
       建议: 同乐器保留完整 fine-grained，跨乐器只保留近距离 (1,2,4)
    """

    fig.text(0.5, 0.15, note_text, ha='center', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
            family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")


def main():
    import os

    # 加载统计数据
    stats_path = '/Users/freeninglu/Desktop/MuseFormer/hid_museformer/analysis/similarity_stats.json'
    output_dir = '/Users/freeninglu/Desktop/MuseFormer/hid_museformer/analysis/'

    if not os.path.exists(stats_path):
        print(f"错误: 找不到 {stats_path}")
        return

    print("加载统计数据...")
    stats = load_stats(stats_path)

    print("\n生成可视化图表...")

    # 生成所有图表
    plot_same_instrument_similarity(stats, os.path.join(output_dir, 'fig1_same_inst_similarity.png'))
    plot_cross_vs_same_instrument(stats, os.path.join(output_dir, 'fig2_cross_vs_same.png'))
    plot_heatmap_same_inst(stats, os.path.join(output_dir, 'fig3_same_inst_heatmap.png'))
    plot_attention_mask_recommendation(stats, os.path.join(output_dir, 'fig4_mask_recommendation.png'))
    plot_periodicity_analysis(stats, os.path.join(output_dir, 'fig5_periodicity_analysis.png'))
    plot_summary_table(stats, os.path.join(output_dir, 'fig6_summary_table.png'))

    print("\n所有图表生成完成!")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()
