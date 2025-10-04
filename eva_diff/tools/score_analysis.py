#!/usr/bin/env python3
"""
评分分析工具

功能：
1. 计算评分统计指标（均值、方差、标准差等）
2. 绘制每个prompt的平均分数折线图
3. 支持多个JSON文件对比分析
4. 可指定具体评估指标
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple
import re


def extract_prompt_index_and_suffix(filename: str) -> Tuple[int, int]:
    """
    从文件名提取prompt索引和图像后缀

    Args:
        filename: 例如 "0_1.png" 或 "5.png"

    Returns:
        (prompt_index, suffix) 例如 (0, 1) 或 (5, 0)
    """
    name_without_ext = filename.split('.')[0]
    if '_' in name_without_ext:
        parts = name_without_ext.split('_')
        return int(parts[0]), int(parts[1])
    else:
        return int(name_without_ext), 0


def load_scores(json_file: str) -> Dict:
    """加载评分JSON文件"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_statistics(scores: Dict, metric: str) -> Dict:
    """
    计算指定指标的统计信息

    Args:
        scores: 评分数据字典
        metric: 指标名称，如 'semantic_coverage'

    Returns:
        统计信息字典
    """
    values = []

    for filename, score_data in scores.items():
        if score_data is None:
            continue
        if metric in score_data:
            values.append(score_data[metric])

    if not values:
        return {
            'count': 0,
            'mean': 0,
            'std': 0,
            'var': 0,
            'min': 0,
            'max': 0
        }

    values = np.array(values)

    return {
        'count': len(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'var': float(np.var(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


def group_scores_by_prompt(scores: Dict, metric: str) -> Dict[int, List[float]]:
    """
    按prompt索引分组评分

    Args:
        scores: 评分数据字典
        metric: 指标名称

    Returns:
        {prompt_index: [scores_list]}
    """
    prompt_scores = defaultdict(list)

    for filename, score_data in scores.items():
        if score_data is None:
            continue
        if metric not in score_data:
            continue

        prompt_index, _ = extract_prompt_index_and_suffix(filename)
        prompt_scores[prompt_index].append(score_data[metric])

    return dict(prompt_scores)


def calculate_prompt_averages(prompt_scores: Dict[int, List[float]]) -> Tuple[List[int], List[float]]:
    """
    计算每个prompt的平均分数

    Args:
        prompt_scores: {prompt_index: [scores_list]}

    Returns:
        (prompt_indices, average_scores)
    """
    prompt_indices = sorted(prompt_scores.keys())
    average_scores = []

    for prompt_idx in prompt_indices:
        scores = prompt_scores[prompt_idx]
        avg_score = np.mean(scores) if scores else 0
        average_scores.append(avg_score)

    return prompt_indices, average_scores


def plot_comparison(score_files: Dict[str, str], metric: str, output_path: str = None):
    """
    绘制多个模型的评分对比折线图

    Args:
        score_files: {label: json_file_path}
        metric: 评估指标名称
        output_path: 图片保存路径
    """
    plt.figure(figsize=(12, 8))

    all_stats = {}

    for label, json_file in score_files.items():
        # 加载数据
        scores = load_scores(json_file)

        # 计算统计信息
        stats = calculate_statistics(scores, metric)
        all_stats[label] = stats

        # 按prompt分组并计算平均值
        prompt_scores = group_scores_by_prompt(scores, metric)
        prompt_indices, average_scores = calculate_prompt_averages(
            prompt_scores)

        # 绘制折线
        plt.plot(prompt_indices, average_scores, marker='o', linewidth=2,
                 markersize=4, label=f'{label} (μ={stats["mean"]:.3f})', alpha=0.8)

    plt.xlabel('Prompt Index', fontsize=12)
    plt.ylabel(f'{metric.replace("_", " ").title()} Score', fontsize=12)
    plt.title(
        f'Average {metric.replace("_", " ").title()} Score by Prompt', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close()

    return all_stats


def plot_score_distribution(score_files: Dict[str, str], metric: str, output_path: str = None, bins: int = 20):
    """
    绘制分数分布柱状图

    Args:
        score_files: {label: json_file_path}
        metric: 评估指标名称
        output_path: 图片保存路径
        bins: 柱状图的分箱数量
    """
    plt.figure(figsize=(12, 8))

    all_stats = {}
    colors = plt.cm.Set1(np.linspace(0, 1, len(score_files)))

    for i, (label, json_file) in enumerate(score_files.items()):
        # 加载数据
        scores = load_scores(json_file)

        # 提取所有分数值
        values = []
        for filename, score_data in scores.items():
            if score_data is None:
                continue
            if metric in score_data:
                values.append(score_data[metric])

        if not values:
            continue

        values = np.array(values)

        # 计算统计信息
        stats = calculate_statistics(scores, metric)
        all_stats[label] = stats

        # 绘制直方图
        plt.hist(values, bins=bins, alpha=0.7, label=f'{label} (n={len(values)}, μ={stats["mean"]:.3f})',
                 color=colors[i], edgecolor='black', linewidth=0.5)

    plt.xlabel(f'{metric.replace("_", " ").title()} Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(
        f'{metric.replace("_", " ").title()} Score Distribution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close()

    return all_stats


def plot_score_distribution_subplots(score_files: Dict[str, str], metric: str, output_path: str = None, bins: int = 20):
    """
    绘制分数分布柱状图（子图形式，每个模型一个子图）

    Args:
        score_files: {label: json_file_path}
        metric: 评估指标名称
        output_path: 图片保存路径
        bins: 柱状图的分箱数量
    """
    n_models = len(score_files)
    fig, axes = plt.subplots(
        n_models, 1, figsize=(10, 4*n_models), sharex=True)

    if n_models == 1:
        axes = [axes]

    all_stats = {}
    colors = plt.cm.Set1(np.linspace(0, 1, n_models))

    for i, (label, json_file) in enumerate(score_files.items()):
        # 加载数据
        scores = load_scores(json_file)

        # 提取所有分数值
        values = []
        for filename, score_data in scores.items():
            if score_data is None:
                continue
            if metric in score_data:
                values.append(score_data[metric])

        if not values:
            continue

        values = np.array(values)

        # 计算统计信息
        stats = calculate_statistics(scores, metric)
        all_stats[label] = stats

        # 绘制直方图
        axes[i].hist(values, bins=bins, alpha=0.7, color=colors[i],
                     edgecolor='black', linewidth=0.5)
        axes[i].set_ylabel('Count', fontsize=10)
        axes[i].set_title(f'{label} (n={len(values)}, μ={stats["mean"]:.3f}, σ={stats["std"]:.3f})',
                          fontsize=11)
        axes[i].grid(True, alpha=0.3, axis='y')

        # 添加统计线
        axes[i].axvline(stats['mean'], color='red', linestyle='--',
                        alpha=0.8, label=f'Mean: {stats["mean"]:.3f}')
        axes[i].legend(fontsize=9)

    # 设置共同的x轴标签
    axes[-1].set_xlabel(f'{metric.replace("_", " ").title()} Score', fontsize=12)

    plt.suptitle(
        f'{metric.replace("_", " ").title()} Score Distribution Comparison', fontsize=14)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close()

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="评分分析和可视化工具")
    parser.add_argument("--score_files", type=str, nargs="+", required=True,
                        help="评分JSON文件路径列表")
    parser.add_argument("--labels", type=str, nargs="+", required=True,
                        help="对应的标签列表")
    parser.add_argument("--metric", type=str, default="semantic_coverage",
                        help="要分析的指标名称")
    parser.add_argument("--output", type=str, default=None,
                        help="图片保存路径")
    parser.add_argument("--plot_type", type=str, default="line",
                        choices=["line", "hist", "hist_subplots"],
                        help="图表类型: line=折线图, hist=分布柱状图, hist_subplots=分布柱状图(子图)")
    parser.add_argument("--bins", type=int, default=20,
                        help="柱状图的分箱数量 (仅用于hist和hist_subplots)")

    args = parser.parse_args()

    # 检查参数
    if len(args.score_files) != len(args.labels):
        raise ValueError("score_files和labels的数量必须相同")

    # 构建文件字典
    score_files = dict(zip(args.labels, args.score_files))

    # 检查文件是否存在
    for label, file_path in score_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

    # 根据图表类型执行相应的分析和绘图
    if args.plot_type == "line":
        stats = plot_comparison(score_files, args.metric, args.output)
    elif args.plot_type == "hist":
        stats = plot_score_distribution(
            score_files, args.metric, args.output, args.bins)
    elif args.plot_type == "hist_subplots":
        stats = plot_score_distribution_subplots(
            score_files, args.metric, args.output, args.bins)
    else:
        raise ValueError(f"Unsupported plot type: {args.plot_type}")

    # 输出汇总统计
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    for label, stat in stats.items():
        print(f"\n{label}:")
        print(f"  Samples: {stat['count']}")
        print(f"  Mean: {stat['mean']:.4f}")
        print(f"  Std: {stat['std']:.4f}")
        print(f"  Variance: {stat['var']:.4f}")
        print(f"  Range: [{stat['min']:.4f}, {stat['max']:.4f}]")

    if args.output:
        print(f"\nChart saved to: {args.output}")
        print("Analysis completed successfully!")


if __name__ == "__main__":
    main()
