#!/usr/bin/env python3
"""
诊断脚本：检查偏好数据集是否正确构建并可以被训练脚本使用
"""

import os
import sys
import argparse
from datasets import load_from_disk


def diagnose_dataset(dataset_path: str):
    """诊断数据集是否正确构建"""

    print(f"=== 诊断数据集: {dataset_path} ===\n")

    # 1. 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
        return False

    if not os.path.isdir(dataset_path):
        print(f"❌ 错误: 数据集路径不是目录: {dataset_path}")
        return False

    print(f"✅ 数据集路径存在: {dataset_path}")

    # 2. 检查是否是有效的 HuggingFace 数据集
    try:
        dataset = load_from_disk(dataset_path)
        print(f"✅ 成功加载 HuggingFace 数据集")
    except Exception as e:
        print(f"❌ 错误: 无法加载 HuggingFace 数据集: {e}")
        return False

    # 3. 检查数据集大小
    dataset_size = len(dataset)
    print(f"✅ 数据集大小: {dataset_size} 个样本")

    if dataset_size == 0:
        print(f"❌ 错误: 数据集为空")
        return False

    # 4. 检查列名
    expected_columns = ['jpg_0', 'jpg_1', 'label_0', 'caption']
    actual_columns = dataset.column_names
    print(f"✅ 数据集列名: {actual_columns}")

    missing_columns = set(expected_columns) - set(actual_columns)
    if missing_columns:
        print(f"❌ 错误: 缺少必需的列: {missing_columns}")
        return False

    print(f"✅ 所有必需的列都存在")

    # 5. 检查第一个样本的数据类型
    try:
        sample = dataset[0]
        print(f"\n=== 第一个样本信息 ===")
        print(f"Caption: {sample['caption'][:100]}..." if len(
            sample['caption']) > 100 else f"Caption: {sample['caption']}")
        print(f"Label: {sample['label_0']}")
        print(
            f"jpg_0 类型: {type(sample['jpg_0'])}, 大小: {len(sample['jpg_0'])} bytes")
        print(
            f"jpg_1 类型: {type(sample['jpg_1'])}, 大小: {len(sample['jpg_1'])} bytes")

        # 检查图像数据是否为 bytes
        if not isinstance(sample['jpg_0'], bytes):
            print(f"❌ 错误: jpg_0 不是 bytes 类型，而是 {type(sample['jpg_0'])}")
            return False

        if not isinstance(sample['jpg_1'], bytes):
            print(f"❌ 错误: jpg_1 不是 bytes 类型，而是 {type(sample['jpg_1'])}")
            return False

        print(f"✅ 图像数据类型正确")

    except Exception as e:
        print(f"❌ 错误: 无法访问第一个样本: {e}")
        return False

    # 6. 检查标签分布
    try:
        labels = [dataset[i]['label_0'] for i in range(min(100, len(dataset)))]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f"\n=== 标签分布 (前100个样本) ===")
        for label, count in label_counts.items():
            print(f"Label {label}: {count} 个样本")

        if len(label_counts) == 1:
            print(f"⚠️  警告: 所有样本都有相同的标签，这可能不是期望的")

    except Exception as e:
        print(f"❌ 错误: 无法分析标签分布: {e}")
        return False

    # 7. 模拟训练脚本的数据加载
    print(f"\n=== 模拟训练脚本数据加载 ===")
    try:
        # 模拟训练脚本中的数据加载逻辑
        from datasets import load_dataset

        # 这是训练脚本中的逻辑
        if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
            dataset_dict = {'train': load_from_disk(dataset_path)}
            print(f"✅ 训练脚本可以正确加载数据集")
            print(f"✅ 数据集字典键: {list(dataset_dict.keys())}")
            print(f"✅ 训练集大小: {len(dataset_dict['train'])}")
        else:
            print(f"❌ 错误: 训练脚本无法识别数据集路径")
            return False

    except Exception as e:
        print(f"❌ 错误: 模拟训练脚本加载失败: {e}")
        return False

    print(f"\n✅ 数据集诊断完成，所有检查都通过！")
    return True


def main():
    parser = argparse.ArgumentParser(description="诊断偏好数据集")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="HuggingFace 格式数据集路径")

    args = parser.parse_args()

    success = diagnose_dataset(args.dataset_path)

    if success:
        print(f"\n🎉 数据集可以用于训练！")
        sys.exit(0)
    else:
        print(f"\n💥 数据集存在问题，请检查并修复后再试")
        sys.exit(1)


if __name__ == "__main__":
    main()
