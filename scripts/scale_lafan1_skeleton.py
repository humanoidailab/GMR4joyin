#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse


def scale_offset_line(line: str, scale: float) -> str:
    """
    把一行 '    OFFSET x y z' 按比例缩放 x y z，前面的缩进和关键词 OFFSET 保持不变。
    """
    stripped = line.lstrip()
    if not stripped.startswith("OFFSET"):
        return line

    indent_len = len(line) - len(stripped)
    indent = line[:indent_len]

    parts = stripped.strip().split()
    # 期望格式：OFFSET x y z
    if len(parts) != 4:
        # 结构不对就不动
        return line

    _, x_str, y_str, z_str = parts
    try:
        x = float(x_str)
        y = float(y_str)
        z = float(z_str)
    except ValueError:
        # 非数字就不动
        return line

    x *= scale
    y *= scale
    z *= scale

    # 保持 6 位小数，你可以按需改成别的格式
    new_line = f"{indent}OFFSET {x:.6f} {y:.6f} {z:.6f}\n"
    return new_line


def process_single_bvh(src_path: str, dst_path: str, scale: float):
    """
    只缩放 BVH 中 HIERARCHY 部分的 OFFSET，MOTION 部分原样复制。
    """
    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    new_lines = []
    in_motion_section = False

    for line in lines:
        # 一旦遇到 "MOTION" 行及之后的内容，全部原样复制
        if not in_motion_section:
            # 判断是否进入 MOTION 段
            if line.strip().startswith("MOTION"):
                in_motion_section = True
                new_lines.append(line)
                continue

            # 在 HIERARCHY 段中，处理 OFFSET 行
            stripped = line.lstrip()
            if stripped.startswith("OFFSET"):
                new_lines.append(scale_offset_line(line, scale))
            else:
                new_lines.append(line)
        else:
            # MOTION 部分：完全不改
            new_lines.append(line)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser(
        description="批量缩放 BVH 骨架（OFFSET）而不修改 MOTION 数据"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="原始 BVH 文件所在目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="缩放后 BVH 输出目录")
    parser.add_argument("--scale", type=float, required=True,
                        help="骨架缩放比例，例如 0.01 把 cm 变成 m")
    parser.add_argument("--ext", type=str, default=".bvh",
                        help="文件扩展名（默认 .bvh）")

    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir
    scale = args.scale
    ext = args.ext

    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"输入目录不存在: {in_dir}")

    for root, _, files in os.walk(in_dir):
        for fname in files:
            if not fname.lower().endswith(ext):
                continue

            src_path = os.path.join(root, fname)
            rel_path = os.path.relpath(src_path, in_dir)
            dst_path = os.path.join(out_dir, rel_path)

            print(f"Processing: {src_path} -> {dst_path}")
            process_single_bvh(src_path, dst_path, scale)


if __name__ == "__main__":
    main()
