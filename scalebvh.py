#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
等比缩放 BVH 文件的长度单位（例如把 113mm 的骨架缩放到 80mm）

默认：scale = 80 / 113
功能：
1. 缩放 HIERARCHY 部分所有 OFFSET 行；
2. 缩放 MOTION 部分所有带 position 的通道值（一般是 ROOT 的 XYZ 平移）。

用法示例：
    python scale_bvh.py --input in.bvh --output out_scaled.bvh --scale 0.7079646
或者（你的情况）：
    python scale_bvh.py --input in.bvh --output out_80mm.bvh
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="输入 BVH 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出 BVH 文件路径")
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=80.0 / 113.0,  # 你的场景默认：113mm -> 80mm
        help="缩放系数（默认 80/113 ≈ 0.7079646）",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    scale = args.scale

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 1. 找到 MOTION 的起始行
    motion_idx = None
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith("MOTION"):
            motion_idx = idx
            break

    if motion_idx is None:
        raise RuntimeError("没有在文件中找到 'MOTION' 行，这可能不是合法的 BVH 文件。")

    header_lines = lines[:motion_idx]
    motion_lines = lines[motion_idx:]  # 包含 'MOTION' 行及之后

    # 2. 在 header 部分收集 channel names，并缩放 OFFSET
    channel_names = []
    new_header_lines = []

    for line in header_lines:
        stripped = line.lstrip()
        # 收集 CHANNELS 信息
        if stripped.startswith("CHANNELS"):
            # 例如：CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
            parts = stripped.split()
            n_chan = int(parts[1])
            chans = parts[2 : 2 + n_chan]
            channel_names.extend(chans)
            new_header_lines.append(line)
        # 缩放 OFFSET
        elif stripped.startswith("OFFSET"):
            # 保留原缩进
            indent_len = len(line) - len(stripped)
            indent = line[:indent_len]
            parts = stripped.split()
            # OFFSET 后面通常是三个数字
            nums = parts[1:4]
            scaled = []
            for v in nums:
                try:
                    val = float(v)
                except ValueError:
                    # 万一有奇怪格式，直接原样保留
                    scaled.append(v)
                    continue
                scaled.append(f"{val * scale:.6f}")
            new_line = indent + "OFFSET " + " ".join(scaled) + "\n"
            new_header_lines.append(new_line)
        else:
            new_header_lines.append(line)

    # 3. 计算所有 position 通道在整条通道序列中的索引
    #    通道顺序就是 header 里所有 CHANNELS 声明的顺序
    pos_indices = [i for i, name in enumerate(channel_names) if name.lower().endswith("position")]

    # 4. 处理 MOTION 部分：只对每帧的数据行中，pos_indices 对应的值做缩放
    new_motion_lines = []
    # 第一行是 MOTION
    new_motion_lines.append(motion_lines[0])

    # 接下来的几行通常是：
    # Frames: N
    # Frame Time: 0.0333333
    # 然后才是每帧数据
    header_done = False
    for line in motion_lines[1:]:
        stripped = line.strip()
        if not header_done:
            # 空行或 Frames/Frame Time 这些行，先照抄
            if stripped == "" or stripped.startswith("Frames:") or stripped.startswith("Frame Time:"):
                new_motion_lines.append(line)
                continue
            else:
                # 从这里开始就是帧数据了
                header_done = True

        if stripped == "":
            new_motion_lines.append(line)
            continue

        parts = stripped.split()
        # 有的 BVH 可能有多余空格或 tab，简单处理
        # 只有长度与 channel 数一致时才处理
        if len(parts) == len(channel_names):
            for idx in pos_indices:
                try:
                    val = float(parts[idx])
                    parts[idx] = f"{val * scale:.6f}"
                except ValueError:
                    # 如果解析失败，就跳过这个值
                    pass
            new_line = " ".join(parts) + "\n"
        else:
            # 长度不对就原样写回（稳一点）
            new_line = line

        new_motion_lines.append(new_line)

    # 5. 写回文件
    with open(args.output, "w", encoding="utf-8") as f:
        for l in new_header_lines:
            f.write(l)
        for l in new_motion_lines:
            f.write(l)

    print(f"缩放完成：scale = {scale:.8f}")
    print(f"输入：{args.input}")
    print(f"输出：{args.output}")

if __name__ == "__main__":
    main()
