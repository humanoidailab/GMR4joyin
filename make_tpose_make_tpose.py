#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

def make_tpose_from_bvh(src_bvh, dst_bvh, n_frames=2):
    with open(src_bvh, "r") as f:
        lines = f.readlines()

    # 1) 找到 MOTION 行、Frames 行、Frame Time 行
    motion_idx = None
    frames_idx = None
    frame_time_idx = None

    for i, line in enumerate(lines):
        if line.strip().startswith("MOTION"):
            motion_idx = i
        elif line.strip().startswith("Frames:"):
            frames_idx = i
        elif line.strip().startswith("Frame Time:"):
            frame_time_idx = i
        if motion_idx is not None and frames_idx is not None and frame_time_idx is not None:
            break

    if motion_idx is None or frames_idx is None or frame_time_idx is None:
        raise RuntimeError("BVH 文件中找不到 MOTION / Frames / Frame Time 行，请检查源 BVH。")

    # 2) 取出第一帧原始数据行
    #  第一帧数据行在 Frame Time 行的下一行
    first_frame_idx = frame_time_idx + 1
    if first_frame_idx >= len(lines):
        raise RuntimeError("BVH 中没有任何帧数据。")

    first_frame_line = lines[first_frame_idx].strip()
    if not first_frame_line:
        # 如果这一行是空的，往后找一个非空行
        i = first_frame_idx + 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            raise RuntimeError("BVH 中没有任何有效帧数据。")
        first_frame_line = lines[i].strip()

    nums = first_frame_line.split()
    print(f"[INFO] 原始第一帧共有 {len(nums)} 个数字。")

    if len(nums) < 69:
        raise RuntimeError(f"原始第一帧数字个数 {len(nums)} < 69，不像是 lafan1 格式，请检查。")

    # lafan1 约定：3 个平移 + 66 个旋转 = 69
    # 我们严格截取前 3 个作为 root pos，后 66 个设为 0
    root_pos = nums[:3]
    # 如果原始行超过 69 个，只取前 69 个（有的工具可能在末尾多写垃圾值）
    # 这里我们直接重新构造：3 + 66
    tpose_frame = root_pos + ["0"] * 66

    if len(tpose_frame) != 69:
        raise RuntimeError(f"构造的 T-pose 帧数字个数不是 69，而是 {len(tpose_frame)}，逻辑有误。")

    tpose_line = " ".join(tpose_frame) + "\n"

    # 3) 写出新的 BVH：
    #    - HIERARCHY + MOTION 之前的内容原样保留
    #    - Frames 改为 n_frames
    #    - Frame Time 原样保留
    #    - 后续帧数据全部替换为 n_frames 行 tpose_line
    with open(dst_bvh, "w") as f:
        # 写 HIERARCHY + MOTION
        for i in range(motion_idx + 1):  # 包含 MOTION 行
            f.write(lines[i])

        # 写 Frames 行（改为 n_frames）
        f.write(f"Frames: {n_frames}\n")

        # 写 Frame Time 行（保持原始）
        frame_time_line = lines[frame_time_idx].strip()
        # 原行可能是: "Frame Time: 0.033333"
        if frame_time_line.startswith("Frame Time:"):
            parts = frame_time_line.split(":")
            if len(parts) == 2:
                frame_time_value = parts[1].strip()
            else:
                frame_time_value = "0.033333"
        else:
            frame_time_value = "0.033333"
        f.write(f"Frame Time: {frame_time_value}\n")

        # 写 n_frames 行 T-pose
        for _ in range(n_frames):
            f.write(tpose_line)

    print(f"[OK] 已生成 T-pose BVH: {dst_bvh}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python make_tpose_from_lafan1.py 源动作.bvh 输出Tpose.bvh")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2]
    make_tpose_from_bvh(src, dst)
