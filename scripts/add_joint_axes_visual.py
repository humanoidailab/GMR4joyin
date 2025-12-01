#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import sys
import math
from pathlib import Path


def parse_axis_xyz(axis_str):
    parts = axis_str.strip().split()
    if len(parts) != 3:
        raise ValueError(f"axis xyz 格式错误: '{axis_str}'")
    return [float(p) for p in parts]


def axis_to_rpy(axis):
    """
    URDF 中 cylinder 默认沿 +Z 轴.
    根据 revolute joint 的 axis，把圆柱旋转过去。

    支持常见的 3 方向：
      Z 轴: (0,0,±1) -> rpy = (0,0,0)
      Y 轴: (0,±1,0) -> rpy = (-pi/2, 0, 0)
      X 轴: (±1,0,0) -> rpy = (0, pi/2, 0)
    """
    x, y, z = axis
    eps = 1e-6

    # Z 轴
    if abs(x) < eps and abs(y) < eps and abs(abs(z) - 1.0) < 1e-3:
        return "0 0 0"
    # Y 轴
    if abs(x) < eps and abs(z) < eps and abs(abs(y) - 1.0) < 1e-3:
        # 把 Z 轴转到 Y 轴: 绕 X 轴 -90°
        return f"{-math.pi/2:.7f} 0 0"
    # X 轴
    if abs(y) < eps and abs(z) < eps and abs(abs(x) - 1.0) < 1e-3:
        # 把 Z 轴转到 X 轴: 绕 Y 轴 +90°
        return f"0 {math.pi/2:.7f} 0"

    # 其他奇怪方向: 不旋转，至少有个轴
    # 如果你之后需要任意轴的精确对齐再扩展这里
    print(f"[WARN] 未识别的 axis={axis}，使用 rpy=0 0 0")
    return "0 0 0"


def add_axis_visual_to_link(link_elem, rpy_str,
                            radius=0.005, length=0.1):
    """
    在某个 <link> 下追加一个 <visual>，用圆柱体表示关节轴。
    不动惯性，只添加一个新的 visual。
    """
    visual = ET.SubElement(link_elem, "visual")

    origin = ET.SubElement(visual, "origin")
    origin.set("xyz", "0 0 0")
    origin.set("rpy", rpy_str)

    geometry = ET.SubElement(visual, "geometry")
    cyl = ET.SubElement(geometry, "cylinder")
    cyl.set("radius", f"{radius:.4f}")
    cyl.set("length", f"{length:.4f}")


def main(input_path, output_path,
         radius=0.005, length=0.1):
    tree = ET.parse(input_path)
    root = tree.getroot()

    # 先建立 link name -> element 的索引
    link_map = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name:
            link_map[name] = link

    processed_links = set()

    # 遍历所有 revolute joint
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue

        name = joint.get("name", "")
        child = joint.find("child")
        axis_elem = joint.find("axis")

        if child is None:
            print(f"[WARN] joint '{name}' 没有 <child>，跳过")
            continue
        child_link_name = child.get("link")
        if not child_link_name:
            print(f"[WARN] joint '{name}' 的 <child> 没有 link 属性，跳过")
            continue

        if axis_elem is None or "xyz" not in axis_elem.attrib:
            print(f"[WARN] joint '{name}' 没有 <axis xyz=...>，默认用 0 0 1")
            axis = [0.0, 0.0, 1.0]
        else:
            try:
                axis = parse_axis_xyz(axis_elem.get("xyz"))
            except Exception as e:
                print(f"[WARN] joint '{name}' axis 解析失败: {e}，默认用 0 0 1")
                axis = [0.0, 0.0, 1.0]

        rpy_str = axis_to_rpy(axis)

        if child_link_name not in link_map:
            print(f"[WARN] joint '{name}' 的子 link '{child_link_name}' 在 URDF 中找不到，跳过")
            continue

        # 如果一个 link 已经添加过轴可视化，就不重复添加
        if child_link_name in processed_links:
            continue

        link_elem = link_map[child_link_name]
        add_axis_visual_to_link(link_elem, rpy_str,
                                radius=radius, length=length)
        processed_links.add(child_link_name)

        print(f"[INFO] joint '{name}' -> link '{child_link_name}' 添加圆柱体，axis={axis}, rpy={rpy_str}")

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"[DONE] 已保存到: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python add_joint_axes_visual.py input.urdf output.urdf [radius] [length]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    r = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    l = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1

    main(in_path, out_path, radius=r, length=l)
