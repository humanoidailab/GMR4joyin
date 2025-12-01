#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整体缩放 URDF 中的几何尺寸 / 质量 / 惯量 / 各种 origin 位置.

用法:
    python scale_urdf.py input.urdf output.urdf --scale 0.01

默认会缩放:
    - 所有 link/joint/visual/collision/inertial 中 origin 的 xyz
    - sphere/cylinder/box 的几何尺寸
    - mesh 的 scale (如果有)

可选参数:
    --scale-mass      同时缩放 mass (m_new = m_old * s^3)
    --scale-inertia   同时缩放 inertia (I_new = I_old * s^5)
"""

import argparse
import xml.etree.ElementTree as ET


def parse_floats(s, n=None):
    vals = [float(x) for x in s.split()]
    if (n is not None) and (len(vals) != n):
        raise ValueError(f"Expect {n} floats, got {len(vals)} in '{s}'")
    return vals


def format_floats(vals):
    def _fmt(v):
        s = f"{v:.8f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    return " ".join(_fmt(v) for v in vals)


def scale_origin_xyz(elem, scale):
    """只缩放 <origin> 的 xyz，不动 rpy."""
    xyz = elem.get("xyz")
    if xyz is not None:
        vals = parse_floats(xyz, 3)
        vals = [v * scale for v in vals]
        elem.set("xyz", format_floats(vals))


def scale_geometry(geom_elem, scale):
    """缩放 geometry 下的 sphere / cylinder / box / mesh."""
    # sphere
    for sphere in geom_elem.findall("sphere"):
        r = sphere.get("radius")
        if r is not None:
            sphere.set("radius", format_floats([float(r) * scale]))

    # cylinder
    for cyl in geom_elem.findall("cylinder"):
        r = cyl.get("radius")
        if r is not None:
            cyl.set("radius", format_floats([float(r) * scale]))
        length = cyl.get("length")
        if length is not None:
            cyl.set("length", format_floats([float(length) * scale]))

    # box
    for box in geom_elem.findall("box"):
        size = box.get("size")
        if size is not None:
            vals = parse_floats(size, 3)
            vals = [v * scale for v in vals]
            box.set("size", format_floats(vals))

    # mesh
    for mesh in geom_elem.findall("mesh"):
        scale_attr = mesh.get("scale")
        if scale_attr is not None:
            vals = parse_floats(scale_attr)
            vals = [v * scale for v in vals]
            mesh.set("scale", format_floats(vals))


def scale_inertial(inertial_elem, scale, scale_mass=False, scale_inertia_flag=False):
    """缩放惯量块:

    - origin.xyz ~ s
    - mass ~ s^3 (可选)
    - inertia ~ s^5 (可选)
    """
    # origin
    origin = inertial_elem.find("origin")
    if origin is not None:
        scale_origin_xyz(origin, scale)

    # mass
    if scale_mass:
        mass = inertial_elem.find("mass")
        if mass is not None and mass.get("value") is not None:
            m = float(mass.get("value"))
            mass.set("value", format_floats([m * (scale ** 3)]))

    # inertia
    if scale_inertia_flag:
        inertia = inertial_elem.find("inertia")
        if inertia is not None:
            for attr in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]:
                v = inertia.get(attr)
                if v is not None:
                    val = float(v)
                    inertia.set(attr, format_floats([val * (scale ** 5)]))


def scale_urdf(tree, scale, scale_mass=False, scale_inertia_flag=False):
    root = tree.getroot()

    # 1) 缩放 link 相关
    for link in root.findall(".//link"):
        # link 直接挂 origin 的情况（不常见，但兼容一下）
        for origin in link.findall("origin"):
            scale_origin_xyz(origin, scale)

        # inertial
        inertial = link.find("inertial")
        if inertial is not None:
            scale_inertial(inertial, scale, scale_mass=scale_mass, scale_inertia_flag=scale_inertia_flag)

        # visual
        for visual in link.findall("visual"):
            v_origin = visual.find("origin")
            if v_origin is not None:
                scale_origin_xyz(v_origin, scale)   # ★ 缩放 <visual> 中几何构型的位置
            geom = visual.find("geometry")
            if geom is not None:
                scale_geometry(geom, scale)

        # collision
        for collision in link.findall("collision"):
            c_origin = collision.find("origin")
            if c_origin is not None:
                scale_origin_xyz(c_origin, scale)
            geom = collision.find("geometry")
            if geom is not None:
                scale_geometry(geom, scale)

    # 2) 缩放所有 joint 的 origin（关节位置）
    for joint in root.findall(".//joint"):
        j_origin = joint.find("origin")
        if j_origin is not None:
            scale_origin_xyz(j_origin, scale)

    # 注意：joint.axis 是单位向量，不缩放
    return tree


def main():
    parser = argparse.ArgumentParser(description="Scale all dimensions in a URDF by a given factor.")
    parser.add_argument("input_urdf", type=str, help="输入 URDF 文件路径")
    parser.add_argument("output_urdf", type=str, help="输出 URDF 文件路径")
    parser.add_argument("--scale", "-s", type=float, required=True, help="缩放系数，例如 0.01 或 10")
    parser.add_argument("--scale-mass", action="store_true", help="同时按 s^3 缩放 mass")
    parser.add_argument("--scale-inertia", action="store_true", help="同时按 s^5 缩放 inertia")
    args = parser.parse_args()

    tree = ET.parse(args.input_urdf)
    tree = scale_urdf(tree, args.scale, scale_mass=args.scale_mass, scale_inertia_flag=args.scale_inertia)

    # Python 3.9+ 可以美化输出
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(args.output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"Scaled URDF saved to: {args.output_urdf}")


if __name__ == "__main__":
    main()
