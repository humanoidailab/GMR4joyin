#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
URDF 3D 骨架可视化（只显示有自由度的关节，Z 轴拉长显示）.

特性：
- 根 link 放在世界坐标 (0, 0, 0)
- 做前向运动学，只用 joint 的 origin(xyz,rpy)，不考虑关节角度
- 只显示 DOF joints：type != "fixed"
- 图中显示的是关节 世界坐标 (x, y, z)，不显示关节名称
- 坐标标签字体更大，并且在关节位置基础上加偏移，避免重叠
- 在可视化阶段对 Z 方向进行缩放，让纵向看起来没那么挤
"""

import argparse
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ========== 可视化缩放参数 ==========
# 在显示上拉长 Z 轴的倍数（只影响渲染，不改变真实坐标）
SCALE_Z = 1.8


# ---------- 工具函数 ----------

def strip_tag(tag: str) -> str:
    """去掉 XML namespace，返回纯标签名"""
    return tag.split('}', 1)[-1]


def rpy_to_rot(r: float, p: float, y: float) -> np.ndarray:
    """roll-pitch-yaw -> 3x3 旋转矩阵（URDF: Rz(yaw)*Ry(pitch)*Rx(roll)）"""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,    0, 1],
    ])
    Ry = np.array([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp],
    ])
    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr],
    ])

    return Rz @ Ry @ Rx


def make_T(xyz, rpy) -> np.ndarray:
    """根据 xyz 和 rpy 构造 4x4 齐次变换矩阵"""
    R = rpy_to_rot(*rpy)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(xyz)
    return T


def set_axes_equal(ax):
    """让 3D 坐标轴比例一致（在缩放后的坐标上做）"""
    xs = ax.get_xlim3d()
    ys = ax.get_ylim3d()
    zs = ax.get_zlim3d()

    x_range = xs[1] - xs[0]
    y_range = ys[1] - ys[0]
    z_range = zs[1] - zs[0]
    max_range = max(x_range, y_range, z_range) / 2.0

    mid_x = (xs[0] + xs[1]) / 2.0
    mid_y = (ys[0] + ys[1]) / 2.0
    mid_z = (zs[0] + zs[1]) / 2.0

    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)


# ---------- URDF 解析 + FK ----------

def parse_urdf_joints(urdf_path):
    """
    解析 URDF 中所有 joint，返回：
    - joints: list of dict(name, type, parent, child, xyz, rpy)
    - links: set of link name
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = []
    links = set()

    for elem in root:
        tag = strip_tag(elem.tag)

        if tag == "joint":
            name = elem.get("name")
            jtype = elem.get("type", "fixed")
            parent = elem.find("parent").get("link")
            child = elem.find("child").get("link")

            origin = elem.find("origin")
            if origin is not None:
                xyz_str = origin.get("xyz", "0 0 0").split()
                rpy_str = origin.get("rpy", "0 0 0").split()
                xyz = list(map(float, xyz_str))
                rpy = list(map(float, rpy_str))
            else:
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]

            joints.append({
                "name": name,
                "type": jtype,
                "parent": parent,
                "child": child,
                "xyz": xyz,
                "rpy": rpy,
            })

            links.add(parent)
            links.add(child)

        elif tag == "link":
            links.add(elem.get("name"))

    return joints, links


def compute_world_transforms(joints, links, root_link=None):
    """
    对关节树做前向运动学（只用 origin xyz/rpy），根 link 放在世界 (0,0,0)

    返回：
    - joint_pos_world: dict(joint_name -> 3D np.array)
    - joint_type: dict(joint_name -> type)
    - edges: list of (parent_pos, child_pos, joint_name)
    """
    # parent link -> [outgoing joints]
    link_to_joints = {}
    for j in joints:
        link_to_joints.setdefault(j["parent"], []).append(j)

    # 自动找根 link（不作为 child 的那个）
    child_links = {j["child"] for j in joints}
    if root_link is None:
        root_candidates = list(links - child_links)
        if not root_candidates:
            raise RuntimeError("找不到 root link，请手动指定。")
        root_link = root_candidates[0]
        print(f"[INFO] 自动检测到 root link: {root_link}")
    else:
        print(f"[INFO] 使用用户指定的 root link: {root_link}")

    # 根 link 在世界系的变换 = 单位阵，即原点 (0,0,0)
    link_T_world = {root_link: np.eye(4)}

    joint_pos_world = {}
    joint_type = {}
    edges = []

    def dfs(link_name, parent_pos=None):
        T_link = link_T_world[link_name]
        link_origin_pos = T_link[:3, 3].copy()

        if parent_pos is None:
            parent_pos = link_origin_pos

        for j in link_to_joints.get(link_name, []):
            T_joint = T_link @ make_T(j["xyz"], j["rpy"])
            joint_pos = T_joint[:3, 3].copy()

            jname = j["name"]
            joint_pos_world[jname] = joint_pos
            joint_type[jname] = j["type"]

            # 画骨架时仍然把所有 joint 连起来，这样结构完整
            edges.append((parent_pos.copy(), joint_pos.copy(), jname))

            # child link 的坐标系原点与 joint 重合
            link_T_world[j["child"]] = T_joint
            dfs(j["child"], joint_pos)

    dfs(root_link)
    return joint_pos_world, joint_type, edges


# ---------- 可视化 ----------

def plot_skeleton_3d_dof(joint_pos_world, joint_type, edges,
                         title="URDF DOF Joints in 3D"):
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 画骨架（所有关节之间的连线，这样整个人形结构不会断）
    for p, c, jname in edges:
        xs = [p[0], c[0]]
        ys = [p[1], c[1]]
        # 显示时把 z 方向乘以 SCALE_Z
        zs = [p[2] * SCALE_Z, c[2] * SCALE_Z]
        ax.plot(xs, ys, zs, linewidth=0.8, alpha=0.7)

    # 收集所有 DOF 关节的位置，用于自适应偏移尺度
    all_dof_positions = []
    for jname, pos in joint_pos_world.items():
        if joint_type.get(jname, "fixed") == "fixed":
            continue
        all_dof_positions.append(pos)

    all_dof_positions = np.array(all_dof_positions) if len(all_dof_positions) > 0 else None

    # 根据模型尺寸自适应生成一个偏移量的基准长度（基于真实坐标，不含缩放）
    if all_dof_positions is not None:
        xyz_min = all_dof_positions.min(axis=0)
        xyz_max = all_dof_positions.max(axis=0)
        bbox_size = np.linalg.norm(xyz_max - xyz_min)
        # 取模型尺度的 5% 作为偏移基准长度
        base_offset = 0.05 * bbox_size if bbox_size > 0 else 0.05
    else:
        base_offset = 0.05

    # 具体偏移向量（可以按需要改方向和比例）
    offset_vec = np.array([0.3, 0.2, 0.5]) * base_offset

    for jname, pos in joint_pos_world.items():
        if joint_type.get(jname, "fixed") == "fixed":
            continue

        # 绘制关节点（z 用缩放后的）
        ax.scatter(pos[0], pos[1], pos[2] * SCALE_Z, s=30)

        # 坐标标签位置 = 原关节位置 + 偏移量（z 方向同样缩放后显示）
        label_pos = pos + offset_vec
        lx, ly, lz = label_pos
        lz_scaled = lz * SCALE_Z

        # 文本显示世界坐标 (x, y, z)，保留 3 位小数（使用真实坐标）
        ax.text(lx, ly, lz_scaled,
                f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})",
                fontsize=5,
                fontweight='bold')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)  (visual scaled)")
    ax.set_title(title)

    # 在缩放坐标系上做等比例 & 调整 box aspect，进一步拉长 Z
    set_axes_equal(ax)
    ax.set_box_aspect([1, 1, 1])  # 这里可以改成 [1,1,2] 再额外拉长

    # 视角：侧视图沿 +Y，看 X-Z 平面；可以自由调整
    ax.view_init(elev=10, azim=90)

    plt.tight_layout()
    plt.show()


# ---------- 主函数 ----------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize URDF DOF joints in 3D (Z stretched)."
    )
    parser.add_argument("urdf_path", help="Path to URDF file.")
    parser.add_argument("--root-link", default=None,
                        help="Root link name (optional, default: auto detect).")
    args = parser.parse_args()

    joints, links = parse_urdf_joints(args.urdf_path)
    joint_pos_world, joint_type, edges = compute_world_transforms(
        joints, links, root_link=args.root_link
    )
    plot_skeleton_3d_dof(
        joint_pos_world,
        joint_type,
        edges,
        title=f"URDF DOF Joints (world coords, Z scaled={SCALE_Z}) : {args.urdf_path}",
    )


if __name__ == "__main__":
    main()
