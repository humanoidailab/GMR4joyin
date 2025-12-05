#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
URDF → MJCF 自动导出脚本（含 actuator 自动补全 + geom 自动命名）
作者: ChatGPT 助手（修改版）
功能:
  1. 从 URDF 读取模型
  2. 导出原始 MJCF XML
  3. 自动为 hinge/slide 关节补全 actuator（与关节名一致）
  4. 自动为所有缺失 name 的 geom 填充 name（基于所属 body 的 name）
  5. 输出最终 XML 文件
"""

import os
import xml.etree.ElementTree as ET

# ========= 用户路径配置 =========
URDF_PATH = "/home/tt/GMR-master4joyin/GMR4joyin/urdfmaker/humanoid_final1.urdf"
OUT_PATH  = "/home/tt/GMR-master4joyin/GMR4joyin/assets/joyin_description/joyin_description/humanoid_final1.xml"
RAW_PATH  = OUT_PATH.replace(".xml", "_raw.xml")

# ========= 核心逻辑 =========

def _iter_joints(body_elem):
    if body_elem is None:
        return
    for j in body_elem.findall("joint"):
        yield j
    for child in body_elem.findall("body"):
        yield from _iter_joints(child)

def _ensure_child(parent, tag):
    node = parent.find(tag)
    if node is None:
        node = ET.Element(tag)
        parent.append(node)
    return node

def _has_actuator_for_joint(actuator_elem, joint_name):
    if actuator_elem is None:
        return False
    for a in list(actuator_elem):
        if a.get("joint") == joint_name:
            return True
    return False

def _add_one_actuator(actuator_elem, joint_name, kind="motor",
                      ctrlrange=(-1.0, 1.0), gear=1.0, kp=None, kv=None):
    a = ET.Element(kind)
    a.set("name", joint_name)
    a.set("joint", joint_name)
    a.set("gear", f"{float(gear):.6g}")
    a.set("ctrlrange", f"{float(ctrlrange[0]):.6g} {float(ctrlrange[1]):.6g}")
    if kind == "position" and kp is not None:
        a.set("kp", f"{float(kp):.6g}")
    if kind == "velocity" and kv is not None:
        a.set("kv", f"{float(kv):.6g}")
    actuator_elem.append(a)

def _auto_name_geoms(root):
    """
    为所有缺失 name 的 geom 自动添加 name：
      - 如果某个 body 只有 1 个无名 geom，则 geom.name = body.name
      - 如果有多个无名 geom：
          第一个: body.name
          第二个: f"{body.name}_geom1"
          第三个: f"{body.name}_geom2"
          ...
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    # 遍历 worldbody 下所有 body（包括嵌套）
    for body in worldbody.findall(".//body"):
        bname = body.get("name")
        if not bname:
            continue

        # 找出该 body 下所有没有 name 的 geom
        unnamed_geoms = [g for g in body.findall("geom") if g.get("name") is None]
        if not unnamed_geoms:
            continue

        if len(unnamed_geoms) == 1:
            # 只有一个，无脑用 body 名
            unnamed_geoms[0].set("name", bname)
        else:
            # 多个：第一个与 body 同名，其余加后缀
            for idx, geom in enumerate(unnamed_geoms):
                if idx == 0:
                    geom.set("name", bname)
                else:
                    geom.set("name", f"{bname}_geom{idx}")

def add_actuators_to_mjcf(xml_in, xml_out,
                          actuator_kind="motor",
                          ctrl_min=-1.0, ctrl_max=1.0,
                          gear=1.0, kp=100.0, kv=2.0,
                          skip_fixed=True):
    tree = ET.parse(xml_in)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Invalid MJCF: <worldbody> not found.")
    actuator = _ensure_child(root, "actuator")

    # ⇩⇩⇩ 新增：自动给 geom 命名 ⇩⇩⇩
    _auto_name_geoms(root)
    # ⇧⇧⇧ 新增：自动给 geom 命名 ⇧⇧⇧

    joints = list(_iter_joints(worldbody))

    added = 0
    skipped = 0
    for j in joints:
        jname = j.get("name")
        jtype = (j.get("type") or "hinge").lower()
        if not jname:
            skipped += 1
            continue
        if skip_fixed and jtype in ("fixed",):
            skipped += 1
            continue
        if jtype in ("ball", "free"):
            skipped += 1
            continue
        if jtype not in ("hinge", "slide"):
            skipped += 1
            continue
        if _has_actuator_for_joint(actuator, jname):
            skipped += 1
            continue
        _add_one_actuator(actuator, jname, kind=actuator_kind,
                          ctrlrange=(ctrl_min, ctrl_max),
                          gear=gear, kp=kp, kv=kv)
        added += 1

    try:
        ET.indent(tree, space="  ", level=0)
    except Exception:
        pass

    tree.write(xml_out, encoding="utf-8", xml_declaration=True)
    print(f"[INFO] 新增 {added} 个 actuator，跳过 {skipped} 个。输出: {xml_out}")

def urdf_to_mjcf_with_act():
    import mujoco
    os.environ.setdefault("MUJOCO_GL", "egl")  # 防止无显卡环境报错

    # Step 1: URDF → MJCF
    model = mujoco.MjModel.from_xml_path(URDF_PATH)
    mujoco.mj_saveLastXML(RAW_PATH, model)
    print(f"[INFO] URDF 转换完成: {RAW_PATH}")

    # Step 2: 自动补全 actuator + 自动 geom 命名
    add_actuators_to_mjcf(
        xml_in=RAW_PATH,
        xml_out=OUT_PATH,
        actuator_kind="motor",  # 可改成 "position" 或 "velocity"
        ctrl_min=-1.0,
        ctrl_max=1.0,
        gear=1.0,
        kp=100.0,
        kv=2.0,
        skip_fixed=True,
    )
    print(f"[✅] 最终带执行器的模型已保存: {OUT_PATH}")

if __name__ == "__main__":
    urdf_to_mjcf_with_act()
