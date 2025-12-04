import yaml
import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# ==============================================================================
# ⚙️ 默认配置
# ==============================================================================
DEFAULT_BVHYAML = "/home/tt/GMR-master4joyin/GMR4joyin/urdfmaker/geometry.yaml"
DEFAULT_URDFYAML = "/home/tt/GMR-master4joyin/GMR4joyin/urdfmaker/urdf.yaml"
DEFAULT_OUTPUT = "/home/tt/GMR-master4joyin/GMR4joyin/urdfmaker/humanoid_final.urdf"

# 物理参数
DENSITY_LINK = 100.0 # 密度
LINK_RADIUS = 0.02    # 默认连杆粗细

class MergedURDFGenerator:
    def __init__(self, geo_data, kin_data):
        self.geo = geo_data
        self.kin = kin_data
        self.defaults = kin_data.get('default_motor', {})
        self.default_motor_geom = self.defaults.get('geometry', {'radius': 0.04, 'length': 0.05})
        self.default_motor_mass = self.defaults.get('mass', 0.5)
        
        self.robot_name = self.kin.get('robot_name', 'g1_generated')
        self.root_name_map = self.kin.get('root_name_map', 'pelvis')
        self.base_mass = self.kin.get('base_mass', 0)
        
        # 🚀 读取附件配置
        self.attachments_config = self.kin.get('attachments', {})
        
        self.xml_links = {} 
        self.xml_joints = []
        self.xml_transmissions = []
        
        # ... (children_map 和 axes_map 构建代码不变) ...
        self.children_map = {}
        for child_name, data in self.geo['joints_geometry'].items():
            p = data['parent']
            if p not in self.children_map: self.children_map[p] = []
            self.children_map[p].append(child_name)
        self.axes_map = {'X': "1 0 0", 'Y': "0 1 0", 'Z': "0 0 1"}

    # ... (数学函数和惯量计算函数保持不变) ...
    # get_rotation_rpy, get_motor_rpy, calculate_xxx_inertia, add_inertia 
    # init_link, add_shape, add_motor, add_bone, create_joint, process_node
    # 请确保这些函数都在 (可以直接复用 v9 版本的代码)
    
    # ⚠️ 这里为了完整性，我把数学函数简写带过，实际运行请保留 v9 的完整实现
    def get_rotation_rpy(self, vec):
        vec = np.array(vec); norm = np.linalg.norm(vec)
        if norm < 1e-6: return [0,0,0], 0.0
        z_axis = np.array([0,0,1]); target_dir = vec / norm
        cross = np.cross(z_axis, target_dir); dot = np.dot(z_axis, target_dir)
        if np.linalg.norm(cross) < 1e-6: return ([0,0,0], norm) if dot > 0 else ([0,np.pi,0], norm)
        angle = np.arccos(np.clip(dot, -1.0, 1.0)); axis = cross / np.linalg.norm(cross)
        return R_scipy.from_rotvec(axis * angle).as_euler('xyz'), norm
    
    def get_motor_rpy(self, c): return "0 1.5708 0" if c=='X' else ("1.5708 0 0" if c=='Y' else "0 0 0")
    
    def calculate_cylinder_inertia(self, m, r, l): return (1/12)*m*(3*r**2+l**2), (1/12)*m*(3*r**2+l**2), 0.5*m*r**2
    def calculate_sphere_inertia(self, m, r): i=0.4*m*r**2; return i,i,i
    def calculate_box_inertia(self, m, x, y, z): return (1/12)*m*(y**2+z**2), (1/12)*m*(x**2+z**2), (1/12)*m*(x**2+y**2)

    def add_inertia(self, link_data, mass, com_xyz, i_local_xyz):
        current_m = link_data['mass']; current_com = link_data['com']; current_I = link_data['I']
        new_m = mass; new_com = np.array(com_xyz); new_I = np.array(i_local_xyz)
        total_m = current_m + new_m
        if total_m < 1e-9: final_com = np.zeros(3); final_I = np.zeros(3)
        else:
            final_com = (current_m * current_com + new_m * new_com) / total_m
            def parallel(Ic, m, v): 
                dx,dy,dz=v; return Ic + m * np.array([dy**2+dz**2, dx**2+dz**2, dx**2+dy**2])
            term1 = parallel(current_I, current_m, current_com - final_com)
            term2 = parallel(new_I, new_m, new_com - final_com)
            final_I = term1 + term2
        link_data['mass'] = total_m; link_data['com'] = final_com; link_data['I'] = final_I

    def init_link(self, name):
        if name not in self.xml_links:
            self.xml_links[name] = {'visuals': [], 'collisions': [], 'mass': 0.0, 'com': np.array([0.,0.,0.]), 'I': np.array([0.,0.,0.])}

    def add_shape(self, link_name, shape_type, dims, xyz, rpy_euler, mass, color, collision=True):
        if link_name not in self.xml_links: self.init_link(link_name)
        rpy_str = f"{rpy_euler[0]:.4f} {rpy_euler[1]:.4f} {rpy_euler[2]:.4f}" if isinstance(rpy_euler, (list, np.ndarray)) else rpy_euler
        xyz_str = f"{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}"
        
        geo, ix, iy, iz = "", 0, 0, 0
        if shape_type == 'cylinder':
            geo = f'<cylinder radius="{dims[0]}" length="{dims[1]}"/>'
            ix, iy, iz = self.calculate_cylinder_inertia(mass, dims[0], dims[1])
        elif shape_type == 'sphere':
            geo = f'<sphere radius="{dims[0]}"/>'
            ix, iy, iz = self.calculate_sphere_inertia(mass, dims[0])
        elif shape_type == 'box':
            geo = f'<box size="{dims[0]} {dims[1]} {dims[2]}"/>'
            ix, iy, iz = self.calculate_box_inertia(mass, dims[0], dims[1], dims[2])

        c_val = "0.5 0.5 0.5 1"
        if color == "Motor": c_val = "0.8 0.2 0.2 1"
        elif color == "Hand": c_val = "0.2 0.2 0.8 1"
        elif color == "Base": c_val = "0.2 0.2 0.2 1"
        elif color == "Foot": c_val = "0.1 0.1 0.1 1"
        
        vis = f'<visual><origin xyz="{xyz_str}" rpy="{rpy_str}"/><geometry>{geo}</geometry><material name="{color}"><color rgba="{c_val}"/></material></visual>'
        self.xml_links[link_name]['visuals'].append(vis)
        if collision:
            col = f'<collision><origin xyz="{xyz_str}" rpy="{rpy_str}"/><geometry>{geo}</geometry></collision>'
            self.xml_links[link_name]['collisions'].append(col)
        self.add_inertia(self.xml_links[link_name], mass, xyz, [ix, iy, iz])

    def add_motor(self, link_name, axis_char, custom_geom, custom_inertial):
        mr = custom_geom.get('radius', self.default_motor_geom['radius']) if custom_geom else self.default_motor_geom['radius']
        ml = custom_geom.get('length', self.default_motor_geom['length']) if custom_geom else self.default_motor_geom['length']
        rpy = self.get_motor_rpy(axis_char)
        mass = float(custom_inertial.get('mass', self.default_motor_mass)) if custom_inertial else self.default_motor_mass
        self.add_shape(link_name, 'cylinder', [mr, ml], [0,0,0], rpy, mass, "Motor", collision=True)

    def add_bone(self, link_name, vec_start, vec_end, is_thin=False):
        offset = vec_end - vec_start
        rpy, length = self.get_rotation_rpy(offset)
        mid_pos = vec_start + offset / 2.0
        radius = 0.015 if is_thin else LINK_RADIUS
        draw_len = max(length - 0.02, 0.01)
        vol = np.pi * (radius**2) * draw_len
        mass = vol * DENSITY_LINK
        self.add_shape(link_name, 'cylinder', [radius, draw_len], mid_pos, rpy, mass, "Bone", collision=False)

    def add_footplate(self, link_name, base_pos):
        pos = base_pos + np.array([0.04, 0, -0.02]) 
        self.add_shape(link_name, 'box', [0.16, 0.06, 0.015], pos, [0,0,0], 0.3, "Foot", collision=True)

    def create_joint(self, name, type, parent, child, xyz, rpy, axis, limits):
        lim_str = ""
        if type == 'revolute':
            l = limits if limits else self.defaults
            lim_str = f'<limit lower="{l.get("lower",-3)}" upper="{l.get("upper",3)}" effort="{l.get("effort",30)}" velocity="{l.get("velocity",10)}"/>'
            self.xml_transmissions.append(f'<transmission name="trans_{name}"><type>transmission_interface/SimpleTransmission</type><joint name="{name}"><hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface></joint><actuator name="motor_{name}"><mechanicalReduction>1</mechanicalReduction></actuator></transmission>')
        
        xml = f"""
  <joint name="{name}" type="{type}">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}" rpy="{rpy[0]:.4f} {rpy[1]:.4f} {rpy[2]:.4f}"/>
    <axis xyz="{axis}"/>
    <dynamics damping="0.1" friction="0.0"/>
    {lim_str}
  </joint>"""
        self.xml_joints.append(xml)

    def process_node(self, bvh_node, current_urdf_link_name, accumulated_xyz, accumulated_rpy):
        kin_info = self.kin['joints'].get(bvh_node)
        
        is_new_joint = False
        rotation_order = kin_info.get('rotation_order', []) if kin_info else []
        active_dofs = kin_info.get('active_dof', []) if kin_info else []
        if rotation_order and active_dofs: is_new_joint = True

        if is_new_joint and bvh_node != list(self.geo['links'])[0]['name']:
            naming = kin_info.get('naming', {})
            motor_geoms = kin_info.get('motor_geometry', {})
            motor_inertials = kin_info.get('motor_inertial', {})
            limits = kin_info.get('limits', {})
            prev_link = current_urdf_link_name
            
            for i, axis in enumerate(rotation_order):
                is_last = (i == len(rotation_order) - 1)
                axis = axis.upper()
                is_active = axis in active_dofs
                
                tgt_link = naming.get(axis, {}).get('link', f"{bvh_node}_link")
                tgt_joint = naming.get(axis, {}).get('joint', f"{bvh_node}_joint")
                self.init_link(tgt_link)
                
                if is_active:
                    self.add_motor(tgt_link, axis, motor_geoms.get(axis), motor_inertials.get(axis))
                
                j_xyz = accumulated_xyz if i == 0 else [0,0,0]
                j_rpy = accumulated_rpy if i == 0 else [0,0,0]
                j_type = 'revolute' if is_active else 'fixed'
                
                self.create_joint(tgt_joint, j_type, prev_link, tgt_link, j_xyz, j_rpy, self.axes_map.get(axis, "1 0 0"), limits.get(axis))
                prev_link = tgt_link
            
            current_urdf_link_name = prev_link
            accumulated_xyz = np.array([0.,0.,0.])
            accumulated_rpy = np.array([0.,0.,0.])

        # Geometric Shapes
        children = self.children_map.get(bvh_node, [])
        if children:
            child = children[0]
            offset = np.array(self.geo['joints_geometry'][child]['origin_xyz'])
            
            if current_urdf_link_name == self.root_name_map:
                if len(self.xml_links[current_urdf_link_name]['visuals']) == 0:
                    self.add_shape(current_urdf_link_name, 'sphere', [0.03,0,0], accumulated_xyz, [0,0,0], self.base_mass, "Base", collision=True)
            else:
                is_thin = ("Foot" in bvh_node or "Ankle" in bvh_node)
                self.add_bone(current_urdf_link_name, accumulated_xyz, accumulated_xyz + offset, is_thin)
            
            if "Hand" in bvh_node or "hand" in bvh_node:
                self.add_shape(current_urdf_link_name, 'sphere', [0.03,0,0], accumulated_xyz, [0,0,0], 0.2, "Hand", collision=True)
        else:
            if current_urdf_link_name != self.root_name_map:
                self.add_shape(current_urdf_link_name, 'sphere', [0.015,0,0], accumulated_xyz, [0,0,0], 0.05, "End", collision=True)

        if "Foot" in bvh_node or "Ankle" in bvh_node:
             self.add_footplate(current_urdf_link_name, accumulated_xyz)

        for child in children:
            offset = np.array(self.geo['joints_geometry'][child]['origin_xyz'])
            self.process_node(child, current_urdf_link_name, accumulated_xyz + offset, accumulated_rpy)

    # 🚀 核心新增：后处理附件 (Post-process Attachments)
    def apply_attachments(self):
        print("🔧 Applying Extra Attachments...")
        # 遍历 urdf.yaml 里的 attachments 配置
        for link_name, items in self.attachments_config.items():
            if link_name not in self.xml_links:
                print(f"⚠️ Warning: Attachment target link '{link_name}' not found in URDF. Skipping.")
                continue
            
            print(f"   -> Adding {len(items)} items to '{link_name}'")
            for item in items:
                name = item.get('name', 'extra')
                mass = item.get('mass', 0.1)
                shape = item.get('type', 'box')
                dims = item.get('dims', [0.05, 0.05, 0.05])
                pos = item.get('pos', [0,0,0])
                rpy = item.get('rpy', [0,0,0])
                color = item.get('color', 'Hand') # 借用 Hand 的蓝色或自定义
                
                # 调用通用的 add_shape 方法
                # 这会自动添加 Visual, Collision 并合并 Inertia
                self.add_shape(link_name, shape, dims, pos, rpy, mass, color, collision=True)

    def generate_xml(self):
        body = ""
        for name, data in self.xml_links.items():
            m = data['mass']; i = data['I']; c = data['com']
            # 兜底防止0质量
            if m < 1e-5: m = 0.01; i = [1e-5]*3
            
            body += f"""
  <link name="{name}">
    <inertial>
      <mass value="{m:.4f}"/>
      <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>
      <inertia ixx="{i[0]:.6f}" ixy="0" ixz="0" iyy="{i[1]:.6f}" iyz="0" izz="{i[2]:.6f}"/>
    </inertial>
    {''.join(data['visuals'])}
    {''.join(data['collisions'])}
  </link>"""
        body += "\n".join(self.xml_joints) + "\n" + "\n".join(self.xml_transmissions)
        return f'<robot name="{self.robot_name}">{body}\n</robot>'

    def print_mass_report(self):
        print("\n" + "="*50)
        print(f"📊 ROBOT MASS REPORT: {self.robot_name}")
        print("="*50)
        print(f"{'Link Name':<35} | {'Mass (kg)':>10}")
        print("-" * 49)
        total = 0.0
        for name in sorted(self.xml_links.keys()):
            m = self.xml_links[name]['mass']
            if m < 1e-5: m = 0.01
            total += m
            print(f"{name:<35} | {m:10.4f}")
        print("-" * 49)
        print(f"{'TOTAL MASS':<35} | {total:10.4f} kg")
        print("="*50 + "\n")

    def run(self):
        bvh_root = list(self.geo['links'])[0]['name']
        self.init_link(self.root_name_map)
        
        # 1. 递归生成主体结构
        self.process_node(bvh_root, self.root_name_map, np.array([0.,0.,0.]), np.array([0.,0.,0.]))
        
        # 2. 🚀 应用额外附件 (Attachments)
        self.apply_attachments()
        
        self.print_mass_report()
        return self.generate_xml()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvhyaml', default=DEFAULT_BVHYAML)
    parser.add_argument('--urdfyaml', default=DEFAULT_URDFYAML)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not os.path.exists(args.bvhyaml): print("Geo missing"); sys.exit(1)
    if not os.path.exists(args.urdfyaml): print("Kin missing"); sys.exit(1)

    with open(args.bvhyaml, 'r') as f: geo = yaml.safe_load(f)
    with open(args.urdfyaml, 'r') as f: kin = yaml.safe_load(f)
    
    gen = MergedURDFGenerator(geo, kin)
    with open(args.output, 'w') as f: f.write(gen.run())
    print(f"✅ Merged URDF Generated: {args.output}")

if __name__ == "__main__":
    main()