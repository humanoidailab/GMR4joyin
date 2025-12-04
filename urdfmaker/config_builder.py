import yaml
import os
import sys
import argparse

# ==============================================================================
# ⚙️ 默认路径
# ==============================================================================
PATH_LIB = "/home/tt/GMR-master4joyin/GMR4joyin/urdfmaker/config/motor_library.yaml"
PATH_MAP = "/home/tt/GMR-master4joyin/GMR4joyin/urdfmaker/config/robot_mapping.yaml"
PATH_OUT = "/home/tt/GMR-master4joyin/GMR4joyin/urdfmaker/urdf.yaml"

class ConfigBuilder:
    def __init__(self, lib_data, map_data):
        self.lib = lib_data['motors']
        self.map = map_data
        self.default_motor_name = map_data.get('default_motor')
        
        # 最终输出结构
        self.output = {
            'robot_name': map_data.get('robot_name', 'generated_robot'),
            'root_name_map': map_data.get('root_map', 'pelvis'),
            'base_mass': map_data.get('base_mass', 0.0), # 读取自定义基座质量
            'default_motor': {
                'effort': 0, 'velocity': 0, 
                'geometry': {'radius': 0.04, 'length': 0.05}
            },
            'joints': {},
            'attachments': {} # 🚀 新增：额外附件存储
        }
        
        # 辅助映射表：BVH节点 -> 最终生成的URDF Link名
        self.bvh_to_urdf_link = {}

    def calculate_cylinder_inertia(self, mass, radius, length):
        """自动计算圆柱体惯量"""
        ixx_iyy = (1/12) * mass * (3 * radius**2 + length**2)
        izz = 0.5 * mass * radius**2
        return {'ixx': ixx_iyy, 'iyy': ixx_iyy, 'izz': izz}

    def process(self):
        # 1. 记录 Root 的映射关系
        root_bvh = "Hips" # 假设 BVH Root 是 Hips，或者你可以遍历 map 找 parent
        # 在这里我们简单记录用户定义的 root_map
        # 注意：实际上我们需要知道哪个 BVH 节点对应 Root。
        # 通常 robot_mapping 不显式写 Root 的关节配置(因为它是浮动基座)，
        # 但如果有 Hips 的配置，我们会处理。
        # 暂时我们只处理 process 循环中出现的节点。
        
        # 遍历映射表中的每个 BVH 节点
        for bvh_node, axes_config in self.map['map'].items():
            
            joint_entry = {
                'rotation_order': [],
                'active_dof': [],
                'naming': {},
                'limits': {},
                'motor_geometry': {},
                'motor_inertial': {}
            }
            
            ordered_axes = list(axes_config.keys())
            joint_entry['rotation_order'] = ordered_axes
            joint_entry['active_dof'] = ordered_axes
            
            # 记录该节点产生的最后一个 Link 的名字 (通常是实体连杆)
            last_link_name = None
            
            for i, (axis, props) in enumerate(axes_config.items()):
                # ... (电机参数处理逻辑保持不变) ...
                motor_name = props.get('motor', self.default_motor_name)
                motor_data = self.lib.get(motor_name, self.lib.get(self.default_motor_name, {}))
                
                eff = motor_data.get('effort', 30.0)
                vel = motor_data.get('velocity', 20.0)
                rad = motor_data.get('radius', 0.04)
                len_ = motor_data.get('length', 0.05)
                mass = motor_data.get('mass', 0.5)
                
                if 'inertia' in motor_data:
                    inertia = motor_data['inertia']
                else:
                    inertia = self.calculate_cylinder_inertia(mass, rad, len_)
                
                base_name = props.get('name', f"{bvh_node}_{axis}")
                link_name = f"{base_name}_link"
                joint_name = f"{base_name}_joint"
                
                joint_entry['naming'][axis] = {'link': link_name, 'joint': joint_name}
                
                joint_entry['limits'][axis] = {
                    'lower': props.get('lower', -3.14),
                    'upper': props.get('upper', 3.14),
                    'effort': eff,
                    'velocity': vel
                }
                
                joint_entry['motor_geometry'][axis] = {'radius': rad, 'length': len_}
                joint_entry['motor_inertial'][axis] = {'mass': mass, 'inertia': inertia}
                
                # 更新最后一个 Link 的名字
                last_link_name = link_name

            self.output['joints'][bvh_node] = joint_entry
            
            # 建立映射：BVH节点 -> 该节点生成的最终 URDF Link
            if last_link_name:
                self.bvh_to_urdf_link[bvh_node] = last_link_name

        # 🚀 处理附件 (Attachments)
        raw_attachments = self.map.get('attachments', {})
        processed_attachments = {}
        
        root_urdf_name = self.output['root_name_map']
        
        for bvh_target, items in raw_attachments.items():
            # 解析目标 Link
            target_link = None
            
            # 情况1: 是 Root (如 Hips)
            if bvh_target == "Hips" or bvh_target == "Root":
                target_link = root_urdf_name
            # 情况2: 在上面的循环中生成过
            elif bvh_target in self.bvh_to_urdf_link:
                target_link = self.bvh_to_urdf_link[bvh_target]
            # 情况3: 用户直接写了 URDF Link 的名字 (兜底)
            else:
                target_link = bvh_target # 假设用户知道自己在做什么
            
            if target_link not in processed_attachments:
                processed_attachments[target_link] = []
            
            processed_attachments[target_link].extend(items)
            
        self.output['attachments'] = processed_attachments

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.output, f, sort_keys=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', default=PATH_LIB)
    parser.add_argument('--map', default=PATH_MAP)
    parser.add_argument('--output', default=PATH_OUT)
    args = parser.parse_args()

    if not os.path.exists(args.lib) or not os.path.exists(args.map):
        print("❌ Config files missing.")
        sys.exit(1)

    print(f"📚 Loading Library: {args.lib}")
    with open(args.lib, 'r') as f: lib = yaml.safe_load(f)
    print(f"🗺️ Loading Mapping: {args.map}")
    with open(args.map, 'r') as f: map_data = yaml.safe_load(f)
    
    print("⚙️ Building full configuration with attachments...")
    builder = ConfigBuilder(lib, map_data)
    builder.process()
    
    print(f"💾 Saving URDF Config to: {args.output}")
    builder.save(args.output)
    print("✅ Ready.")

if __name__ == "__main__":
    main()