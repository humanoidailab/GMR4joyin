import numpy as np
import yaml
import re
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from scipy.spatial.transform import Rotation as R_scipy

# ==============================================================================
# ⚙️ 默认配置
# ==============================================================================
DEFAULT_BVH_PATH = "/home/tt/GMR-master4joyin/GMR4joyin/TPOSE.bvh"
DEFAULT_OUTPUT_PATH = "/home/tt/GMR-master4joyin/GMR4joyin/urdfmaker/geometry.yaml" # 输出文件名变更为 geometry
UNIT_SCALE = 0.01 
USER_SCALE = 1
DEFAULT_SCALE = UNIT_SCALE * USER_SCALE

# ==============================================================================
# 🛠️ 辅助工具
# ==============================================================================
def euler_to_matrix(z_deg, y_deg, x_deg):
    r = R_scipy.from_euler('zyx', [z_deg, y_deg, x_deg], degrees=True)
    return r.as_matrix()

def convert_numpy_types(data):
    """清洗 Numpy 类型，防止 yaml 保存报错"""
    if isinstance(data, dict): return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list): return [convert_numpy_types(v) for v in data]
    elif isinstance(data, np.generic): return data.item()
    elif isinstance(data, np.ndarray): return convert_numpy_types(data.tolist())
    else: return data

# ==============================================================================
# 1. 解析 BVH
# ==============================================================================
def parse_bvh_full(file_path):
    print(f"📂 读取 BVH: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    iterator = iter(lines)
    nodes = {}
    node_stack = []
    joint_order = []
    root_name = None
    motion_values = []
    
    try:
        while True:
            line = next(iterator)
            if line == "MOTION": break
            parts = line.split()
            token = parts[0]
            if token == 'ROOT' or token == 'JOINT':
                name = parts[1]
                if token == 'ROOT': root_name = name
                parent = node_stack[-1] if node_stack else None
                nodes[name] = {'name': name, 'parent': parent, 'children': [], 'offset': np.zeros(3), 'channels': [], 'is_end_site': False}
                if parent: nodes[parent]['children'].append(name)
                node_stack.append(name)
                joint_order.append(name)
            elif token == 'End':
                name = node_stack[-1] + "_EndSite"
                parent = node_stack[-1]
                nodes[name] = {'name': name, 'parent': parent, 'children': [], 'offset': np.zeros(3), 'channels': [], 'is_end_site': True}
                nodes[parent]['children'].append(name)
                node_stack.append(name)
            elif token == '}':
                if node_stack: node_stack.pop()
            elif token == 'OFFSET':
                nodes[node_stack[-1]]['offset'] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            elif token == 'CHANNELS':
                nodes[node_stack[-1]]['channels'] = parts[2:]
    except StopIteration: pass
    try:
        while True:
            line = next(iterator)
            if "Frames:" in line or "Frame Time:" in line: continue
            motion_values = [float(x) for x in line.split()]
            break
    except StopIteration: pass
    return nodes, root_name, joint_order, motion_values

# ==============================================================================
# 2. 计算 Raw T-Pose
# ==============================================================================
def compute_raw_tpose(nodes, root_name, joint_order, motion_data):
    print("🧮 计算 Raw T-Pose (Root Fixed @ 0,0,0)...")
    raw_positions = {}
    data_idx = 0
    def traverse(node_name, parent_matrix):
        nonlocal data_idx
        node = nodes[node_name]
        local_offset = node['offset'].copy()
        rot_matrix = np.eye(3)
        if not node['is_end_site']:
            channels = node['channels']
            vals = motion_data[data_idx : data_idx + len(channels)]
            data_idx += len(channels)
            z, y, x = 0, 0, 0
            for i, ch in enumerate(channels):
                val = vals[i]
                if 'Xposition' in ch: local_offset[0] += val
                elif 'Yposition' in ch: local_offset[1] += val
                elif 'Zposition' in ch: local_offset[2] += val
                elif 'Zrotation' in ch: z = val
                elif 'Yrotation' in ch: y = val
                elif 'Xrotation' in ch: x = val
            
            # 强制根节点归零，消除原始数据的位移和旋转干扰
            if node_name == root_name:
                local_offset = np.zeros(3)
                z, y, x = 0.0, 0.0, 0.0
            
            rot_matrix = euler_to_matrix(z, y, x)
        local_mat = np.eye(4)
        local_mat[:3, :3] = rot_matrix
        local_mat[:3, 3] = local_offset
        curr_global = parent_matrix @ local_mat
        raw_positions[node_name] = curr_global[:3, 3]
        for child in node['children']: traverse(child, curr_global)
    traverse(root_name, np.eye(4))
    return raw_positions

# ==============================================================================
# 3. 增强型自动校准 (Robust Search)
# ==============================================================================
def find_best_match(nodes, patterns):
    all_names = list(nodes.keys())
    for pattern in patterns:
        for name in all_names:
            if pattern.lower() in name.lower() and "endsite" not in name.lower():
                return name
    return None

def snap_vector_to_axis(vec):
    idx = np.argmax(np.abs(vec))
    snapped = np.zeros(3)
    snapped[idx] = 1.0 if vec[idx] > 0 else -1.0
    return snapped

def auto_calculate_clean_matrix(nodes, raw_positions, root_name, manual_args):
    print("🔧 自动校准坐标系 (Geometry Alignment)...")
    
    # 匹配模式 (优先级: CC > Biped > 通用)
    head_patterns = ["cc_base_head", "bip01_head", "head", "neck_02", "neck"]
    l_hand_patterns = ["cc_base_l_hand", "bip01_l_hand", "left_hand", "lefthand", "l_hand", "lhand", "left_wrist"]
    r_hand_patterns = ["cc_base_r_hand", "bip01_r_hand", "right_hand", "righthand", "r_hand", "rhand", "right_wrist"]

    # 尝试定位
    head = manual_args.head if manual_args.head else find_best_match(nodes, head_patterns)
    l_hand = manual_args.left_hand if manual_args.left_hand else find_best_match(nodes, l_hand_patterns)
    r_hand = manual_args.right_hand if manual_args.right_hand else find_best_match(nodes, r_hand_patterns)
    
    if not head or not l_hand or not r_hand:
        print("\n❌ 错误: 无法自动定位关键骨骼节点!")
        print("📋 部分骨骼列表:", list(nodes.keys())[:15])
        raise ValueError("Key nodes not found. Use --head, --left-hand args.")

    print(f"   ✅ 参考点: Head='{head}', L_Hand='{l_hand}'")

    p_root = raw_positions[root_name]
    p_head = raw_positions[head]
    p_l = raw_positions[l_hand]
    p_r = raw_positions[r_hand]

    # 向量计算
    vec_up_norm = (p_head - p_root) / np.linalg.norm(p_head - p_root)
    vec_side_norm = (p_l - p_r) / np.linalg.norm(p_l - p_r)
    
    # Forward = Side x Up
    vec_fwd_norm = np.cross(vec_side_norm, vec_up_norm)
    vec_fwd_norm /= np.linalg.norm(vec_fwd_norm)
    
    # Left = Up x Forward
    vec_left_norm = np.cross(vec_up_norm, vec_fwd_norm)

    # 轴吸附
    clean_fwd = snap_vector_to_axis(vec_fwd_norm)
    clean_left = snap_vector_to_axis(vec_left_norm)
    clean_up = snap_vector_to_axis(vec_up_norm)
    
    # 检查奇异
    if np.linalg.norm(np.cross(clean_fwd, clean_left)) < 0.5:
         print("⚠️ 警告: 轴吸附导致奇异，使用原始向量。")
         clean_fwd, clean_left, clean_up = vec_fwd_norm, vec_left_norm, vec_up_norm

    R_clean = np.array([clean_fwd, clean_left, clean_up])
    return R_clean

# ==============================================================================
# 4. 生成几何配置 (PURE GEOMETRY YAML)
# ==============================================================================
def generate_geometry_config(nodes, raw_positions, rotation_matrix, root_name, scale_factor):
    """
    生成仅包含拓扑和相对位置的 YAML。
    不包含：电机、旋转轴、限位等信息。
    """
    config = {
        'robot_name': 'bvh_robot_geometry',
        'export_settings': {
            'units': 'meters', 
            'scale_factor': scale_factor, 
            'aligned_axes': {'up': 'Z', 'fwd': 'X', 'left': 'Y'}
        },
        'links': [], 
        'joints_geometry': {}, 
        '_visualization': {}
    }
    
    # 1. 应用校准矩阵和缩放
    calibrated = {k: (rotation_matrix @ v) * scale_factor for k,v in raw_positions.items()}

    def process(node_name):
        node = nodes[node_name]
        curr = calibrated[node_name]
        
        # 存储可视化数据
        config['_visualization'][node_name] = {'pos': curr.tolist(), 'parent': node['parent']}
        
        if node['is_end_site']: return

        # 计算 Link 长度 (仅供后续可视化圆柱体参考)
        length = 0.05 * scale_factor
        if node['children']:
            length = np.linalg.norm(calibrated[node['children'][0]] - curr)
        
        config['links'].append({'name': node_name, 'length': round(length, 6)})
        
        # 计算相对于父节点的几何偏移 (Offset)
        if node['parent']:
            parent_pos = calibrated[node['parent']]
            offset = curr - parent_pos
            
            # ⚠️ 关键改变：这里不再定义 revolute 关节，只定义物理连接关系
            config['joints_geometry'][node_name] = {
                'parent': node['parent'],
                'child': node_name,
                'origin_xyz': [round(x, 6) for x in offset.tolist()],
                # 假设所有局部坐标系在 T-Pose 下都已对齐到全局，故 RPY 为 0
                'origin_rpy': [0, 0, 0] 
            }

        for child in node['children']: process(child)

    process(root_name)
    return config

# ==============================================================================
# 5. 可视化
# ==============================================================================
def visualize_data(vis_data):
    print("🎨 可视化几何结构...")
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        all_x, all_y, all_z = [], [], []
        for name, data in vis_data.items():
            curr = data['pos']
            all_x.append(curr[0]); all_y.append(curr[1]); all_z.append(curr[2])
            c = 'g' if 'EndSite' in name else 'r'
            ax.scatter(curr[0], curr[1], curr[2], c=c, s=20)
            if data['parent'] and data['parent'] in vis_data:
                p = vis_data[data['parent']]['pos']
                ax.plot([p[0], curr[0]], [p[1], curr[1]], [p[2], curr[2]], c='b')
            if "EndSite" not in name and any(k in name for k in ["Spine", "Head", "Hand", "Foot", "Hips", "Neck"]):
                ax.text(curr[0], curr[1], curr[2], name, fontsize=8)
        
        m = np.max([np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)]) / 2.0
        mx, my, mz = np.mean(all_x), np.mean(all_y), np.mean(all_z)
        ax.set_xlim(mx-m, mx+m); ax.set_ylim(my-m, my+m); ax.set_zlim(mz-m, mz+m)
        ax.set_xlabel('X (Fwd)'); ax.set_ylabel('Y (Left)'); ax.set_zlabel('Z (Up)')
        plt.show()
    except Exception as e: print(f"Vis error: {e}")

# ==============================================================================
# 主入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default=DEFAULT_BVH_PATH)
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--scale', '-s', type=float, default=DEFAULT_SCALE)
    parser.add_argument('--no-vis', action='store_true')
    parser.add_argument('--head', type=str)
    parser.add_argument('--left-hand', type=str)
    parser.add_argument('--right-hand', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found {args.input}"); sys.exit(1)

    # 1. 解析
    nodes, root_name, joint_order, motion_data = parse_bvh_full(args.input)
    if not motion_data: print("Error: No motion data"); sys.exit(1)

    # 2. 计算 & 校准
    raw_pos = compute_raw_tpose(nodes, root_name, joint_order, motion_data)
    try:
        clean_mat = auto_calculate_clean_matrix(nodes, raw_pos, root_name, args)
    except ValueError: sys.exit(1)
    except Exception as e: print(f"Unknown error: {e}"); sys.exit(1)
    
    # 3. 生成几何配置 (Geometry Only)
    config = generate_geometry_config(nodes, raw_pos, clean_mat, root_name, args.scale)
    
    # 4. 提取可视化数据并保存
    vis_data_backup = config.get('_visualization', {}).copy()
    if '_visualization' in config: del config['_visualization']
    
    print("🧹 清洗数据类型...")
    config_clean = convert_numpy_types(config)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)
    
    print(f"💾 保存 Geometry YAML 至: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(config_clean, f, sort_keys=False, default_flow_style=False)
    print("✅ 完成。")

    if not args.no_vis: visualize_data(vis_data_backup)

if __name__ == "__main__":
    main()