import json
import os
import argparse

def add_collision_pairs_from_json(ik_config_path: str, collision_geoms_json_path: str, output_path: str = None) -> str:
    """
    从碰撞几何体JSON文件中读取数据并添加到IK配置文件的collision_pairs字段
    
    Args:
        ik_config_path: 输入IK配置路径
        collision_geoms_json_path: 碰撞几何体JSON文件路径
        output_path: 输出配置路径
        
    Returns:
        输出文件路径，如果失败则返回None
    """
    try:
        # 读取IK配置
        with open(ik_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 读取碰撞几何体数据
        with open(collision_geoms_json_path, 'r', encoding='utf-8') as f:
            collision_data = json.load(f)
        
        # 生成碰撞对
        collision_pairs = []
        for geom in collision_data.get("geoms", []):
            geom_name = geom.get("name")
            if geom_name:
                collision_pairs.append([[geom_name], ["floor"]])
        
        if not collision_pairs:
            print("[WARNING] 未找到有效的碰撞几何体")
            return None
        
        # 更新配置
        config["collision_pairs"] = collision_pairs
        
        # 确定输出路径
        if not output_path:
            base_name = os.path.splitext(ik_config_path)[0]
            output_path = f"{base_name}_with_collision.json"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 写入更新后的配置
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] 添加了 {len(collision_pairs)} 个碰撞对，配置已保存 -> {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] 处理碰撞几何体失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从碰撞几何体JSON文件中读取数据并添加到IK配置文件的collision_pairs字段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ik_config", "-i", required=True, help="IK配置文件的路径（输入）")
    parser.add_argument("--collision_geoms", "-c", required=True, help="碰撞几何体JSON文件的路径")
    parser.add_argument("--output", "-o", default=None, help="输出配置文件的路径（可选，如果不指定则自动生成）")
    
    args = parser.parse_args()
    
    try:
        result = add_collision_pairs_from_json(
            ik_config_path=args.ik_config,
            collision_geoms_json_path=args.collision_geoms,
            output_path=args.output
        )
        if result:
            print(f"[INFO] 处理完成，输出文件路径: {result}")
        else:
            print("[INFO] 处理未成功，未生成输出文件")
    except FileNotFoundError:
        print("[ERROR] 指定的文件未找到，请检查路径是否正确")
    except json.JSONDecodeError:
        print("[ERROR] 读取JSON文件时出错，请确保文件格式正确")
    except Exception as e:
        print(f"[ERROR] 处理过程中出现错误: {e}")


