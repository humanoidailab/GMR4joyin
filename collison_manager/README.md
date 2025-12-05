# 运动重定向中碰撞避免

## 安装要求
```bash
pip install dm-control
```

## 基础配置
-assets中存在配置好机器人碰撞模型的.xml文件 \
-general_motion_retargeting/ik_configs中添加collison_pairs配置

## 生成包含机器人模型碰撞几何体名称的.xml文件并提取出其碰撞几何体名称
```bash
python collison_manager/fix_collision_geoms.py \
    --input assets/unitree_g1/g1_custom_collision_29dof_origin.xml \
    --output assets/unitree_g1/g1_custom_collision_29dof_fixed.xml \
    --extract-geoms assets/unitree_g1/g1_custom_collision_29dofs_geoms.txt \
    --transparent
```
-添加--transparent表明将碰撞模型透明化 \
-提取出机器人模型中可用于碰撞检测的碰撞几何体

## 添加collison_pairs
```bash
python collison_manager/add_collision_pairs.py \
    --i general_motion_retargeting/ik_configs/bvh_lafan1_to_g1_auto.json \
    -c assets/unitree_g1/g1_custom_collision_29dofs_geoms.json \
    -o general_motion_retargeting/ik_configs/bvh_lafan1_to_g1_auto1.json
```