python scripts/generate_keypoint_mapping_bvh.py \
    --bvh_file TPOSE.bvh \
    --robot unitree_g1 \
    --loop \
    --robot_qpos_init pose_inits/unitree_g1_tpose.json \
    --ik_config_in general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json \
    --ik_config_out general_motion_retargeting/ik_configs/bvh_lafan1_to_g1_auto.json