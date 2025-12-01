import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT, IK_CONFIG_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print
from .model_constructer import ModelConstructor


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=False,
                motion_fps=30,
                transparent_robot=0,
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480,
                keyboard_callback=None,
                collision_detection=True,
                ):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        
        if collision_detection:
            self.model = ModelConstructor.construct_model(
                robot_type, 
                collision_pairs=None,
                add_ground=True,
                add_visual=True
            )
        else:
            self.model = mj.MjModel.from_xml_path(str(self.xml_path))

        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video


        self.viewer = mjv.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False, 
            key_callback=keyboard_callback
            )      

        self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)

    def get_body_transform(self, body_name):
        """获取指定身体的全局位置和旋转矩阵"""
        try:
            body_id = self.model.body(body_name).id
            
            # 获取身体位置和旋转
            pos = self.data.xpos[body_id].copy()
            mat = self.data.xmat[body_id].reshape(3, 3).copy()
            
            return pos, mat
        except:
            print(f"[Warning] Body '{body_name}' not found in the model")
            return None, None
        
    def step(self, 
             # robot data
             root_pos, root_rot, dof_pos, 
             # human data
             human_motion_data=None, 
             show_human_body_name=False,
             # scale for human point visualization
             human_point_scale=0.1,
             # human pos offset add for visualization    
             human_pos_offset=np.array([0.0, 0.0, 0]),
             # robot body frames visualization
             robot_frames=None,
             robot_frame_scale=0.2,
             show_robot_body_name=False,
             # rate limit
             rate_limit=True, 
             follow_camera=False,
             # ==== 新增：质心可视化相关参数 ====
             show_com_projection=True,
             com_ball_radius=0.02,   # 质心小球半径
             ground_height=0.0,      # 投影落到的地面高度（默认 z = 0）
             ):
        """
        By default visualize robot motion.
        Also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.
        robot_frames is a list of body names to visualize coordinate frames.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        # ---- 更新机器人 qpos ----
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot  # quat: scalar first (wxyz) for Mujoco
        self.data.qpos[7:] = dof_pos
        
        # 更新动力学量（xpos/xmat/subtree_com 等）
        mj.mj_forward(self.model, self.data)
        
        # ---- 跟随机器人移动相机（可选）----
        if follow_camera:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.viewer.cam.distance = self.viewer_cam_distance
            self.viewer.cam.elevation = -10  # 轻微俯视
            # self.viewer.cam.azimuth = 180  # 如需固定方位可以打开
        
        # 清空上一帧的用户自定义几何（marker）
        self.viewer.user_scn.ngeom = 0
        
        # ==========================================
        # 🔴🟢 画质心与地面投影（小球标记）
        # ==========================================
        if show_com_projection:
            # MuJoCo: subtree_com[0] 是 world body 子树的质心 -> 通常就是整机质心（世界系）
            com_world = np.array(self.data.subtree_com[0], dtype=float)  # (3,)
            com_proj = com_world.copy()
            com_proj[2] = ground_height  # 重力方向投影到地面 z = ground_height

            scn = self.viewer.user_scn

            # 🔴 真实质心（红球）
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                g.type = mj.mjtGeom.mjGEOM_SPHERE
                g.size[:] = [com_ball_radius, com_ball_radius, com_ball_radius]
                g.pos[:] = com_world
                g.rgba[:] = [1.0, 0.1, 0.1, 1.0]
                g.mat[:, :] = np.eye(3)   # 👈 这里

                scn.ngeom += 1

            # 🟢 地面投影（绿球）
            if scn.ngeom < scn.maxgeom:
                g = scn.geoms[scn.ngeom]
                g.type = mj.mjtGeom.mjGEOM_SPHERE
                g.size[:] = [com_ball_radius, com_ball_radius, com_ball_radius]
                g.pos[:] = com_proj
                g.rgba[:] = [0.1, 1.0, 0.1, 1.0]
                g.mat[:, :] = np.eye(3)   # 👈 这里

                scn.ngeom += 1

        
        # ==========================================
        # 人体数据可视化（任务目标/原始人类动作等）
        # ==========================================
        if human_motion_data is not None:
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None,
                )
        
        # ==========================================
        # 机器人各 body 的坐标系可视化（可选）
        # ==========================================
        if robot_frames is not None:
            for robot_body_name in robot_frames:
                pos, mat = self.get_body_transform(robot_body_name)
                if pos is not None and mat is not None:
                    draw_frame(
                        pos,
                        mat,
                        self.viewer,
                        robot_frame_scale,
                        joint_name=robot_body_name if show_robot_body_name else None,
                    )

        # ==========================================
        # 渲染 & 限速 & 录像
        # ==========================================
        self.viewer.sync()
        if rate_limit:
            self.rate_limiter.sleep()

        if self.record_video:
            # 用 renderer 做 offscreen 渲染写入视频
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)



    def close(self):
        self.viewer.close()
        time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
