
import numpy as np
import mujoco as mj
from .params import ROBOT_XML_DICT
import xml.etree.ElementTree as ET
from dm_control import mjcf
from lxml import etree

class ModelConstructor:
    """独立的模型构建器，用于创建包含地面、视觉设置和碰撞可视化点的模型"""
    
    @staticmethod
    def construct_model(robot_type: str, collision_pairs=None, add_ground=True, add_visual=True):
        """动态构建包含地面和视觉设置的模型"""
        if collision_pairs is None:
            collision_pairs = []
            
        xml_file = str(ROBOT_XML_DICT[robot_type])
        root = mjcf.RootElement()
        elem = ET.parse(xml_file).getroot()

        # 检查并添加视觉设置
        if add_visual:
            visual_elem = elem.find("visual")
            if visual_elem is None:
                # 添加完整的视觉设置
                root.visual.headlight.diffuse = ".6 .6 .6"
                root.visual.headlight.ambient = ".1 .1 .1"
                root.visual.headlight.specular = ".9 .9 .9"
                root.visual.rgba.haze = "0.15 0.25 0.35 1"
                root.visual.quality.shadowsize = "8192"

                getattr(root.visual, "global").azimuth = "-140"
                getattr(root.visual, "global").elevation = "-20"
                getattr(root.visual, "global").offheight = "2080"
                getattr(root.visual, "global").offwidth = "1170"

            # 检查并添加资产设置（天空盒等）
            asset_elem = elem.find("asset")
            has_skybox = False
            if asset_elem is not None:
                for texture in asset_elem.findall("texture"):
                    if texture.get("type") == "skybox":
                        has_skybox = True
                        break
                        
            if not has_skybox:
                # 添加天空盒纹理
                root.asset.add(
                    "texture", type="skybox", builtin="gradient", rgb1="1 1 1", 
                    rgb2="1 1 1", width="800", height="800"
                )

        # 检查并添加地面
        if add_ground:
            has_floor = False
            for geom in elem.findall(".//geom"):
                if geom.get("name") == "floor" or geom.get("type") == "plane":
                    has_floor = True
                    break

            if not has_floor:
                root.asset.add(
                    "texture", name="groundplane", type="2d", builtin="checker",  rgb1="1 1 1", 
                    rgb2="1 1 1", markrgb="0 0 0", width="300", height="300", mark="edge",
                )
                root.asset.add(
                    "material", name="groundplane", texture="groundplane", texuniform="true", 
                    texrepeat="5 5", reflectance="0",
                )
                root.worldbody.add(
                    "geom", name="floor", type="plane", size="0 0 .01", material="groundplane", 
                    contype="1", conaffinity="1", priority="1", friction="0.6", condim="3"
                )
        
        # 为碰撞对设置可视点
        if collision_pairs:
            keypoints = set()
            for collision_pair in collision_pairs:
                a,b = collision_pair
                for geom_name in a:
                    keypoints.add(geom_name)
                for geom_name in b:
                    keypoints.add(geom_name)
            
            # 排除"floor"
            if "floor" in keypoints:
                keypoints.remove("floor")

            for keypoint in keypoints:
                body = root.worldbody.add(
                    "body", name=f"keypoint_{keypoint}", mocap="true"
                )
                rgb = np.random.rand(3)
                body.add(
                    "site",
                    name=f"site_{keypoint}",
                    type="sphere",
                    size="0.02",
                    rgba=f"{rgb[0]} {rgb[1]} {rgb[2]} 1",
                )
        
        # 包含原始机器人模型
        humanoid_mjcf = mjcf.from_path(xml_file)
        root.include_copy(humanoid_mjcf)
        
        root_str = ModelConstructor.to_string(root, pretty=True)
        assets = ModelConstructor.get_assets(root)
        return mj.MjModel.from_xml_string(root_str, assets)

    @staticmethod
    def to_string(
        root: mjcf.RootElement,
        precision: float = 17,
        zero_threshold: float = 0.0,
        pretty: bool = False,
    ) -> str:
        """将mjcf元素转换为XML字符串"""
        xml_string = root.to_xml_string(precision=precision, zero_threshold=zero_threshold)
        root_elem = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))

        # Remove hashes from asset filenames.
        tags = ["mesh", "texture"]
        for tag in tags:
            assets = [
                asset
                for asset in root_elem.find("asset").iter()
                if asset.tag == tag and "file" in asset.attrib
            ]
            for asset in assets:
                name, extension = asset.get("file").split(".")
                asset.set("file", ".".join((name[:-41], extension)))

        if not pretty:
            return etree.tostring(root_elem, pretty_print=True).decode()

        # Remove auto-generated names.
        for elem in root_elem.iter():
            for key in elem.keys():
                if key == "name" and "unnamed" in elem.get(key):
                    elem.attrib.pop(key)

        # Get string from lxml.
        xml_string = etree.tostring(root_elem, pretty_print=True)

        # Remove redundant attributes.
        xml_string = xml_string.replace(b' gravcomp="0"', b"")

        # Insert spaces between top-level elements.
        lines = xml_string.splitlines()
        newlines = []
        for line in lines:
            newlines.append(line)
            if line.startswith(b"  <"):
                if line.startswith(b"  </") or line.endswith(b"/>"):
                    newlines.append(b"")
        newlines.append(b"")
        xml_string = b"\n".join(newlines)

        return xml_string.decode()

    @staticmethod
    def get_assets(root: mjcf.RootElement):
        """获取mjcf元素中的资产"""
        assets = {}
        for file, payload in root.get_assets().items():
            name, extension = file.split(".")
            assets[".".join((name[:-41], extension))] = payload
        return assets