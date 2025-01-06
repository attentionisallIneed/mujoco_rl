from urdfpy import URDF, Link, Joint, Inertial, Visual, Collision, Geometry, Box
import numpy as np
import xml.etree.ElementTree as ET

# 加载原始 URDF 文件
robot = URDF.load("x1.urdf")

# 确定胸膛的 link
chest_link_name = "lumber_yaw"
if chest_link_name not in robot.link_map:
    raise ValueError(f"未找到指定的 link: {chest_link_name}")
print(f"已找到胸膛 link: {chest_link_name}")

# 定义立方体属性
cube_size = 0.2  # m，立方体边长
cube_mass = 2.0  # kg，立方体质量
inertia_value = (1 / 6) * cube_mass * (cube_size ** 2)
inertia_matrix = np.diag([inertia_value] * 3)

# 创建齐次变换矩阵：视觉和碰撞的原点位姿
def create_origin(xyz, rpy):
    """创建一个4x4的齐次变换矩阵."""
    from scipy.spatial.transform import Rotation
    translation = np.eye(4)
    translation[:3, 3] = xyz
    rotation = np.eye(4)
    rotation[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    return np.dot(translation, rotation)

# 创建新的 Link 表示立方体
cube_link = Link(
    name="chest_cube",
    inertial=Inertial(
        mass=cube_mass,
        inertia=inertia_matrix,
    ),
    visuals=[
        Visual(
            name="chest_cube_visual",  # 添加视觉元素的名称
            geometry=Geometry(
                box=Box(size=(cube_size, cube_size, cube_size))
            ),
            origin=create_origin([cube_size, 0, 0], [0, 0, 0])  # 位于父链接正上方
        )
    ],
    collisions=[
        Collision(
            name="chest_cube_collision",  # 添加碰撞元素的名称
            geometry=Geometry(
                box=Box(size=(cube_size, cube_size, cube_size))
            ),
            origin=create_origin([cube_size, 0, 0], [0, 0, 0])  # 与视觉一致
        )
    ]
)

# 创建固定 Joint 将立方体连接到胸膛
cube_joint = Joint(
    name="chest_cube_joint",
    parent=chest_link_name,  # 父Link
    child="chest_cube",     # 子Link
    joint_type="fixed",
    origin=create_origin([0, 0, 0.1], [0, 0, 0]),  # 位于胸膛正前方
)

# 创建新的 URDF 对象，包含原始内容和新内容
new_robot = URDF(
    name=robot.name,
    links=robot.links + [cube_link],  # 添加新 Link
    joints=robot.joints + [cube_joint]  # 添加新 Joint
)

# 保存临时 URDF 文件
temp_file = "temp_robot.urdf"
new_robot.save(temp_file)

# 加载 XML 文件
tree = ET.parse(temp_file)
root = tree.getroot()

# 添加 <mujoco> 标签到 <robot> 标签后
mujoco = ET.Element('mujoco')
compiler = ET.SubElement(mujoco, 'compiler', {
    'meshdir': '../meshes/',
    'balanceinertia': 'true',
    'discardvisual': 'false',
    'fusestatic': 'false'
})
root.insert(0, mujoco)  # 将 <mujoco> 插入到 <robot> 的后面

# 保存最终的 URDF 文件
final_file = "robot_with_cube.urdf"
tree.write(final_file, encoding="utf-8", xml_declaration=True)

print(f"Modified URDF saved to {final_file}")
