import time
import numpy as np
import mujoco, mujoco_viewer
import mediapy as media

from scipy.optimize import minimize

from policy_infer import RealtimeController
from utils import quaternion_to_euler_array
from configs import env_cfg, cfg

def load_model_and_data(model_path):
    """
    加载MuJoCo模型和数据
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data

def create_simulation(model, data):
    """
    创建MuJoCo模拟器和可视化器
    """
    sim = mujoco.MjSim(model, data)
    viewer = mujoco_viewer.MujocoViewer(sim, init_width=800, init_height=600)
    return sim, viewer

def get_object_properties(model, data, object_name):
    """
    获取物体的位置和尺寸
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if body_id == -1:
        raise ValueError(f"Object '{object_name}' not found in the model.")
    
    # 获取物体的位置
    pos = data.xpos[body_id].copy()
    
    # 获取物体的几何体尺寸
    # 获取物体所有的geom ID
    geom_ids = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == body_id:
            geom_ids.append(geom_id)
    
    if not geom_ids:
        raise ValueError(f"No geom found for object '{object_name}'.")
    
    # 假设第一个geom是主要的几何体
    size = model.geom_size[geom_ids[0]].copy()
    
    return pos, size

def compute_hand_targets(object_pos, object_size, hand_offset=0.05):
    """
    计算左右手的目标位置
    """
    # 假设物体在y轴上分布，左右手分别在物体的左右两侧
    left_target = object_pos.copy()
    right_target = object_pos.copy()
    
    left_target[1] -= (object_size[1]/2 + hand_offset)
    right_target[1] += (object_size[1]/2 + hand_offset)
    
    return left_target, right_target

def inverse_kinematics(model, data, target_pos, joint_names, eef_name):
    """
    简单的逆运动学实现，使用优化的方法最小化末端执行器与目标位置的距离
    """
    # 获取臂的关节索引
    joint_ids = []
    for j in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if joint_name and joint_name in joint_names:
            joint_ids.append(j)
    
    if not joint_ids:
        raise ValueError(f"No joints found for arm '{eef_name}'.")
    
    # 假设末端执行器的body名称为 "{arm_name}_link"
    eef_name = f"{eef_name}_link"
    eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, eef_name)
    if eef_id == -1:
        raise ValueError(f"End-effector '{eef_name}' not found in the model.")
    
    def objective(q):
        # 保存当前关节角度
        original_qpos = data.qpos.copy()
        
        # 设置新的关节角度
        data.qpos[joint_ids] = q
        mujoco.mj_forward(model, data)
        
        # 获取末端执行器位置
        eef_pos = data.xpos[eef_id]
        
        # 恢复原始关节角度
        data.qpos[:] = original_qpos
        
        # 计算目标与末端执行器的距离平方和
        return np.sum((eef_pos - target_pos)**2)
    
    # 初始猜测为当前关节角度
    initial_q = data.qpos[joint_ids].copy()
    
    # 优化
    res = minimize(objective, initial_q, method='BFGS')
    
    if res.success:
        return res.x
    else:
        raise ValueError(f"Inverse kinematics for {eef_name} failed.")

def smooth_joint_movement(model, data, viewer, joint_ids, target_angles, steps=100):
    """
    平滑地移动关节到目标角度
    """
    current_angles = data.qpos[joint_ids].copy()
    delta = (target_angles - current_angles) / steps
    
    for _ in range(steps):
        data.qpos[joint_ids] += delta
        mujoco.mj_forward(model, data)
        time.sleep(0.01)
        if viewer.is_alive:
            viewer.render()
        else:
            break

def main():
    # 替换为你的MuJoCo模型文件路径
    model_path = cfg.sim_config.mujoco_model_path
    
    # 替换为你模型中物体的名称
    object_name = "object"
    
    # 替换为左臂和右臂的命名（根据你的模型）
    left_arm_name = "left_arm"
    right_arm_name = "right_arm"
    
    # 加载模型和数据
    model, data = load_model_and_data(model_path)
    
    # 创建模拟器和可视化器
    sim, viewer = create_simulation(model, data)
    
    # 获取物体的位置和尺寸
    object_pos, object_size = get_object_properties(sim, object_name)
    print(f"Object Position: {object_pos}, Size: {object_size}")
    
    # 计算左右手的目标位置
    left_target, right_target = compute_hand_targets(object_pos, object_size)
    print(f"Left Hand Target: {left_target}, Right Hand Target: {right_target}")
    
    # 计算逆运动学得到目标关节角度
    left_target_angles = inverse_kinematics(sim, left_target, left_arm_name)
    right_target_angles = inverse_kinematics(sim, right_target, right_arm_name)
    print(f"Left Arm Target Angles: {left_target_angles}")
    print(f"Right Arm Target Angles: {right_target_angles}")
    
    # 获取左右臂的关节索引
    def get_joint_ids(sim, arm_name):
        joints = []
        for j in range(sim.model.njnt):
            joint_name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if joint_name and arm_name in joint_name:
                joints.append(j)
        return joints
    
    left_arm_joints = get_joint_ids(sim, left_arm_name)
    right_arm_joints = get_joint_ids(sim, right_arm_name)
    
    if not left_arm_joints or not right_arm_joints:
        raise ValueError("Failed to retrieve arm joints. Check arm names.")
    
    # 平滑移动关节到目标角度
    smooth_joint_movement(sim, left_arm_joints, left_target_angles)
    smooth_joint_movement(sim, right_arm_joints, right_target_angles)
    
    # 运行仿真以观察结果
    while viewer.is_alive:
        sim.step()
        viewer.render()
        time.sleep(0.01)

if __name__ == "__main__":
    main()
