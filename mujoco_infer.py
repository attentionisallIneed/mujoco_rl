import math
import numpy as np
import mujoco, mujoco_viewer

from policy_infer import RealtimeController
from utils import quaternion_to_euler_array


class MujocoController(RealtimeController):
    def __init__(self):
        super().__init__()
        self.data, self.model = self.init_env()
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def init_env(self):
        print("Load mujoco xml from:", self.cfg.sim_config.mujoco_model_path)
        # load model xml
        model = mujoco.MjModel.from_xml_path(self.cfg.sim_config.mujoco_model_path)
        # simulation timestep
        model.opt.timestep = self.cfg.sim_config.dt
        # model data
        data = mujoco.MjData(model)
        num_actuated_joints = self.env_cfg.env.num_actions  # This should match the number of actuated joints in your model
        data.qpos[-num_actuated_joints:] = self.cfg.robot_config.default_dof_pos
        mujoco.mj_step(model, data)
        return data, model

    def get_obs(self):
        q = self.data.qpos.astype(np.double)  # 关节位置 策略需要
        dq = self.data.qvel.astype(np.double)  # 关节速度 策略需要
        quat = self.data.sensor('body-orientation').data[[1, 2, 3, 0]].astype(np.double)  # 本体姿态 策略需要
        omega = self.data.sensor('body-angular-velocity').data.astype(np.double)  # 角速度 策略需要
        base_pos = [0, 0]
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name == 'x1-body':  # according to model name
                base_pos = self.data.xpos[i][:3].copy().astype(np.double)
        return q, dq, quat, omega, base_pos

    def mujoco_step(self):
        # self.tau = np.delete(self.tau, 1)
        self.data.ctrl = self.tau
        mujoco.mj_step(self.model, self.data)
    
    def control(self, lin_vel, ang_vel, count):
        self.cmd = (lin_vel, 0.0, ang_vel)
        self.count = count
        q, dq, quat, omega, base_pos = self.get_obs()
        self.step(q, dq, quat, omega)
        self.mujoco_step()

    def walk(self, dist):
        print('walking')
        self.cmd = (0.8, 0.0, 0.0)
        self.count = 0
        q, dq, quat, omega, base_pos = self.get_obs()
        x0, y0 = base_pos[0], base_pos[1]
        x1, y1 = base_pos[0], base_pos[1]

        while np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) < dist:
            self.step(q, dq, quat, omega)
            self.mujoco_step()
            q, dq, quat, omega, base_pos = self.get_obs()
            x1, y1 = base_pos[0], base_pos[1]
            if self.count % self.env_cfg.env.render_step == 0:
                self.viewer.render()

    def rotate(self, angle, walk=False):
        print(f'rotating and walking is {walk}')
        if walk:
            lin_vel_x = 0.8
        else:
            lin_vel_x = 0.0
        if angle > 0:
            ang_vel_yaw = 0.6
        else:
            ang_vel_yaw = -0.6

        self.cmd = (lin_vel_x, 0.0, ang_vel_yaw)
        self.count = 0
        q, dq, quat, omega, base_pos = self.get_obs()
        eu_ang = quaternion_to_euler_array(quat)
        # 将角度值“归一化”到−𝜋到𝜋之间
        eu_ang[2] = (eu_ang[2] + math.pi) % (2 * math.pi) - math.pi
        angle0 = eu_ang[2]
        angle1 = eu_ang[2]

        while np.abs((angle1 - angle0 + math.pi) % (2 * math.pi) - math.pi) <= np.abs(angle):
            self.step(q, dq, quat, omega)
            self.mujoco_step()
            q, dq, quat, omega, base_pos = self.get_obs()
            eu_ang = quaternion_to_euler_array(quat)
            angle1 = (eu_ang[2] + math.pi) % (2 * math.pi) - math.pi
            if self.count % self.env_cfg.env.render_step == 0:
                self.viewer.render()
        print('rotating and walking is done')


if __name__ == '__main__':
    controller = MujocoController()
    controller.walk(1)
    controller.rotate(math.pi / 2, walk=False)
    controller.walk(1)
    controller.rotate(math.pi / 2, walk=False)
    controller.walk(1)
    controller.rotate(math.pi / 2, walk=False)
    controller.walk(1)
    # controller.rotate(math.pi / 2, walk=True)
    # controller.rotate(math.pi / 2, walk=True)
