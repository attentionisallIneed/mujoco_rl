import json
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
        self.data.ctrl = self.tau
        mujoco.mj_step(self.model, self.data)

    def walk(self, dist):
        print('walking')
        self.cmd = (0.8, 0.0, 0.0)
        self.count = 0
        q, dq, quat, omega, base_pos = self.get_obs()
        x0, y0 = base_pos[0], base_pos[1]
        x1, y1 = base_pos[0], base_pos[1]

        record = []
        action_dt = self.cfg.sim_config.dt * self.cfg.sim_config.decimation
        env_dt = self.cfg.sim_config.dt
        print(f'action_dt: {action_dt}, env_dt: {env_dt}')

        while np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) < dist:
            self.step(q, dq, quat, omega)
            self.mujoco_step()
            q, dq, quat, omega, base_pos = self.get_obs()
            x1, y1 = base_pos[0], base_pos[1]
            if self.count % self.env_cfg.env.render_step == 0:
                self.viewer.render()

            if self.count * env_dt <= 2:
                tmp = {'t': self.count * env_dt,
                       'action': self.action.tolist(),
                       'tau': self.tau.tolist(),
                       'q': q.tolist()[-12:]}
                record.append(tmp)

        with open('logs/log_0.001_10.json', 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    controller = MujocoController()
    controller.walk(2)
