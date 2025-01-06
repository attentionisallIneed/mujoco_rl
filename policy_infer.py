import math
import os
import torch
import json

import numpy as np

from collections import deque

from utils import quaternion_to_euler_array, pd_control, class_to_dict
from configs import env_cfg, cfg
# from carry_configs import env_cfg, cfg
from actor_critic import ActorCriticDH


class RealtimeController:
    def __init__(self):
        self.env_cfg = env_cfg
        self.cfg = cfg

        self.policy = self.load_policy()

        self.hist_obs = deque()
        for _ in range(self.env_cfg.env.frame_stack):
            self.hist_obs.append(np.zeros([1, self.env_cfg.env.num_single_obs], dtype=np.double))
        self.count = 0
        self.action = np.zeros(self.env_cfg.env.num_actions, dtype=np.double)
        self.target_q = np.zeros(self.env_cfg.env.num_actions, dtype=np.double)
        self.target_dq = np.zeros(self.env_cfg.env.num_actions, dtype=np.double)
        self.tau = np.zeros(self.env_cfg.env.num_actions, dtype=np.double)
        self.cmd = [0.5, 0.0, 0.0]

        self.record = []

        action_dt = self.cfg.sim_config.dt * self.cfg.sim_config.decimation
        env_dt = self.cfg.sim_config.dt
        print(f'action_dt: {action_dt}, env_dt: {env_dt}')

        if self.cfg.sim_config.dt == 0.005:
            self.cfg.robot_config.kps *= 0.8
            self.cfg.robot_config.kds *= 0.8
        print(f'kps: {self.cfg.robot_config.kps}\nkds: {self.cfg.robot_config.kds}')

    def load_policy(self):
        root_path = self.cfg.sim_config.policy_path
        model_path = os.listdir(root_path)
        model_path.sort()
        model_path = os.path.join(root_path, model_path[2])

        if model_path.endswith('.pt'):
            policy_cfg = class_to_dict(self.env_cfg.policy)
            policy = ActorCriticDH(
                self.env_cfg.env.num_short_obs,
                self.env_cfg.env.num_single_obs,
                self.env_cfg.env.num_privileged_obs,
                self.env_cfg.env.num_actions,
                **policy_cfg
            )
            loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))
            policy.load_state_dict(loaded_dict["model_state_dict"])
        else:
            jit_name = os.listdir(model_path)
            model_path = os.path.join(model_path, jit_name[-1])
            policy = torch.jit.load(model_path)
        print("Load model from:", model_path)
        return policy

    def update_cmd(self, x_vel_cmd, yaw_vel_cmd):
        x_vel_cmd = np.clip(x_vel_cmd, 0.0, 0.8)
        yaw_vel_cmd = np.clip(yaw_vel_cmd, -0.6, 0.6)
        self.cmd = [x_vel_cmd, 0.0, yaw_vel_cmd]

    def step(self,  q, dq, quat, omega):
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = self.cmd

        # Obtain an observation
        q = q[-self.env_cfg.env.num_actions:]
        dq = dq[-self.env_cfg.env.num_actions:]

        if self.count % self.cfg.sim_config.decimation == 0:
            obs = np.zeros([1, self.env_cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            # obs
            obs[0, 0] = math.sin(
                2 * math.pi * self.count * self.cfg.sim_config.dt / self.env_cfg.rewards.cycle_time)
            obs[0, 1] = math.cos(
                2 * math.pi * self.count * self.cfg.sim_config.dt / self.env_cfg.rewards.cycle_time)
            obs[0, 2] = x_vel_cmd * self.env_cfg.normalization.obs_scales.lin_vel
            obs[0, 3] = y_vel_cmd * self.env_cfg.normalization.obs_scales.lin_vel
            obs[0, 4] = yaw_vel_cmd * self.env_cfg.normalization.obs_scales.ang_vel
            # obs[0, 4] = heading
            obs[0, self.env_cfg.env.num_commands:self.env_cfg.env.num_commands + self.env_cfg.env.num_actions] = \
                (q - self.cfg.robot_config.default_dof_pos) * self.env_cfg.normalization.obs_scales.dof_pos
            obs[0,
            self.env_cfg.env.num_commands + self.env_cfg.env.num_actions:self.env_cfg.env.num_commands + 2 * self.env_cfg.env.num_actions] = \
                dq * self.env_cfg.normalization.obs_scales.dof_vel
            obs[0,
            self.env_cfg.env.num_commands + 2 * self.env_cfg.env.num_actions:self.env_cfg.env.num_commands + 3 * self.env_cfg.env.num_actions] = self.action
            obs[0,
            self.env_cfg.env.num_commands + 3 * self.env_cfg.env.num_actions:self.env_cfg.env.num_commands + 3 * self.env_cfg.env.num_actions + 3] = omega
            obs[0,
            self.env_cfg.env.num_commands + 3 * self.env_cfg.env.num_actions + 3:self.env_cfg.env.num_commands + 3 * self.env_cfg.env.num_actions + 6] = eu_ang

            ####### for stand only #######
            if self.env_cfg.env.add_stand_bool:
                vel_norm = np.sqrt(x_vel_cmd ** 2 + y_vel_cmd ** 2 + yaw_vel_cmd ** 2)
                stand_command = (vel_norm <= self.env_cfg.commands.stand_com_threshold)
                obs[0, -1] = stand_command

            obs = np.clip(obs, -self.env_cfg.normalization.clip_observations,
                          self.env_cfg.normalization.clip_observations)

            self.hist_obs.append(obs)
            self.hist_obs.popleft()

            policy_input = np.zeros([1, self.env_cfg.env.num_observations], dtype=np.float32)
            for j in range(self.env_cfg.env.frame_stack):
                policy_input[0, j * self.env_cfg.env.num_single_obs: (j + 1) * self.env_cfg.env.num_single_obs] = \
                self.hist_obs[j][0, :]

            if hasattr(self.policy, 'act_inference'):
                self.action[:] = self.policy.act_inference(torch.tensor(policy_input))[0].detach().numpy()
            else:
                self.action[:] = self.policy(torch.tensor(policy_input))[0].detach().numpy()
            self.action = np.clip(self.action, -self.env_cfg.normalization.clip_actions,
                                  self.env_cfg.normalization.clip_actions)
            self.target_q = self.action * self.env_cfg.control.action_scale

        # Generate PD control
        self.tau = pd_control(self.target_q, q, self.cfg.robot_config.kps,
                         self.target_dq, dq, self.cfg.robot_config.kds, self.cfg)  # Calc torques
        self.tau = np.clip(self.tau, -self.cfg.robot_config.tau_limit, self.cfg.robot_config.tau_limit)  # Clamp torques

        self.count += 1
        self.log()

    def log(self):
        if self.count % self.cfg.sim_config.decimation == 0:
            info = {
                'step': self.count,
                'action': self.action,
                'tau': self.tau,
            }
            self.record.append(info)

    def write_log(self):
        with open('logs/log.json', 'w') as f:
            json.dump(self.record, f)


def get_observation():
    """
    :return q: joint angles
    :return dq: joint velocities
    :return quat: orientation quaternion
    :return omega: angular velocity
    joints:
    left_hip_pitch_joint
    left_hip_roll_joint
    left_hip_yaw_joint
    left_knee_pitch_joint
    left_ankle_pitch_joint
    left_ankle_roll_joint
    right_hip_pitch_joint
    right_hip_roll_joint
    right_hip_yaw_joint
    right_knee_pitch_joint
    right_ankle_pitch_joint
    right_ankle_roll_joint
    """
    q = np.array([-7.25877873e-22, -1.83623480e-22,  7.99990190e-01,  1.00000000e+00,
        9.54274968e-22,  0.00000000e+00,  0.00000000e+00,  4.00000000e-01,
        5.00000000e-02, -3.10000000e-01,  4.90000000e-01, -2.10000000e-01,
       -4.83474122e-21, -4.00000000e-01, -5.00000000e-02,  3.10000000e-01,
        4.90000000e-01, -2.10000000e-01, -1.30657935e-19])
    dq = np.array([-7.25877873e-19, -1.83623480e-19, -9.81000000e-03, -5.32021531e-19,
        1.76877645e-18,  4.80568241e-19,  1.81946478e-17,  6.86430917e-18,
       -9.88916573e-18,  2.19597862e-17, -6.02225298e-18, -4.83474122e-18,
       -1.77004915e-17, -1.44339033e-17,  2.19047293e-17,  2.72449021e-17,
       -6.40993121e-18, -1.30657935e-16])
    quat = np.array([0., 0., 0., 1.])
    omega = np.array([0., 0., 0.])
    q = q[-env_cfg.env.num_actions:]
    dq = dq[-env_cfg.env.num_actions:]
    return q, dq, quat, omega


class Env:
    def step(self, action):
        pass


if __name__ == "__main__":
    # init
    controller = RealtimeController()
    env = Env()
    while True:
        # 线速度、角速度
        controller.update_cmd(1, 0)
        # 获取observation
        q, dq, quat, omega = get_observation()
        # 获取action
        controller.step(q, dq, quat, omega)
        # 位置
        action = controller.action
        # 缩放后的位置
        target_q = controller.target_q
        # 力矩
        tau = controller.tau
        # 执行action
        env.step(tau)
