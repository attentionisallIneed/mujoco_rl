import os
import inspect
import numpy as np

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key == "__class__":
                continue
            # get the corresponding attribute object
            var = getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)


class LeggedRobotCfg(BaseConfig):
    class env:
        short_frame_stack = 4
        num_envs = 4096
        num_observations = 235
        num_privileged_obs = None  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        num_commands = 5
        add_stand_bool = True  # used for stand

    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        # robot surrending measure points
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_base_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                  0.8]  # 1mx1.6m rectangle (without center line)
        measured_base_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_feet_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                                  0.8]  # 1mx1.6m rectangle (without center line)
        measured_feet_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        num_height = len(measured_points_x) * len(measured_points_y)
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        platform = 3.
        terrain_dict = {"flat": 0.15,
                        "rough flat": 0.15,
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "slope up": 0.,
                        "slope down": 0.,
                        "stairs up": 0.35,
                        "stairs down": 0.25,
                        "discrete": 0.0,
                        "wave": 0.0, }
        terrain_proportions = list(terrain_dict.values())

        rough_flat_range = [0.005, 0.02]  # meter
        slope_range = [0, 0.4]  # rad
        rough_slope_range = [0.005, 0.02]
        stair_width_range = [0.25, 0.25]
        stair_height_range = [0.04, 0.1]
        discrete_height_range = [0.05, 0.25]
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = True
        max_curriculum = 1.
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "joint_a": 0.,
            "joint_b": 0.}

    class control:
        control_type = 'P'  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None"  # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = False
        friction_range = [0.2, 1.3]
        restitution_range = [0.0, 0.4]

        push_robots = False
        push_interval_s = 4
        update_step = 2000 * 60
        push_duration = [0, 0.1, 0.2, 0.3]
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.2

        add_ext_force = False
        ext_force_max_xy = 10
        ext_force_max_z = 5
        ext_torque_max = 0
        ext_force_interval_s = 10
        add_update_step = 2000 * 60
        add_duration = [0, 0.1, 0.2, 0.3]

        continuous_push = False
        max_push_force = 0.5
        max_push_torque = 0.5
        push_force_noise = 0.5
        push_torque_noise = 0.5

        randomize_base_mass = False
        added_mass_range = [-2.5, 2.5]

        randomize_com = False
        com_displacement_range = [[-0.05, 0.05],
                                  [-0.05, 0.05],
                                  [-0.05, 0.05]]

        randomize_link_com = False
        link_com_displacement_range = [[-0.005, 0.005],
                                       [-0.005, 0.005],
                                       [-0.005, 0.005]]

        randomize_base_inertia = False
        base_inertial_range = [[0.98, 1.02],
                               [0.98, 1.02],
                               [0.98, 1.02]]

        randomize_link_inertia = False
        link_inertial_range = [[0.98, 1.02],
                               [0.98, 1.02],
                               [0.98, 1.02]]

        randomize_gains = False
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]  # Factor

        randomize_torque = False
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = False
        added_link_mass_range = [0.9, 1.1]

        randomize_motor_offset = False
        motor_offset_range = [-0.035, 0.035]  # Offset to add to the motor angles

        randomize_joint_friction = False
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]  # multiplier
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.3]
        joint_5_friction_range = [0.5, 1.3]
        joint_6_friction_range = [0.01, 1.15]
        joint_7_friction_range = [0.01, 1.15]
        joint_8_friction_range = [0.01, 1.15]
        joint_9_friction_range = [0.5, 1.3]
        joint_10_friction_range = [0.5, 1.3]

        randomize_joint_damping = False
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.3, 1.5]  # multiplier
        joint_1_damping_range = [0.3, 1.5]
        joint_2_damping_range = [0.3, 1.5]
        joint_3_damping_range = [0.3, 1.5]
        joint_4_damping_range = [0.9, 1.5]
        joint_5_damping_range = [0.9, 1.5]
        joint_6_damping_range = [0.3, 1.5]
        joint_7_damping_range = [0.3, 1.5]
        joint_8_damping_range = [0.3, 1.5]
        joint_9_damping_range = [0.9, 1.5]
        joint_10_damping_range = [0.9, 1.5]

        randomize_joint_armature = False
        randomize_joint_armature_each_joint = False
        joint_armature_range = [0.0001, 0.05]  # Factor
        joint_1_armature_range = [0.0001, 0.05]
        joint_2_armature_range = [0.0001, 0.05]
        joint_3_armature_range = [0.0001, 0.05]
        joint_4_armature_range = [0.0001, 0.05]
        joint_5_armature_range = [0.0001, 0.05]
        joint_6_armature_range = [0.0001, 0.05]
        joint_7_armature_range = [0.0001, 0.05]
        joint_8_armature_range = [0.0001, 0.05]
        joint_9_armature_range = [0.0001, 0.05]
        joint_10_armature_range = [0.0001, 0.05]

        add_lag = False
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [5, 70]

        add_dof_lag = False  # 这个是接收信号（dof_pos和dof_vel)的延迟,dof_pos 和dof_vel延迟一样
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False  # 不常用always False
        dof_lag_timesteps_range = [0, 40]

        add_dof_pos_vel_lag = False  # 这个是接收信号（dof_pos和dof_vel)的延迟,dof_pos 和dof_vel延迟不同
        randomize_dof_pos_lag_timesteps = True
        randomize_dof_pos_lag_timesteps_perstep = False  # 不常用always False
        dof_pos_lag_timesteps_range = [7, 25]
        randomize_dof_vel_lag_timesteps = True
        randomize_dof_vel_lag_timesteps_perstep = False  # 不常用always False
        dof_vel_lag_timesteps_range = [7, 25]

        add_imu_lag = False  # 这个是 imu 的延迟
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False  # 不常用always False
        imu_lag_timesteps_range = [1, 10]

        randomize_coulomb_friction = False
        joint_coulomb_range = [0.1, 0.9]
        joint_viscous_range = [0.10, 0.70]

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            lin_vel_z = -0.
            ang_vel_xy = -0.0
            orientation = -0.
            torques = -0.0
            dof_vel = -0.
            dof_acc = -0.0
            base_height = -0.
            feet_air_time = 0.0
            collision = -0.
            feet_stumble = -0.0
            action_rate = -0.
            stand_still = -0.

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1.  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 100.  # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class X1DHStandCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """

    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 66  # all histroy obs num
        short_frame_stack = 5  # short history step
        c_frame_stack = 3  # all histroy privileged obs num
        num_single_obs = 50
        num_observations = int(frame_stack * num_single_obs)
        num_short_obs = int(num_single_obs * short_frame_stack)
        single_num_privileged_obs = 77
        single_linvel_index = 53
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 13
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False
        num_commands = 5  # sin_pos cos_pos vx vy vz
        render_step = 50

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/urdf/x1.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/x1/mjcf/xyber_x1_flat.xml'
        policy_path = '{LEGGED_GYM_ROOT_DIR}/exported_policies'

        name = "x1"
        foot_name = "ankle_roll"
        knee_name = "knee_pitch"

        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 5  # starting curriculum state
        platform = 3.
        terrain_dict = {"flat": 0.3,
                        "rough flat": 0.2,
                        "slope up": 0.2,
                        "slope down": 0.2,
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "stairs up": 0.,
                        "stairs down": 0.,
                        "discrete": 0.1,
                        "wave": 0.0, }
        terrain_proportions = list(terrain_dict.values())

        rough_flat_range = [0.005, 0.01]  # meter
        slope_range = [0, 0.1]  # rad
        rough_slope_range = [0.005, 0.02]
        stair_width_range = [0.25, 0.25]
        stair_height_range = [0.01, 0.1]
        discrete_height_range = [0.0, 0.01]
        restitution = 0.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.02
            dof_vel = 1.5
            ang_vel = 0.2
            lin_vel = 0.1
            quat = 0.1
            gravity = 0.05
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.7]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint': 0.4,
            'left_hip_roll_joint': 0.05,
            'left_hip_yaw_joint': -0.31,
            'left_knee_pitch_joint': 0.49,
            'left_ankle_pitch_joint': -0.21,
            'left_ankle_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.4,
            'right_hip_roll_joint': -0.05,
            'right_hip_yaw_joint': 0.31,
            'right_knee_pitch_joint': 0.49,
            'right_ankle_pitch_joint': -0.21,
            'right_ankle_roll_joint': 0.0,
            'chest_cube_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {'hip_pitch_joint': 30, 'hip_roll_joint': 40, 'hip_yaw_joint': 35,
                     'knee_pitch_joint': 100, 'ankle_pitch_joint': 35, 'ankle_roll_joint': 35}
        damping = {'hip_pitch_joint': 3, 'hip_roll_joint': 3.0, 'hip_yaw_joint': 4,
                   'knee_pitch_joint': 10, 'ankle_pitch_joint': 0.5, 'ankle_roll_joint': 0.5}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 200 Hz 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.3]
        restitution_range = [0.0, 0.4]

        # push
        push_robots = True
        push_interval_s = 4  # every this second, push robot
        update_step = 2000 * 24  # after this count, increase push_duration index
        push_duration = [0, 0.05, 0.1, 0.15, 0.2, 0.25]  # increase push duration during training
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.2

        randomize_base_mass = True
        added_mass_range = [-3, 3]  # base mass rand range, base mass is all fix link sum mass

        randomize_com = True
        com_displacement_range = [[-0.05, 0.05],
                                  [-0.05, 0.05],
                                  [-0.05, 0.05]]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]  # Factor

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = True
        added_link_mass_range = [0.9, 1.1]

        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035]  # Offset to add to the motor angles

        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.3]
        joint_5_friction_range = [0.5, 1.3]
        joint_6_friction_range = [0.01, 1.15]
        joint_7_friction_range = [0.01, 1.15]
        joint_8_friction_range = [0.01, 1.15]
        joint_9_friction_range = [0.5, 1.3]
        joint_10_friction_range = [0.5, 1.3]

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.3, 1.5]
        joint_1_damping_range = [0.3, 1.5]
        joint_2_damping_range = [0.3, 1.5]
        joint_3_damping_range = [0.3, 1.5]
        joint_4_damping_range = [0.9, 1.5]
        joint_5_damping_range = [0.9, 1.5]
        joint_6_damping_range = [0.3, 1.5]
        joint_7_damping_range = [0.3, 1.5]
        joint_8_damping_range = [0.3, 1.5]
        joint_9_damping_range = [0.9, 1.5]
        joint_10_damping_range = [0.9, 1.5]

        randomize_joint_armature = True
        randomize_joint_armature_each_joint = False
        joint_armature_range = [0.0001, 0.05]  # Factor
        joint_1_armature_range = [0.0001, 0.05]
        joint_2_armature_range = [0.0001, 0.05]
        joint_3_armature_range = [0.0001, 0.05]
        joint_4_armature_range = [0.0001, 0.05]
        joint_5_armature_range = [0.0001, 0.05]
        joint_6_armature_range = [0.0001, 0.05]
        joint_7_armature_range = [0.0001, 0.05]
        joint_8_armature_range = [0.0001, 0.05]
        joint_9_armature_range = [0.0001, 0.05]
        joint_10_armature_range = [0.0001, 0.05]

        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [5, 40]

        add_dof_lag = True
        randomize_dof_lag_timesteps = True
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 40]

        add_dof_pos_vel_lag = False
        randomize_dof_pos_lag_timesteps = False
        randomize_dof_pos_lag_timesteps_perstep = False
        dof_pos_lag_timesteps_range = [7, 25]
        randomize_dof_vel_lag_timesteps = False
        randomize_dof_vel_lag_timesteps_perstep = False
        dof_vel_lag_timesteps_range = [7, 25]

        add_imu_lag = False
        randomize_imu_lag_timesteps = True
        randomize_imu_lag_timesteps_perstep = False
        imu_lag_timesteps_range = [1, 10]

        randomize_coulomb_friction = True
        joint_coulomb_range = [0.1, 0.9]
        joint_viscous_range = [0.05, 0.1]

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 25.  # time before command are changed[s]
        gait = ["walk_omnidirectional", "stand", "walk_omnidirectional"]  # gait type during training
        # proportion during whole life time
        gait_time_range = {"walk_sagittal": [2, 6],
                           "walk_lateral": [2, 6],
                           "rotate": [2, 3],
                           "stand": [2, 3],
                           "walk_omnidirectional": [4, 6]}

        heading_command = False  # if true: compute ang vel command from heading error
        stand_com_threshold = 0.05  # if (lin_vel_x, lin_vel_y, ang_vel_yaw).norm < this, robot should stand
        sw_switch = True  # use stand_com_threshold or not

        class ranges:
            lin_vel_x = [-0.4, 1.2]  # min max [m/s]
            lin_vel_y = [-0.4, 0.4]  # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        soft_dof_pos_limit = 0.98
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.61
        foot_min_dist = 0.2
        foot_max_dist = 1.0

        # final_swing_joint_pos = final_swing_joint_delta_pos + default_pos
        final_swing_joint_delta_pos = [0.25, 0.05, -0.11, 0.35, -0.16, 0.0, -0.25, -0.05, 0.11, 0.35, -0.16, 0.0]
        target_feet_height = 0.03
        target_feet_height_max = 0.06
        feet_to_ankle_distance = 0.041
        cycle_time = 0.7
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(-error*sigma)
        tracking_sigma = 5
        max_contact_force = 700  # forces above this value are penalized

        class scales:
            ref_joint_pos = 2.2
            feet_clearance = 1.
            feet_contact_number = 2.0
            # gait
            feet_air_time = 1.2
            foot_slip = -0.1
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.8
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 1.0
            orientation = 1.
            feet_rotation = 0.3
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -8e-9
            dof_vel = -2e-8
            dof_acc = -1e-7
            collision = -1.
            stand_still = 2.5
            # limits
            dof_vel_limits = -1
            dof_pos_limits = -10.
            dof_torque_limits = -0.1

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100.

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        state_estimator_hidden_dims = [256, 128, 64]

        # for long_history cnn only
        kernel_size = [6, 4]
        filter_size = [32, 16]
        stride_size = [3, 2]
        lh_output_dim = 64  # long history output dim
        in_channels = 66


env_cfg = X1DHStandCfg()
env_cfg.seed = 42


class Sim2simCfg:

    class sim_config:
        mujoco_model_path = env_cfg.asset.xml_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        policy_path = env_cfg.asset.policy_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        sim_duration = 100.0
        dt = 0.001
        decimation = 10

    class robot_config:
        # get PD gain
        kps = np.array([env_cfg.control.stiffness[joint] for joint in env_cfg.control.stiffness.keys()] * 2 + [35],
                       dtype=np.double)
        kds = np.array([env_cfg.control.damping[joint] for joint in env_cfg.control.damping.keys()] * 2 + [0],
                       dtype=np.double)

        tau_limit = 500. * np.ones(env_cfg.env.num_actions, dtype=np.double)  # 定义关节力矩的限制

        default_dof_pos = np.array(list(env_cfg.init_state.default_joint_angles.values()))


cfg = Sim2simCfg()
