# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class train_cfg:

    """
    after 1500 iter with first_train True, 
    then set first_train False and train 1000 iter
    """
    first_train = False 
                # True False
                           
    # paras for randomization ----------  
    turn_on_rand = True
    action_rand = 1.0
    vel_track_amp = 2.0

    if first_train:
        turn_on_rand = False
        action_rand = 0.0
        vel_track_amp = 1.0
    # -----------------------------------

    """
    after trained first_train with False, 
    then set turn_on_curriculum to True and train 4000 
    """
    turn_on_curriculum = False
                       # True False
    # paras for curriculum learning -----
    terrain_type = 'plane'

    if turn_on_curriculum:
        terrain_type = 'trimesh'
    # -----------------------------------

    """
    before final implementation, 
    set True for smoothness and train 
    """
    Final_train_on = True
                    # True False
    # paras for final training ---------
    smoothness_amp = 1.0
    energy_amp = 1.0
    vel_track_amp_final = 1.0

    if Final_train_on:
        smoothness_amp = 100.0
        energy_amp = 10.0
        vel_track_amp_final = 5.0
    # -----------------------------------


class HhfcCfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        episode_length_s = 24     # episode length in seconds
        use_ref_actions = False   # speed up training by using reference actions

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/hhfc_sf/urdf/hhfc_rel_vel_fake_lim.urdf'

        name = "hhfc_sf"
        foot_name = "ankle_r"
        knee_name = "knee"

        terminate_after_contacts_on = ['base_thorax']
        penalize_contacts_on = ["base_thorax"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = train_cfg.terrain_type
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 15  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.876]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'Lleg_hip_p_joint': 0.3,
            'Lleg_hip_r_joint': 0.,
            'Lleg_hip_y_joint': 0.,
            'Lleg_knee_joint': -0.6,
            'Lleg_ankle_p_joint': -0.3,
            'Lleg_ankle_r_joint': 0.,
            'Rleg_hip_p_joint': 0.3,
            'Rleg_hip_r_joint': 0.,
            'Rleg_hip_y_joint': 0.,
            'Rleg_knee_joint': -0.6,
            'Rleg_ankle_p_joint': -0.3,
            'Rleg_ankle_r_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            'Lleg_hip_p_joint': 150.,
            'Lleg_hip_r_joint': 150.,
            'Lleg_hip_y_joint': 100.,
            'Lleg_knee_joint': 150.,
            'Lleg_ankle_p_joint': 30.,
            'Lleg_ankle_r_joint': 30.,
            'Rleg_hip_p_joint': 150.,
            'Rleg_hip_r_joint': 150.,
            'Rleg_hip_y_joint': 100.,
            'Rleg_knee_joint': 150.,
            'Rleg_ankle_p_joint': 30.,
            'Rleg_ankle_r_joint': 30.,
        }
        damping = {
            'Lleg_hip_p_joint': 1.5,
            'Lleg_hip_r_joint': 1.5,
            'Lleg_hip_y_joint': 1.0,
            'Lleg_knee_joint': 1.5,
            'Lleg_ankle_p_joint': 1.0,
            'Lleg_ankle_r_joint': 1.0,
            'Rleg_hip_p_joint': 1.5,
            'Rleg_hip_r_joint': 1.5,
            'Rleg_hip_y_joint': 1.0,
            'Rleg_knee_joint': 1.5,
            'Rleg_ankle_p_joint': 1.0,
            'Rleg_ankle_r_joint': 1.0,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5 # experience
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        # terrain randomization
        randomize_friction = train_cfg.turn_on_rand
        friction_range = [0.1, 2.0]
        randomize_base_mass = train_cfg.turn_on_rand
        added_mass_range = [-5., 15.]
        push_robots = train_cfg.turn_on_rand # True
        push_interval_s = 4
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        # dynamic randomization
        action_delay = train_cfg.action_rand * 0.5
        action_noise = train_cfg.action_rand * 0.02
        # pd gain randomization
        stiffness_noise = train_cfg.action_rand * 0.15
        damping_noise = train_cfg.action_rand * 0.1

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.3, 0.6]   # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3] # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.867
        min_dist = 0.25
        max_dist = 1.0
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.2    # rad
        target_feet_height = 0.06        # m
        cycle_time = 0.8                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 800  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 3.2
            feet_clearance = 0.5 #0.5 # to rarget feet height
            feet_contact_number = 1.8
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2 * train_cfg.vel_track_amp * train_cfg.vel_track_amp_final
            tracking_ang_vel = 1.1 * train_cfg.vel_track_amp
            vel_mismatch_exp = 0.5 * train_cfg.vel_track_amp * train_cfg.vel_track_amp_final  # lin_z; ang x,y
            low_speed = 0.2 * train_cfg.vel_track_amp
            track_vel_hard = 0.5 * train_cfg.vel_track_amp * train_cfg.vel_track_amp_final
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.5 #0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002 * train_cfg.smoothness_amp
            torques = -1e-5 * train_cfg.energy_amp
            dof_vel = -5e-4 * train_cfg.energy_amp
            dof_acc = -1e-7 * train_cfg.energy_amp
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.


class HhfcCfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1501  # number of policy updates

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'HHFC_ppo'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
