from __future__ import annotations

from omni.isaac.lab.utils import configclass
from bhr_template.mrl.cfg import MrlPpoRunnerCfg, MrlPpoAlgorithmCfg
from .rsl_rl_ppo_cfg import RslRlPpoActorCriticCfg

@configclass
class BHR8_FC2ExpMirrorPPORunnerCfg(MrlPpoRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "bhr8_fc2_exp_mirror"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims =[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = MrlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )

    def __post_init__(self):
        super().__post_init__()
        
        self.algorithm.mirror_mean_loss_coef = 5.0
        self.algorithm.mirror_std_loss_coef = 1.0

        lhipYaw         = 0 
        lshoulderPitch  = 1
        rhipYaw         = 2
        rshoulderPitch  = 3
        lhipRoll        = 4
        lshoulderRoll   = 5
        rhipRoll        = 6
        rshoulderRoll   = 7
        lhipPitch       = 8
        lshoulderYaw    = 9
        rhipPitch       = 10
        rshoulderYaw    = 11
        lknee           = 12
        lelbow          = 13
        rknee           = 14
        relbow          = 15
        lankle1         = 16
        rankle1         = 17
        lankle2         = 18
        rankle2         = 19
        JOINT_NUM       = 20

        left_joint_ids  = [
            lhipYaw,
            lshoulderPitch,
            lhipRoll,
            lshoulderRoll,
            lhipPitch,
            lshoulderYaw,
            lknee,
            lelbow,
            lankle1,
            lankle2,
        ]
        right_joint_ids = [
            rhipYaw,
            rshoulderPitch,
            rhipRoll,
            rshoulderRoll,
            rhipPitch,
            rshoulderYaw,
            rknee,
            relbow,
            rankle1,
            rankle2,
        ]

        self.algorithm.action_mirror_id_left  = left_joint_ids
        self.algorithm.action_mirror_id_right = right_joint_ids
        self.algorithm.action_opposite_id = [
            lhipRoll,
            lhipYaw,
            lankle2,
            lshoulderRoll,
            lshoulderYaw,
            rhipRoll,
            rhipYaw,
            rankle2,
            rshoulderRoll,
            rshoulderYaw,
        ]

        BASE_ANG_VEL      = 0
        PROJECTED_GRAVITY = BASE_ANG_VEL + 3
        VELOCITY_COMMANDS = PROJECTED_GRAVITY + 3
        JOINT_POS         = VELOCITY_COMMANDS + 3
        JOINT_VEL         = JOINT_POS + JOINT_NUM
        ACTIONS           = JOINT_VEL + JOINT_NUM

        self.algorithm.policy_obvs_mirror_id_left = [JOINT_POS + joint_id for joint_id in left_joint_ids]
        self.algorithm.policy_obvs_mirror_id_left.extend([JOINT_VEL + joint_id for joint_id in left_joint_ids])
        self.algorithm.policy_obvs_mirror_id_left.extend([ACTIONS + joint_id for joint_id in left_joint_ids])
        self.algorithm.policy_obvs_mirror_id_right = [JOINT_POS + joint_id for joint_id in right_joint_ids]
        self.algorithm.policy_obvs_mirror_id_right.extend([JOINT_VEL + joint_id for joint_id in right_joint_ids])
        self.algorithm.policy_obvs_mirror_id_right.extend([ACTIONS + joint_id for joint_id in right_joint_ids])
        
        self.algorithm.policy_obvs_opposite_id = [
            BASE_ANG_VEL + 0,
            BASE_ANG_VEL + 2,
            PROJECTED_GRAVITY + 1,
            VELOCITY_COMMANDS + 1,
            VELOCITY_COMMANDS + 2,
        ]
        self.algorithm.policy_obvs_opposite_id.extend([JOINT_POS + joint_id for joint_id in self.algorithm.action_opposite_id])
        self.algorithm.policy_obvs_opposite_id.extend([JOINT_VEL + joint_id for joint_id in self.algorithm.action_opposite_id])
        self.algorithm.policy_obvs_opposite_id.extend([ACTIONS   + joint_id for joint_id in self.algorithm.action_opposite_id])

@configclass
class BHR8_FC2_NoArm_ExpMirrorPPORunnerCfg(BHR8_FC2ExpMirrorPPORunnerCfg):
    experiment_name = "bhr8_fc2_noarm_exp_mirror"

    def __post_init__(self):
        super().__post_init__()

        lhipYaw         = 0
        rhipYaw         = 1
        lhipRoll        = 2
        rhipRoll        = 3
        lhipPitch       = 4
        rhipPitch       = 5
        lknee           = 6
        rknee           = 7
        lankle1         = 8
        rankle1         = 9
        lankle2         = 10
        rankle2         = 11
        JOINT_NUM       = 12

        left_joint_ids  = [
            lhipYaw,
            lhipRoll,
            lhipPitch,
            lknee,
            lankle1,
            lankle2,
        ]
        right_joint_ids = [
            rhipYaw,
            rhipRoll,
            rhipPitch,
            rknee,
            rankle1,
            rankle2,
        ]

        self.algorithm.action_mirror_id_left  = left_joint_ids
        self.algorithm.action_mirror_id_right = right_joint_ids
        self.algorithm.action_opposite_id = [
            lhipRoll,
            lhipYaw,
            lankle2,
            rhipRoll,
            rhipYaw,
            rankle2,
        ]

        BASE_ANG_VEL      = 0
        PROJECTED_GRAVITY = BASE_ANG_VEL + 3
        VELOCITY_COMMANDS = PROJECTED_GRAVITY + 3
        JOINT_POS         = VELOCITY_COMMANDS + 3
        JOINT_VEL         = JOINT_POS + JOINT_NUM
        ACTIONS           = JOINT_VEL + JOINT_NUM

        self.algorithm.policy_obvs_mirror_id_left = [JOINT_POS + joint_id for joint_id in left_joint_ids]
        self.algorithm.policy_obvs_mirror_id_left.extend([JOINT_VEL + joint_id for joint_id in left_joint_ids])
        self.algorithm.policy_obvs_mirror_id_left.extend([ACTIONS + joint_id for joint_id in left_joint_ids])
        self.algorithm.policy_obvs_mirror_id_right = [JOINT_POS + joint_id for joint_id in right_joint_ids]
        self.algorithm.policy_obvs_mirror_id_right.extend([JOINT_VEL + joint_id for joint_id in right_joint_ids])
        self.algorithm.policy_obvs_mirror_id_right.extend([ACTIONS + joint_id for joint_id in right_joint_ids])
        
        self.algorithm.policy_obvs_opposite_id = [
            BASE_ANG_VEL + 0,
            BASE_ANG_VEL + 2,
            PROJECTED_GRAVITY + 1,
            VELOCITY_COMMANDS + 1,
            VELOCITY_COMMANDS + 2,
        ]
        self.algorithm.policy_obvs_opposite_id.extend([JOINT_POS + joint_id for joint_id in self.algorithm.action_opposite_id])
        self.algorithm.policy_obvs_opposite_id.extend([JOINT_VEL + joint_id for joint_id in self.algorithm.action_opposite_id])
        self.algorithm.policy_obvs_opposite_id.extend([ACTIONS   + joint_id for joint_id in self.algorithm.action_opposite_id])