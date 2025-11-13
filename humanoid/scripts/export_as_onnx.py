from numpy.lib import NumpyVersion
from humanoid import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from humanoid.envs import *
from humanoid.utils import (
    get_args,
    export_policy_as_jit,
    task_registry,
    Logger,
    get_load_path,
    class_to_dict,
    terrain,
)

from humanoid.algo.ppo.actor_critic import ActorCritic


import numpy as np
import torch
import copy


def export_policy_as_onnx(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    resume_path = get_load_path(
        log_root,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
    )
    loaded_dict = torch.load(resume_path)
    print("Loaded policy: ", resume_path)

    actor_critic_class = eval(train_cfg.runner.policy_class_name)
    if env_cfg.env.num_privileged_obs is None:
        env_cfg.env.num_privileged_obs = env_cfg.env.num_observations
    actor_critic = actor_critic_class(
        env_cfg.env.num_observations,
        env_cfg.env.num_privileged_obs,
        env_cfg.env.num_actions,
        **class_to_dict(train_cfg.policy),
    ).to(args.rl_device)
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    # export policy as an onnx file
    path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
        "policies",
    )
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.onnx")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()

    # dummy_input = torch.randn([1, env_cfg.env.num_observations])
    dummy_input = torch.randn((1, env_cfg.env.num_observations), dtype=torch.float32)
    input_names = ["nn_input"]
    output_names = ["nn_output"]

    torch.onnx.export(
        model,
        dummy_input,
        path,
        # verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        # opset_version=13,
    )
    print("Exported policy as onnx script to: ", path)

    torch_input = torch.zeros(705)
    print(model(torch_input))
    test_onnx(path)


def test_onnx(path, input_size=705):
    print("onnxruntime test")
    import numpy as np
    import onnxruntime as ort

    ## ONNX Runtime test
    # Create ONNX Runtime session
    providers = ["CPUExecutionProvider"]  # Can also use 'CUDAExecutionProvider' for GPU
    session = ort.InferenceSession(path, providers=providers)

    # 2. Get input/output information
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # print(f"Input name: {input_name}, Shape: {input_shape}")
    # print(f"Output name: {output_name}")

    # 3. Prepare input data
    # Example: create a random input with proper shape and dtype
    input_data = np.zeros((1, input_size)).astype(np.float32)

    # 4. Run inference
    outputs = session.run([output_name], {input_name: input_data})

    # 5. Process outputs
    output = outputs[0]
    print("Inference results:")
    print(output)

    ## OpenVINO test
    print("openvino test")
    import openvino as ov

    core = ov.Core()
    model = core.read_model(path)
    compiled_model = core.compile_model(model, "CPU")
    infer_request = compiled_model.create_infer_request()
    input_tensor = ov.Tensor(input_data)
    infer_request.set_input_tensor(input_tensor)
    infer_request.infer()
    output_tensor = infer_request.get_output_tensor()
    print("Inference results:")
    print(output_tensor.data)


def export_dh_as_onnx(args):

    class PolicyDH(torch.nn.Module):

        def __init__(
            self,
            actor: torch.nn.Module,
            long_history: torch.nn.Module,
            state_estimator: torch.nn.Module,
            num_short_obs,
            in_channels,
            num_proprio_obs,
        ):
            super().__init__()
            self.actor = copy.deepcopy(actor).cpu()
            self.long_history = copy.deepcopy(long_history).cpu()
            self.state_estimator = copy.deepcopy(state_estimator).cpu()
            self.num_short_obs = num_short_obs
            self.in_channels = in_channels
            self.num_proprio_obs = num_proprio_obs

        def forward(self, obs):
            # obs = [t-H, ..., t-1, t]
            short_history = obs[..., -self.num_short_obs :]
            estimated_vel = self.state_estimator(short_history)
            latent_vector = self.long_history(
                obs.view(-1, self.in_channels, self.num_proprio_obs)
            )
            print(short_history.shape)
            print(estimated_vel.shape)
            print(latent_vector.shape)

            actor_obs = torch.cat((short_history, estimated_vel, latent_vector), dim=-1)
            action_mean = self.actor(actor_obs)
            return action_mean

    # Get state dictionary
    env_cfg: DHCfg
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    resume_path = get_load_path(
        log_root,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
    )
    loaded_dict = torch.load(resume_path)
    print("Loaded policy: ", resume_path)

    # Create actor critic class
    actor_critic_class = eval(train_cfg.runner.policy_class_name)
    if env_cfg.env.num_privileged_obs is None:
        env_cfg.env.num_privileged_obs = env_cfg.env.num_observations
    num_short_obs = env_cfg.env.short_frame_stack * env_cfg.env.num_single_obs
    num_critic_obs = env_cfg.env.num_privileged_obs
    if env_cfg.terrain.measure_heights:
        num_critic_obs = env_cfg.env.c_frame_stack * (
            env_cfg.env.single_num_privileged_obs + env_cfg.terrain.num_height
        )
    print(num_critic_obs)
    actor_critic: ActorCritic = actor_critic_class(
        num_short_obs,
        env_cfg.env.num_single_obs,
        num_critic_obs,
        env_cfg.env.num_actions,
        **class_to_dict(train_cfg.policy),
    ).to(args.rl_device)
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])

    # Create path
    path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
        "policies",
    )
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy.onnx")

    dh_policy = PolicyDH(
        actor_critic.actor,
        actor_critic.long_history,
        actor_critic.state_estimator,
        num_short_obs,
        env_cfg.env.frame_stack,
        env_cfg.env.num_single_obs,
    )
    dh_policy.eval()

    dummy_input = torch.randn((1, env_cfg.env.num_observations), dtype=torch.float32)
    input_names = ["nn_input"]
    output_names = ["nn_output"]

    torch.onnx.export(
        dh_policy,
        dummy_input,
        path,
        # verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        # opset_version=13,
    )
    print("Exported policy as onnx script to: ", path)

    torch_input = torch.zeros((1, env_cfg.env.num_observations))
    print(dh_policy(torch_input))
    test_onnx(path, env_cfg.env.num_observations)


if __name__ == "__main__":
    args = get_args()
    export_policy_as_onnx(args)

