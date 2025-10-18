import os
import numpy as np
import gymnasium as gym
import torch
import tyro
import wandb
from src.dqn.batch_trainer import BatchTrainer
from src.option_dqn.option_trainer import BatchedOptionTrainer
from lightning.fabric import Fabric
from src.dqn.agent import DQNAgent
from src.option_dqn.option_agent import OptionDQNAgent
import inspect
from dataclasses import dataclass, field
from typing import Optional, Union, Literal, Dict, Type, Tuple, Callable
import time
from wandb.integration.lightning.fabric import WandbLogger
from typing import Annotated
import ale_py  # noqa: F401


@dataclass
class FabricArgs:
    """Arguments for Lightning Fabric initialization."""

    accelerator: Optional[str] = "cuda"
    """Accelerator to use for training ('cpu', 'cuda', 'mps', etc.)"""
    strategy: Optional[str] = "auto"
    """Strategy to use for distributed training"""
    devices: Optional[int] = 1
    """Number of devices to use"""
    num_nodes: int = 1
    """Number of nodes to use"""
    precision: Union[str, int] = "32-true"
    """Precision for training ('32-true', 'bf16-mixed', etc.)"""
    plugins: Optional[list] = None
    """Plugins to use with Fabric"""
    callbacks: Optional[list] = None
    """Callbacks to use with Fabric"""
    loggers: Optional[list] = None
    """Loggers to use with Fabric"""

    @classmethod
    def create_from_fabric(cls):
        """Create FabricArgs with defaults from Fabric's __init__ if ours is set to None."""
        signature = inspect.signature(Fabric.__init__)
        fabric_defaults = {
            name: param.default if param.default != param.empty else None
            for name, param in signature.parameters.items()
            if name != "self"
        }

        # Create instance with our defaults
        instance = cls()

        # Replace None values with Fabric defaults
        for name, value in fabric_defaults.items():
            if getattr(instance, name) is None:
                setattr(instance, name, value)

        return instance


@dataclass
class WandbLoggerArgs:
    """Arguments for WandB initialization."""

    project: str = "testing"
    """The wandb's project name"""
    entity: Optional[str] = "ualberta-bowling"
    """The entity (team) of wandb's project"""
    mode: Literal["online", "offline", "disabled"] = os.environ.get("WANDB_MODE", "online")
    """Whether to use wandb or not"""
    notes: str = ""
    """Optional description of what is going on"""
    name: str = ""
    """Name of the run"""
    save_dir: str = "./wandb"
    """Directory where wandb will save local files. If None, wandb will use the default directory."""


# Registry of available experiment types
EXPERIMENT_REGISTRY: Dict[str, Tuple[Type[DQNAgent], Type[BatchTrainer]]] = {
    "dqn": (DQNAgent, BatchTrainer),
    "option_dqn": (OptionDQNAgent, BatchedOptionTrainer),
}


@dataclass
class ExperimentConfig:
    """Configuration for the experiment type."""

    type: Literal["dqn", "option_dqn", "no_target_dqn", "adjacency_matrix_option_dqn"]
    """Type of experiment to run"""

    def __post_init__(self):
        if self.type not in EXPERIMENT_REGISTRY:
            raise ValueError(
                f"Unknown experiment type: {self.type}. Available types: {list(EXPERIMENT_REGISTRY.keys())}"
            )


@dataclass
class CombinedArgs:
    """Combined arguments for batch training and Fabric."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    """Experiment configuration"""
    agent: Union[DQNAgent.ARGS, OptionDQNAgent.ARGS] = None
    """Agent-specific arguments."""
    trainer: Union[BatchTrainer.ARGS, BatchedOptionTrainer.ARGS] = None
    """Training configuration arguments"""
    fabric: FabricArgs = field(default_factory=FabricArgs.create_from_fabric)
    """Fabric initialization arguments."""
    wandb_logger: WandbLoggerArgs = field(default_factory=WandbLoggerArgs)
    """WandB initialization arguments."""

    def __post_init__(self):
        if self.agent is None or self.trainer is None:
            agent_cls = EXPERIMENT_REGISTRY[self.experiment.type][0]
            trainer_cls = EXPERIMENT_REGISTRY[self.experiment.type][1]

            # Only set if not already set
            if self.agent is None:
                self.agent = agent_cls.ARGS()
            if self.trainer is None:
                self.trainer = trainer_cls.ARGS()


def setup_wandb_logger(args: CombinedArgs) -> WandbLogger:
    """Setup WandB logger."""
    os.makedirs(name=args.wandb_logger.save_dir, exist_ok=True)
    logger = WandbLogger(
        project=args.wandb_logger.project,
        entity=args.wandb_logger.entity,
        name=args.wandb_logger.name
        or f"{args.trainer.env_id}__{args.experiment.type}__{args.trainer.seed}__{int(time.time())}",
        config=vars(args),
        save_code=True,
        notes=args.wandb_logger.notes,
        mode=args.wandb_logger.mode,
        save_dir=args.wandb_logger.save_dir,
    )
    print(f"saving to {args.wandb_logger.save_dir=}")
    logger.experiment.define_metric("*", step_metric="xaxis/env_frames")
    return logger


@dataclass
class DQNArgs(CombinedArgs):
    """DQN-specific combined arguments."""

    agent: DQNAgent.ARGS = field(default_factory=DQNAgent.ARGS)
    trainer: BatchTrainer.ARGS = field(default_factory=BatchTrainer.ARGS)
    experiment: ExperimentConfig = field(default_factory=lambda: ExperimentConfig(type="dqn"))


@dataclass
class OptionDQNArgs(CombinedArgs):
    """Option DQN-specific combined arguments."""

    agent: OptionDQNAgent.ARGS = field(default_factory=OptionDQNAgent.ARGS)
    trainer: BatchedOptionTrainer.ARGS = field(default_factory=BatchedOptionTrainer.ARGS)
    experiment: ExperimentConfig = field(default_factory=lambda: ExperimentConfig(type="option_dqn"))


def make_env(env_id, seed, idx, capture_video, run_name, frameskip: int = 5) -> Callable[[], gym.Env]:
    """Creates an Atari environment using the Nature preprocessing."""

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", frameskip=1)
            env = gym.wrappers.RecordVideo(env, f"{run_name}/videos", episode_trigger=lambda e: e == 0, fps=60)
        else:
            env = gym.make(env_id, frameskip=1)
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=frameskip)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TransformReward(env, np.sign)
        env = gym.wrappers.FrameStackObservation(env, 4)

        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def setup_environment(args: CombinedArgs, run_name: str) -> gym.vector.SyncVectorEnv:
    """Set up the environment based on the provided arguments."""
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id=args.trainer.env_id,
                seed=args.trainer.seed + i,
                idx=i,
                capture_video=False,
                run_name=run_name,
                frameskip=args.trainer.frame_skip,
            )
            for i in range(args.trainer.num_envs)
        ]
    )
    assert isinstance(envs.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)), (
        "only discrete action spaces are supported"
    )
    return envs


def main(args: CombinedArgs):
    """Main function to train the agent."""
    torch.set_float32_matmul_precision(args.trainer.matmul_precision)

    # Setup wandb logger
    logger = setup_wandb_logger(args)

    # Get valid Fabric kwargs from args
    fabric_kwargs = {
        name: getattr(args.fabric, name)
        for name in inspect.signature(Fabric.__init__).parameters.keys()
        if name != "self" and hasattr(args.fabric, name)
    }

    # Add logger to Fabric kwargs
    fabric_kwargs["loggers"] = [logger]

    # Initialize Fabric
    fabric = Fabric(**fabric_kwargs)
    fabric.launch()

    envs = setup_environment(args, logger.name)

    # Get agent and trainer classes from registry
    agent_cls, trainer_cls = EXPERIMENT_REGISTRY[args.experiment.type]

    agent = agent_cls(
        args=args.agent,
        env=envs,
        fabric=fabric,
        observation_space=envs.observation_space,
        action_space=envs.action_space,
    )

    # Initialize trainer
    try:
        trainer = trainer_cls(args=args.trainer, fabric=fabric, logger=logger, envs=envs, agent=agent)
        trainer.train(args.trainer.continue_criteria_str)
    except KeyboardInterrupt:
        print("Keyboard interrupt trainer")
        if args.trainer.progress_bar:
            trainer.pbar.close()
        trainer.envs.close()
        wandb.finish()


if __name__ == "__main__":
    from typing import Annotated
    import tyro

    ExperimentType = Union[
        Annotated[DQNArgs, tyro.conf.subcommand(name="dqn")],
        Annotated[OptionDQNArgs, tyro.conf.subcommand(name="option_dqn")],
    ]

    args = tyro.cli(ExperimentType)
    main(args)
