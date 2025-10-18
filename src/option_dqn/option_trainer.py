import numpy as np
from typing import Optional
from lightning.fabric import Fabric
from wandb.integration.lightning.fabric import WandbLogger
from torchrl._utils import timeit
from dataclasses import dataclass

from src.dqn.utils import TimeScaleMeanBuffer
from src.dqn.batch_trainer import BatchTrainer, TrainerArgs, Experience
from src.option_dqn.option_agent import OptionDQNAgent


@dataclass
class BatchedOptionTrainerArgs(TrainerArgs):
    """Arguments for temporal option training."""

    option_deliberation_cost: Optional[float] = 0
    """Cost added to reward when agent makes a new decision"""


class BatchedOptionTrainer(BatchTrainer):
    """Trainer class for temporal DQN with action repeats."""

    ARGS = BatchedOptionTrainerArgs

    def __init__(
        self,
        args: BatchedOptionTrainerArgs,
        fabric: Fabric,
        logger: WandbLogger,
        envs,
        agent: OptionDQNAgent,
    ):
        super().__init__(args, fabric, logger, envs, agent)
        self.args: BatchedOptionTrainerArgs = args  # Type hint for better IDE support
        self.agent: OptionDQNAgent = agent  # Type hint for better IDE support

        self.reward_per_decision_without_cost = TimeScaleMeanBuffer(100)

        self.additional_metrics_to_log.update(
            {
                "agent/reward_per_decision_over_last_100_decisions_without_deliberation_cost": self.reward_per_decision_without_cost.mean,
            }
        )

    def execute_agent_decision(self, action) -> tuple[list[Experience], bool, bool, dict]:
        """Execute an option (repeated action) in the environment.

        Args:
            action: The action to execute

        Returns:
            Tuple of (experiences, termination, truncation, info)
        """
        base_action, option_length = self.agent.decode_action(action[0])

        assert option_length > 0, "Option length must be greater than 0"

        # Store initial observation for experience
        accumulated_reward = 0

        # Apply computation cost if specified
        if self.args.option_deliberation_cost is not None:
            accumulated_reward = self.args.option_deliberation_cost

        # Execute action for option_length steps
        for repeat_i in range(option_length):
            with timeit("Environment Step"):
                next_obs, reward, termination, truncation, info = self.envs.step(np.array([base_action]))
                self.reward_per_step.add(reward[0])
                self.step_count += 1
                self.frame_no = int(info["frame_number"][0])

                # Accumulate discounted reward
                accumulated_reward += (self.agent.args.gamma**repeat_i) * reward[0]

                if termination or truncation:
                    break

        # Track accumulated reward.
        self.reward_per_decision.add(accumulated_reward)
        self.reward_per_decision_without_cost.add(accumulated_reward - self.args.option_deliberation_cost)

        experiences = [
            Experience(
                observation=self.obs,
                next_observation=next_obs,
                action=action[0],
                reward=accumulated_reward,
                done=termination,
                info=info,
            )
        ]
        self.obs = next_obs
        return experiences, termination, truncation, info
