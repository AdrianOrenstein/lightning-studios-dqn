import torch
from typing import List, Annotated, Tuple
import gymnasium as gym
from lightning.fabric import Fabric
from dataclasses import dataclass, field
import tyro

from src.dqn.agent import DQNAgent, AgentArgs
from src.dqn.network import QNetwork
from src.dqn.compressed_rbuffer import ReplayBufferSamples


@dataclass
class OptionDQNAgentArgs(AgentArgs):
    """Arguments specific to the tDQN agent algorithm."""

    option_lengths: Annotated[
        List[int],
        tyro.conf.arg(
            metavar="INT INT ...",
            help="List of temporal option lengths. For standard DQN use [1], for temporal options use e.g. [1,2,4] to allow 1, 2, or 4 step decisions.",
        ),
    ] = field(default_factory=lambda: [1])
    """List of temporal option lengths, e.g. [1] for standard DQN which takes 1 step per decision or [1, 2, 4] for temporal options which takes 1, 2, or 4 steps per decision. Each option takes the same action each time."""

    def __post_init__(self):
        # Remove duplicates and sort
        self.option_lengths = sorted(list(set(self.option_lengths)))
        if len(self.option_lengths) == 0:
            raise ValueError("Must provide at least one option length")
        if any(option_length <= 0 for option_length in self.option_lengths):
            raise ValueError("All option lengths must be positive integers")


class OptionDQNAgent(DQNAgent):
    ARGS = OptionDQNAgentArgs

    def __init__(
        self,
        args: OptionDQNAgentArgs,
        env: gym.vector.SyncVectorEnv,
        fabric: Fabric,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
    ):
        # Convert option lengths to tensor and move to device
        self.option_lengths = torch.tensor(args.option_lengths, device=fabric.device)
        super().__init__(args, env, fabric, observation_space, action_space)

    def transform_action_space(self, action_space: gym.spaces.Space) -> gym.spaces.Space:
        """Transform the action space to include options."""
        if isinstance(action_space, gym.spaces.Discrete):
            n_actions = action_space.n
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            n_actions = action_space.nvec[0]  # Assuming all dimensions have same size
        else:
            raise ValueError("Only Discrete and MultiDiscrete action spaces are supported")

        return gym.spaces.MultiDiscrete([n_actions * len(self.option_lengths)])

    def decode_action(self, action):
        """Decode expanded action into base action and option length."""
        base_action = action // len(self.option_lengths)
        repeat_idx = action % len(self.option_lengths)
        option_length = self.option_lengths[repeat_idx]
        return base_action, option_length

    def encode_action(self, base_action: int, option_idx: int) -> int:
        """Encode base action and option index into expanded action space.

        Args:
            base_action: The original action from the base action space
            option_idx: Index into self.option_lengths for the desired option length

        Returns:
            The encoded action combining both the base action and option choice
        """
        return base_action * len(self.option_lengths) + option_idx

    def update(self, data: ReplayBufferSamples) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the Q-network using a batch of option experiences from the replay buffer."""

        loss, est_q_values = self._update(
            fabric=self.fabric,
            compute_td_target_fn=self._compute_td_targets,
            q_network=self.q_network,
            target_network=self.target_network,
            optimizer=self.optimizer,
            data=data,
            gamma=self.args.gamma,
            max_norm=self.args.max_norm,
            option_lengths=self.option_lengths,
        )
        self.update_count += 1

        if self.update_count % self.args.target_network_frequency == 0:
            self.target_update_count += 1
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.detach(), est_q_values.detach()

    @staticmethod
    @torch.no_grad()
    def _compute_td_targets(
        target_network: QNetwork,
        data: ReplayBufferSamples,
        gamma: float,
        option_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TD targets for temporal Q-learning update."""
        next_q_max, _ = target_network(data.next_observations).max(dim=1)
        repeat_indices = data.actions % len(option_lengths)
        repeat_amounts = torch.index_select(option_lengths, 0, repeat_indices.squeeze())
        return data.rewards.flatten() + (gamma**repeat_amounts) * next_q_max * (1 - data.dones.flatten())
