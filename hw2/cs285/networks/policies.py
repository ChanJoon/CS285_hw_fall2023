import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        action = self(ptu.from_numpy(obs)).sample()
        action = ptu.to_numpy(action)

        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            # DONE: define the forward pass for a policy with a discrete action space.
            obs = self.logits_net(obs)
            # MEMO: What is difference between softmax(obs) and Categorical(logits=obs) or Categorical(softmax(obs))?
            dist = distributions.Categorical(logits=obs)
        else:
            # DONE: define the forward pass for a policy with a continuous action space.
            obs = self.mean_net(obs)
            # MEMO: We can use Normal or MultivariateNormal.
            # Since each dimension of action is not correlated, the covariance matrix is a positive definite diagonal matrix
            # dist = distributions.Normal(loc=obs, scale=torch.exp(self.logstd))
            dist = distributions.MultivariateNormal(loc=obs, covariance_matrix=torch.diag(torch.exp(self.logstd)))
        return dist

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        
        dist = self(obs)
        # negative_likelihood = F.cross_entropy(dist, actions, reduction='none')
        # loss = torch.mean(negative_likelihood * advantages)
        pi_theta = dist.log_prob(actions)
        
        # DONE: implement the policy gradient actor update.
        # MEMO: In NN libraries, the optimizer minimizes the loss. Therefore, you should pass the negative to the gradient equation.
        loss = torch.neg(torch.mean(torch.mul(pi_theta, advantages)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }
