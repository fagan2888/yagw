# Copyright 2020 Angelos Filos. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Game engine backend for `pycolab`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text

import numpy as np
import labmaze
from pycolab.engine import Engine
from pycolab import ascii_art
from pycolab import things as _things
from pycolab.prefab_parts import sprites as prefab_sprites
from pycolab.rendering import Observation, ObservationToArray, ObservationToFeatureArray

from . import defaults


def make_game(
    width: int = defaults.WIDTH,
    height: int = defaults.HEIGHT,
    max_rooms: int = defaults.MAX_ROOMS,
    seed: Optional[int] = defaults.SEED,
    slippery_coefficient: float = defaults.SLIPPERY_COEFFICIENT,
    default_reward: float = defaults.DEFAULT_REWARD,
    goal_reward: float = defaults.GOAL_REWARD,
    catastrophe_reward: float = defaults.CATASTROPHE_REWARD,
) -> Engine:
  """Builds a gridworld `pycolab` game.

  Args:

  Returns:
    A `pycolab` game.
  """
  maze = labmaze.RandomMaze(
      width=width,
      height=height,
      max_rooms=max_rooms,
      random_seed=seed,
      spawns_per_room=1,
      spawn_token="P",
      objects_per_room=1,
      object_token="G",
  )
  # Keep only one agent position.
  agent_positions = np.asarray(np.where(maze.entity_layer == "P"))
  I_p = np.random.choice(agent_positions.shape[-1])
  maze.entity_layer[maze.entity_layer == "P"] = " "
  maze.entity_layer[tuple(agent_positions[:, I_p])] = "P"
  # Keep only one goal.
  goal_positions = np.asarray(np.where(maze.entity_layer == "G"))
  I_g, I_c = np.random.choice(goal_positions.shape[-1], size=2, replace=False)
  maze.entity_layer[maze.entity_layer == "G"] = " "
  maze.entity_layer[tuple(goal_positions[:, I_g])] = "G"
  maze.entity_layer[tuple(goal_positions[:, I_c])] = "C"
  art = str(maze.entity_layer).split("\n")[:-1]
  sprites = {
      "P":
          ascii_art.Partial(
              AgentSprite,
              default_reward=default_reward,
              slippery_coefficient=slippery_coefficient,
              seed=seed,
          )
  }
  drapes = {
      "G":
          ascii_art.Partial(
              BoxDrape,
              reward=goal_reward,
              terminal=True,
          ),
      "C":
          ascii_art.Partial(
              BoxDrape,
              reward=catastrophe_reward,
              terminal=True,
          )
  }
  return ascii_art.ascii_art_to_game(
      art,
      what_lies_beneath=" ",
      sprites=sprites,
      drapes=drapes,
  )


class AgentSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for the gridworld agents."""

  def __init__(
      self,
      corner,
      position,
      character,
      default_reward: float,
      slippery_coefficient: float,
      seed: Optional[int] = None,
  ):
    """Inform superclass that we can go anywhere,
    but not off the board and not through the walls.
    
    Args:
      default_reward: The reward obtained when an object
        is not reached.
      slippery_coefficient: The probability of making
        a random action, in [0.0, 1.0].
      seed: The (local) random number generator seed.
    """
    assert 0.0 <= slippery_coefficient <= 1.0

    super(AgentSprite, self).__init__(
        corner,
        position,
        character,
        impassable="*",  # use the `labmaze` API
        confined_to_board=True,
    )
    self._default_reward = default_reward
    self._slippery_coefficient = slippery_coefficient
    self._rng = np.random.RandomState(seed)  # pylint: disable=no-member

  def update(
      self,
      actions,
      board,
      layers,
      backdrop,
      things,
      the_plot,
  ) -> None:
    del layers, backdrop  # Unused.

    # Slippery environment
    if self._rng.rand() < self._slippery_coefficient:
      actions = self._rng.randint(4)

    # Apply motion commands.
    if actions == 0:  # walk upward?
      self._north(board, the_plot)
    elif actions == 1:  # walk downward?
      self._south(board, the_plot)
    elif actions == 2:  # walk leftward?
      self._west(board, the_plot)
    elif actions == 3:  # walk rightward?
      self._east(board, the_plot)
    else:
      # All other actions are ignored. Although humans using the CursesUi can
      # issue action 4 (no-op), agents should only have access to actions 0-3.
      # Otherwise staying put is going to look like a terrific strategy.
      return

    # Flag for reaching a box
    hit = False

    for _, box in things.items():
      # Keep boxes only
      if isinstance(box, BoxDrape):
        # Check for collisions
        if self.position in box.positions:
          # Get reward of the box
          the_plot.add_reward(box.reward)
          hit = True
          # Check if terminal state
          if box.terminal:
            the_plot.terminate_episode()

    # Return default reward
    if not hit:
      the_plot.add_reward(self._default_reward)


class BoxDrape(_things.Drape):
  """A `Drape` for the gridworld (unmovable) boxes."""

  def __init__(
      self,
      curtain,
      character,
      reward: float,
      terminal: bool,
  ):
    """Initializes a dummy `Drape` and stores reward and
    terminal state properties.

    Args:
      reward: Reward received when an agent reaches this box.
      terminal: Flag, determines if reaching this box
        terminates the episode.
    """
    super(BoxDrape, self).__init__(curtain, character)
    self.reward = reward
    self.terminal = terminal

  def update(self, *args, **kwargs) -> None:
    """Boxes are not allowed to move."""
    pass

  @property
  def positions(self):
    """Returns the positions of the boxes found in `curtain`."""
    return [
        _things.Sprite.Position(*location.tolist())
        for location in np.argwhere(self.curtain)
    ]