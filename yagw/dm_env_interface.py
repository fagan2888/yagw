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
"""Grid world `dm_env.Environment` based on `pycolab` engine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Optional, Text, Union

import dm_env
import numpy as np
import matplotlib.pyplot as plt

from .pycolab_interface import make_game, Observation, ObservationToArray, ObservationToFeatureArray


class GridWorld(dm_env.Environment):
  """Generic API for grid world environments."""

  def __init__(
      self,
      num_layouts: int = -1,
      max_steps_count: int = 50,
  ):
    """Initializes a new grid world environment.

    Args:
      num_layouts: The number of distinct maze configurations,
        if `-1` then there is no limit.
      max_steps_count: The number of time-steps before the
        episode is terminated.
    """
    if -1 < num_layouts == 0:
      raise ValueError(
          "A positive or `-1` number of layouts should be provided")

    # Game state used by engine.
    self._reset_next_step = True
    self._reset_game_engine = False
    self._max_steps_count = max_steps_count
    self._steps_count = 0
    self._state = None
    self._np_random = np.random.RandomState(None)  # pylint: disable=no-member
    self._layout_ids = self._np_random.randint(
        int(1e6),
        size=num_layouts if num_layouts != -1 else int(1e6),
    )
    self._current_layout_id = self._np_random.choice(self._layout_ids)
    # Pycolab game.
    self._game = make_game(seed=self._current_layout_id)
    # Renderer.
    self._renderer = dict()

  def observation_spec(self) -> dm_env.specs.BoundedArray:
    """Returns the observation spec."""
    return dm_env.specs.BoundedArray(
        shape=(
            len(self._game.things.keys()) + 2,
            self._game.rows,
            self._game.cols,
        ),
        dtype=np.float32,
        minimum=0.0,
        maximum=1.0,
    )

  def action_spec(self) -> dm_env.specs.DiscreteArray:
    """Returns the action spec."""
    return dm_env.specs.DiscreteArray(4)  # {"NORTH", "SOUTH", "EAST", "WEST"}

  def seed(self, state: int) -> None:
    """Sets random number generators' state."""
    self._np_random = np.random.RandomState(state)  # pylint: disable=no-member

  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode.

    Returns:
      A `dm_env.Timestep`.
    """
    self._reset_next_step = False
    self._steps_count = 0
    # Reset the game engine.
    if self._reset_game_engine:
      self._current_layout_id = self._np_random.choice(self._layout_ids)
      self._game = make_game(seed=self._current_layout_id)
    # Initialize the game.
    self._game.its_showtime()
    # Sample initial configuration.
    self._state = self._board2state(self._game._board)
    return dm_env.restart(self._state)

  def step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action.

    Args:
      action: The action taken by the agent in the game.

    Returns:
      A `dm_env.Timestep`.
    """
    if self._reset_next_step:
      return self.reset()
    self._steps_count += 1
    # Perform a step in the game.
    _, reward, _ = self._game.play(action)
    self._state = self._board2state(self._game._board)
    # Termination condition.
    if self._game.game_over or self._steps_count >= self._max_steps_count:
      self._reset_next_step = True
      self._reset_game_engine = True
      return dm_env.termination(reward, self._state)
    else:
      return dm_env.transition(reward, self._state)

  def render(
      self,
      mode: Text = "human",
  ) -> Optional[Union[np.ndarray, Text]]:
    """Renders the environment.

    Args:
      mode: OpenAI Gym rendering modes:
        - `human`: render to the current display or terminal and
            return nothing. Usually for human consumption.
        - `rgb_array`: Return an numpy.ndarray with shape (x, y, 3),
            representing RGB values for an x-by-y pixel image, suitable
            for turning into a video.
        - `ansi`: Return a string (str) or StringIO.StringIO containing a
            terminal-style text representation. The text can include newlines
            and ANSI escape sequences (e.g. for colors).
    """
    assert mode in ("rgb_array", "human", "ansi")

    # Lazy initialization of different renderers.
    if mode in ("rgb_array", "human"):
      if not mode is self._renderer:
        value_mapping = {
            "P": (0, 0, 1),  # agent -> blue
            "G": (0, 1, 0),  # goal -> green
            "C": (1, 0, 0),  # catastrophe -> red
            "*": (0, 0, 0),  # wall -> black
            " ": (1, 1, 1),  # space -> white
        }
        self._renderer[mode] = ObservationToArray(
            value_mapping,
            dtype=np.float32,
        )
    elif mode is "ansi":
      if not mode is self._renderer:
        value_mapping = {x: x for x in ("P", "G", "C", "*", " ")}
        self._renderer[mode] = ObservationToArray(
            value_mapping,
            dtype=str,
        )

    # Apply transofrmations.
    display = self._renderer[mode](self._game._board).swapaxes(0, 2)

    # Renders or returns the display.
    if mode is "human":
      fig, ax = plt.subplots(figsize=(3.0, 3.0))
      ax.imshow(display)
      ax.grid(None)
      ax.axis("off")
      fig.tight_layout()
      fig.show()
    else:
      return display

  @staticmethod
  def _board2state(board: Observation) -> np.ndarray:
    """Converts the `pycolab.engine.Engine._board` to
    a binary tensor, its channel represents an object-type.

    Args:
      board: A `namedtuple` with attributes:
        * `layers`, a dictionary with keys the character symbol and a boolean
        `[width, height]` mask,
        * `board`: a 2D array.

    Returns:
      The game state in a binary tensor format, with shape
      `[channels, width, height]`.
    """
    return ObservationToFeatureArray(
        collections.OrderedDict(sorted(board.layers.items())))(board)
