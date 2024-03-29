import torch
from collections import deque, namedtuple
import random

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer(object):
    """Replay buffer that stores online (s, a, r, s', d) transitions for training."""

    def __init__(self, maxsize=100000):
        # TODO: Initialize the buffer using the given parameters.
        # HINT: Once the buffer is full, when adding new experience we should not care about very old data.
        self.buffer = deque(maxlen=maxsize)

    def __len__(self):
        # TODO: Return the length of the buffer (i.e. the number of transitions).
        return len(self.buffer)

    def add_experience(self, state, action, reward, next_state, done):
        # TODO: Add (s, a, r, s', d) to the buffer.
        # HINT: See the transition data type defined at the top of the file for use here.
        self.buffer.append(
            Transition(state, action, reward, next_state, torch.Tensor([done]))
        )

    def sample(self, batch_size):
        # TODO: Sample 'batch_size' transitions from the buffer.
        # Return a tuple of torch tensors representing the states, actions, rewards, next states, and terminal signals.
        # HINT: Make sure the done signals are floats when you return them.
        return tuple(
            map(
                lambda t: torch.stack(t).squeeze(),
                zip(
                    *[
                        self.buffer[idx]
                        for idx in random.sample(range(len(self.buffer)), batch_size)
                    ]
                ),
            )
        )
