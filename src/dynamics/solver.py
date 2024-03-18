
from typing import Optional, Tuple

import torch


class DynamicsSolver():
    """
    A class that represents a dynamics solver for simulating physical systems. It is assumed that the system is a point mass with constant forces applied for a given discrete time interval.

    Parameters:
    - mass (float): The mass of the object.
    - dimensions (int): The number of dimensions in the system.

    Methods:
    - apply_force(x, v=None, force=None, dt=1): Applies a force to the system.
    - compute_acceleration(x, v0=None): Computes the acceleration of the system.
    - compute_force(x, v0=None, a=None): Computes the force of the system.
    """

    def __init__(self, mass: float = 1, dimensions: int = 3):
        self.mass = mass
        self.dimensions = dimensions

    def apply_force(self, x: torch.Tensor, v: Optional[torch.Tensor] = None, force: Optional[torch.Tensor] = None, dt: int = 1):
        """
        Applies a force to the system.

        Parameters:
        - x (torch.Tensor): The position of the system.
        - v (torch.Tensor, optional): The velocity of the system. If not provided, it is initialized as zero.
        - force (torch.Tensor, optional): The force to be applied. If not provided, it is initialized as zero.
        - dt (int): The number of timesteps to apply the force.

        Returns:
        - x (torch.Tensor): The updated position of the system.
        - v (torch.Tensor): The updated velocity of the system.
        """
        batch_size = x.size(0)  # [batch_size, lookback, dimensions]

        if v is None:
            v = torch.zeros(x.shape, dtype=torch.float32)

        if force is None:
            force = torch.zeros((batch_size, 1, self.dimensions), dtype=torch.float32)

        acc = (force / self.mass).repeat(1, dt, 1)  # apply force for dt timesteps
        time = torch.arange(1,dt+1, dtype=torch.float32).reshape((1, dt, 1)).repeat(batch_size, 1, self.dimensions)  # time account for dt timesteps

        new_v = v[:, -1, :].unsqueeze(1).repeat(1, dt, 1) + acc * time
        new_x = x[:, -1, :].unsqueeze(1).repeat(1, dt, 1) + v[:, -1, :].unsqueeze(1).repeat(1, dt, 1) * time + 0.5 * acc * time ** 2

        v = torch.cat((v, new_v), dim=1)
        x = torch.cat((x, new_x), dim=1)

        return x, v  # [batch_size, lookback+dt, dimensions]

    def compute_acceleration(self, x: torch.Tensor, v0 : Optional[torch.Tensor] = None, return_velocity : bool = False):
        """
        Computes the acceleration of the system.

        Parameters:
        - x (torch.Tensor): The position of the system.
        - v0 (torch.Tensor, optional): The initial velocity of the system. If not provided, it is initialized as zero.
        - return_velocity (bool): Whether to return the velocity of the system.

        Returns:
        - a (torch.Tensor): The acceleration of the system. We assume the acceleration is constant between each timestep and is the result of the application of an external force.
        - v (torch.Tensor): The velocity of the system. Only returned if return_velocity is True. We assume the velocity is linear between each timestep and we report the current velocity at each timestep (i.e. not the average velocity between two coordinates).
        """
        batch_size = x.size(0)
        time_window = x.size(1)

        if v0 is None:
            v0 = torch.zeros((batch_size, 1, self.dimensions), dtype=torch.float32)

        a = torch.zeros((batch_size, time_window, self.dimensions), dtype=torch.float32) # constant acceleration between each timestep
        v = torch.zeros((batch_size, time_window, self.dimensions), dtype=torch.float32) # linear velocity between each timestep, the final velocity is reported at each timestep
        v[:, 0, :] = v0[:, 0, :]
        for i in range(time_window-1):
            a[:, i, :] = 2 * ((x[:, i+1, :] - x[:, i, :]) - v[:, i, :])
            v[:, i+1, :] = v[:, i, :] + a[:, i, :]

        if return_velocity:
            return a, v
        else:
            return a # [batch_size, lookback, dimensions]
    
    def compute_force(self, x: torch.Tensor, v: Optional[torch.Tensor] = None, v0 : Optional[torch.Tensor] = None, a: Optional[torch.Tensor] = None, return_velocity : bool = False):
        """
        Computes the force of the system.

        Parameters:
        - x (torch.Tensor): The position of the system.
        - v0 (torch.Tensor, optional): The initial velocity of the system. If not provided, it is initialized as zero. Only used if v is not provided.
        - a (torch.Tensor, optional): The acceleration of the system. If not provided, it is computed using compute_acceleration().
        - return_velocity (bool): Whether to return the velocity of the system.

        Returns:
        - force (torch.Tensor): The force of the system.
        - a (torch.Tensor): The acceleration of the system. Only returned if return_velocity is True.
        - v (torch.Tensor): The velocity of the system. Only returned if return_velocity is True. We assume the velocity is linear between each timestep and we report the current velocity at each timestep (i.e. not the average velocity between two coordinates).
        """
        if a is None:
            a, v = self.compute_acceleration(x, v0, return_velocity=True)

        force = self.mass * a

        if return_velocity:
            return force, a, v
        else:
            return force