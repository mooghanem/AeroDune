import numpy as np


class SandSimulation2D:
    """
    2D Physics Engine solving Bagnold's Saltation Flux Law coupled with 
    the Exner Bed Elevation Conservation Partial Differential Equation.
    Includes slope-wind speedup for realistic terrain-driven sand transport.
    """

    def __init__(self, elevation_grid, cell_size=30.0, dt=86400.0):
        self.Z_init = np.copy(elevation_grid).astype(np.float64)
        self.Z = np.copy(elevation_grid).astype(np.float64)
        self.ny, self.nx = self.Z.shape
        self.dx = cell_size
        self.dy = cell_size
        self.dt = dt

        # Physical Constants
        self.C_bagnold = 1.8            # Empirical sorting factor for Sinai quartz sand
        self.rho_air = 1.225            # Air density (kg/m^3)
        self.g = 9.81                   # Gravity (m/s^2)
        self.u_threshold = 6.5          # Threshold wind speed for saltation (m/s)
        self.rho_sand = 2650.0          # Quartz grain density (kg/m^3)
        self.porosity = 0.40            # Loose sand packing porosity (eta)
        self.beta_slope = 0.6           # Slope speedup coupling factor

    def compute_bagnold_flux(self, wind_speed, wind_dir_deg):
        if wind_speed <= self.u_threshold:
            return np.zeros_like(self.Z), np.zeros_like(self.Z)

        # Convert meteorological wind direction to motion direction vector
        towards_deg = (wind_dir_deg + 180.0) % 360.0
        rad = np.radians(towards_deg)
        wx = np.sin(rad)
        wy = -np.cos(rad)

        # Calculate local terrain slope along wind direction
        dZ_dy, dZ_dx = np.gradient(self.Z, self.dy, self.dx)
        directional_slope = dZ_dx * wx + dZ_dy * wy

        # Modulate local wind speed based on terrain slope (speedup on ridges, slowing in dips)
        local_u = wind_speed * (1.0 + self.beta_slope * directional_slope)
        local_u = np.maximum(local_u, 0.0)

        # Compute spatial Bagnold flux matrix
        q_mag = np.where(
            local_u > self.u_threshold,
            self.C_bagnold * (self.rho_air / self.g) * ((local_u - self.u_threshold) ** 3),
            0.0
        )

        qx = q_mag * wx
        qy = q_mag * wy

        return qx, qy

    def apply_windbreak_effects(self, qx, qy, barrier_mask, porosity=0.3):
        if barrier_mask is None or not np.any(barrier_mask):
            return qx, qy

        qx_mod = np.copy(qx)
        qy_mod = np.copy(qy)

        # Attenuate sediment flux at barrier location
        qx_mod[barrier_mask > 0] *= porosity
        qy_mod[barrier_mask > 0] *= porosity

        # Propagate downwind shelter zone
        sheltered = np.copy(barrier_mask)
        for _ in range(4):
            sheltered = np.roll(sheltered, shift=1, axis=0) | np.roll(sheltered, shift=1, axis=1)
            qx_mod[sheltered > 0] *= (porosity + 0.15)
            qy_mod[sheltered > 0] *= (porosity + 0.15)

        return qx_mod, qy_mod

    def step(self, wind_speed, wind_dir_deg, barrier_mask=None, speed_multiplier=1.0):
        effective_speed = wind_speed * speed_multiplier
        qx, qy = self.compute_bagnold_flux(effective_speed, wind_dir_deg)

        if barrier_mask is not None:
            qx, qy = self.apply_windbreak_effects(qx, qy, barrier_mask)

        # Compute spatial divergence dqx/dx + dqy/dy
        dqx_dx = np.gradient(qx, self.dx, axis=1)
        dqy_dy = np.gradient(qy, self.dy, axis=0)
        divergence = dqx_dx + dqy_dy

        # Exner Bed Elevation Update Rate (m/s)
        gamma = 1.0 / (self.rho_sand * (1.0 - self.porosity))
        dz_dt = -gamma * divergence

        # Apply elevation change per timestep
        delta_Z = np.clip(dz_dt * self.dt, -0.15, 0.15)
        self.Z += delta_Z

        return self.Z

    def get_accumulation_map(self):
        return self.Z - self.Z_init

    def reset(self):
        self.Z = np.copy(self.Z_init)
