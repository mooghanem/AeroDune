import numpy as np


class SandSimulation2D:
    """
    2D Physics Engine solving Bagnold's Saltation Flux Law coupled with 
    the Exner Bed Elevation Conservation Partial Differential Equation.
    """

    def __init__(self, elevation_grid, cell_size=30.0, dt=86400.0):
        """
        Args:
            elevation_grid (2D np.ndarray): Initial bed height z(x,y) in meters.
            cell_size (float): Spatial grid resolution (dx = dy) in meters.
            dt (float): Physical time step per iteration in seconds (default = 1 day = 86,400s).
        """
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

    def compute_bagnold_flux(self, wind_speed, wind_dir_deg):
        """
        Calculates 2D sediment transport flux vectors (qx, qy) in kg/(m*s).
        """
        if wind_speed <= self.u_threshold:
            return np.zeros_like(self.Z), np.zeros_like(self.Z)

        # Bagnold Saltation Magnitude (kg / (m * s))
        q_mag = self.C_bagnold * (self.rho_air / self.g) * ((wind_speed - self.u_threshold) ** 3)

        # Convert meteorological wind direction (FROM) to sand motion direction (TOWARDS)
        towards_deg = (wind_dir_deg + 180.0) % 360.0
        rad = np.radians(towards_deg)

        # Vector decomposition into 2D grid coordinates
        qx = q_mag * np.sin(rad) * np.ones_like(self.Z)
        qy = -q_mag * np.cos(rad) * np.ones_like(self.Z) # Negative due to array row orientation

        return qx, qy

    def apply_windbreak_effects(self, qx, qy, barrier_mask, barrier_height=4.0, porosity=0.3):
        """
        Applies localized flux attenuation downwind of user-placed windbreak fences/tree lines.
        """
        if barrier_mask is None or not np.any(barrier_mask):
            return qx, qy

        qx_mod = np.copy(qx)
        qy_mod = np.copy(qy)

        # Attenuate sediment flux at and around barrier locations
        reduction_factor = porosity # Residual flux passing through barrier
        qx_mod[barrier_mask > 0] *= reduction_factor
        qy_mod[barrier_mask > 0] *= reduction_factor

        # Simple 3-cell downwind shelter effect propagation
        sheltered = np.copy(barrier_mask)
        for _ in range(3):
            sheltered = np.roll(sheltered, shift=1, axis=0) | np.roll(sheltered, shift=1, axis=1)
            qx_mod[sheltered > 0] *= (reduction_factor + 0.2)
            qy_mod[sheltered > 0] *= (reduction_factor + 0.2)

        return qx_mod, qy_mod

    def step(self, wind_speed, wind_dir_deg, barrier_mask=None, speed_multiplier=1.0):
        """
        Executes one temporal step of the 2D Exner PDE:
        dz/dt = -1 / (rho_s * (1 - eta)) * (dqx/dx + dqy/dy)
        """
        effective_speed = wind_speed * speed_multiplier
        qx, qy = self.compute_bagnold_flux(effective_speed, wind_dir_deg)

        if barrier_mask is not None:
            qx, qy = self.apply_windbreak_effects(qx, qy, barrier_mask)

        # Compute spatial derivatives using 2nd-order central differences
        dqx_dx = np.gradient(qx, self.dx, axis=1)
        dqy_dy = np.gradient(qy, self.dy, axis=0)

        divergence = dqx_dx + dqy_dy

        # Exner Bed Height Change (m/s)
        gamma = 1.0 / (self.rho_sand * (1.0 - self.porosity))
        dz_dt = -gamma * divergence

        # Update bed elevation Z
        delta_Z = dz_dt * self.dt
        
        # Soft numerical smoothing to prevent grid instability
        delta_Z = np.clip(delta_Z, -2.0, 2.0)
        self.Z += delta_Z

        # Ensure elevation doesn't drop below hard bedrock floor (-5m relative to init)
        self.Z = np.maximum(self.Z, self.Z_init - 5.0)

        return self.Z

    def get_accumulation_map(self):
        """
        Returns net sand accumulation/erosion relative to initial DEM state (in meters).
        """
        return self.Z - self.Z_init

    def reset(self):
        """Resets bed height grid to initial state."""
        self.Z = np.copy(self.Z_init)
