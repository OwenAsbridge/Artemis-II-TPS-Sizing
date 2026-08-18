import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from fluids.atmosphere import ATMOSPHERE_1976
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

# ══════════════════════════════════════════════════════════════════════════════
# ARTEMIS II REENTRY — STAGNATION POINT HEAT FLUX AND TPS THICKNESS ANALYSIS
#
# This script:
#   1. Reads NASA AROW trajectory data (position + velocity vectors in M50 frame)
#   2. Computes altitude and speed at each time step
#   3. Queries US Standard Atmosphere 1976 for freestream air properties
#   4. Computes stagnation point heat flux using:
#        - Fay-Riddell equation  (M >= 1, hypersonic/supersonic regime)
#        - Sutton-Graves equation (M < 1,  subsonic fallback)
#   5. Iterates to find radiation equilibrium outer wall temperature
#   6. Uses finite difference heat conduction to find minimum TPS thickness
#      that keeps the inner aluminium wall below T_limit
#   7. Produces four figures:
#        Fig 1 — Altitude and speed vs time
#        Fig 2 — Freestream air properties vs time
#        Fig 3 — Heat flux and wall temperature vs time
#        Fig 4 — 3D temperature profile through shield vs time
# ══════════════════════════════════════════════════════════════════════════════


# ── Unit Conversion Factors ────────────────────────────────────────────────────
FT2M   = 0.3048    # feet  → metres
FPS2MS = 0.3048    # ft/s  → m/s


# ── Physical and Mission Constants ────────────────────────────────────────────
RE_FT  = 6371000.0 / FT2M    # Earth radius in feet (for altitude calculation)
Rn     = 2.5                  # Orion nose radius [m] — controls stagnation heating


# ── Avcoat 5026-39H/CG Material Properties (Char State) ───────────────────────
# Source: NASA CR-111834 (Aerotherm 1971), charred material values
# Used in finite difference heat conduction solver
epsilon_avcoat = 0.49       # surface emissivity — used in radiation equilibrium BC
k_avcoat       = 2.42e-01   # thermal conductivity [W/m·K]
rho_avcoat     = 2.64e+02   # density [kg/m³]
cp_avcoat      = 2.74e+03   # specific heat [J/kg·K]
sigma          = 5.67e-8    # Stefan-Boltzmann constant [W/m²·K⁴]


# ── Input File ─────────────────────────────────────────────────────────────────
# NASA AROW PROP_MAN ephemeris file
# Format: time(s)  x  y  z  vx  vy  vz  (position in ft, velocity in ft/s, M50 frame)
AROW_FILE = r"G:\My Drive\01 - Academics\01 - Courses\ME313 - Heat Transfer\2026.04.10 - Post-RTC3 to EI"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PARSE TRAJECTORY FILE
# Reads each line, extracts position and velocity vectors,
# computes scalar altitude (km) and speed (km/s) at each time step.
# ══════════════════════════════════════════════════════════════════════════════
times      = []   # elapsed time since first data point [s]
altitudes  = []   # altitude above Earth surface [km]
velocities = []   # scalar speed [km/s]
t0 = None         # reference time — set to first valid time stamp

with open(AROW_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()

        # skip header lines that can't be parsed as floats
        try:
            t = float(parts[0])
        except ValueError:
            continue

        # require all 7 columns: time, x, y, z, vx, vy, vz
        if len(parts) < 7:
            continue

        try:
            x,  y,  z  = float(parts[1]), float(parts[2]), float(parts[3])
            vx, vy, vz = float(parts[4]), float(parts[5]), float(parts[6])
        except ValueError:
            continue

        # radial distance from Earth centre [ft] → altitude [km]
        r_ft    = np.sqrt(x**2 + y**2 + z**2)
        alt_km  = (r_ft - RE_FT) * FT2M / 1000.0

        # scalar speed: magnitude of velocity vector [ft/s] → [km/s]
        spd_kms = np.sqrt(vx**2 + vy**2 + vz**2) * FPS2MS / 1000.0

        # set time origin at first data point
        if t0 is None:
            t0 = t

        times.append(t - t0)
        altitudes.append(alt_km)
        velocities.append(spd_kms)

# convert lists to numpy arrays for vectorised operations
times      = np.array(times)
altitudes  = np.array(altitudes)
velocities = np.array(velocities)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FREESTREAM AIR PROPERTIES
# Queries US Standard Atmosphere 1976 model at each altitude.
# All values are freestream (subscript ∞) — conditions far ahead of the shock.
# ══════════════════════════════════════════════════════════════════════════════
rho_free    = np.empty(len(altitudes))   # freestream density        [kg/m³]
T_free      = np.empty(len(altitudes))   # freestream temperature    [K]
P_free      = np.empty(len(altitudes))   # freestream pressure       [Pa]
mu_free     = np.empty(len(altitudes))   # freestream dyn. viscosity [Pa·s]
Speed_Sound = np.empty(len(altitudes))   # freestream speed of sound [m/s]

for i, alt in enumerate(altitudes):
    # floor altitude at 10 m to avoid model singularity at sea level
    atm = ATMOSPHERE_1976(max(alt, 0.01) * 1000.0)   # input in metres
    rho_free[i]    = atm.rho
    T_free[i]      = atm.T
    P_free[i]      = atm.P
    mu_free[i]     = atm.mu
    Speed_Sound[i] = atm.v_sonic

# Mach number at each trajectory point
Mach_free = velocities / Speed_Sound


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AIR PROPERTY FUNCTIONS
# These helper functions compute temperature-dependent air properties.
# All are defined at module level so they can be called from anywhere.
# ══════════════════════════════════════════════════════════════════════════════

def sutherland(T, mu_i, T_i):
    """
    Sutherland's law for dynamic viscosity of air.
    Uses freestream mu and T as the reference pair.

    μ(T) = μᵢ · (T/Tᵢ)^1.5 · (Tᵢ + S) / (T + S)

    Parameters
    ----------
    T    : float or array — temperature to evaluate at [K]
    mu_i : float or array — reference viscosity [Pa·s]
    T_i  : float or array — reference temperature [K]

    Returns
    -------
    mu : float or array [Pa·s]
    """
    S = 110.4   # Sutherland constant for air [K]
    return mu_i * (T / T_i)**1.5 * (T_i + S) / (T + S)


def specific_heat(T):
    """
    Specific heat of air at constant pressure cp [J/kg·K].
    Piecewise polynomial curve fit capturing:
      - ideal gas behaviour below 800 K
      - vibrational excitation 800–2000 K
      - O2 dissociation 2000–4000 K
      - N2 dissociation above 4000 K

    Parameters
    ----------
    T : float or array [K]

    Returns
    -------
    cp : float or array [J/kg·K]
    """
    T = np.asarray(T, dtype=float)
    return np.where(T < 800,  1005.0,
           np.where(T < 2000, 1005.0 + 0.2  * (T - 800),
           np.where(T < 4000, 1245.0 + 0.15 * (T - 2000),
                               1545.0 + 0.05 * (T - 4000))))


def k_air_sutherland(T):
    """
    Thermal conductivity of air [W/m·K] via Sutherland's law.
    Note: uses a different Sutherland constant (194 K) than viscosity (110.4 K).

    Parameters
    ----------
    T : float or array [K]

    Returns
    -------
    k : float or array [W/m·K]
    """
    k_ref = 0.02624   # reference conductivity at T_ref [W/m·K]
    T_ref = 300.0     # reference temperature [K]
    S_k   = 194.0     # Sutherland constant for thermal conductivity of air [K]
    return k_ref * (T / T_ref)**1.5 * (T_ref + S_k) / (T + S_k)


def Pr_air(T, mu, cp):
    """
    Prandtl number of air [-].
    Derived from definition: Pr = μ · cp / k
    Varies mildly with temperature (~0.68–0.74 across reentry range).

    Parameters
    ----------
    T  : float or array [K]
    mu : float or array [Pa·s]
    cp : float or array [J/kg·K]

    Returns
    -------
    Pr : float or array [-]
    """
    k = k_air_sutherland(T)
    return (mu * cp) / k


def Le_air(T, mu, cp, Sc=0.75):
    """
    Lewis number of air [-].
    Relates thermal diffusivity to mass diffusivity: Le = Sc / Pr
    Schmidt number Sc ≈ 0.75 is treated as constant for air.
    Lewis number ≈ 1.4 at typical reentry conditions.

    Parameters
    ----------
    T   : float or array [K]
    mu  : float or array [Pa·s]
    cp  : float or array [J/kg·K]
    Sc  : float          Schmidt number [-], default 0.75

    Returns
    -------
    Le : float or array [-]
    """
    return Sc / Pr_air(T, mu, cp)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HEAT FLUX CALCULATION
# Computes stagnation point heat flux at each trajectory point using:
#   Fay-Riddell  (M ≥ 1) — full boundary layer equation with dissociation term
#   Sutton-Graves (M < 1) — simpler empirical correlation for subsonic points
#
# The function accepts arrays for all inputs and loops internally,
# applying the appropriate equation at each point.
# ══════════════════════════════════════════════════════════════════════════════

def Fay_Riddell(vel_inf, R, rho_inf, T_inf, P_inf, mu_inf, M_inf, T_wall):
    """
    Stagnation point heat flux via Fay-Riddell (M≥1) or Sutton-Graves (M<1).

    Parameters
    ----------
    vel_inf : array  — freestream velocity [km/s]
    R       : float  — nose radius [m]
    rho_inf : array  — freestream density [kg/m³]
    T_inf   : array  — freestream temperature [K]
    P_inf   : array  — freestream pressure [Pa]
    mu_inf  : array  — freestream dynamic viscosity [Pa·s]
    M_inf   : array  — freestream Mach number [-]
    T_wall  : array  — outer wall temperature [K] (radiation equilibrium)

    Returns
    -------
    q_wall : array [W/m²]
    """

    # ── Constants ──────────────────────────────────────────────────────────
    gamma = 1.4                                       # ratio of specific heats for air
    R_g   = 8.31446 / 0.0289644                       # specific gas constant for air [J/kg·K]
    h_D   = 0.21 * 15.576e6 + 0.79 * 33.550e6        # dissociation enthalpy of air [J/kg]
                                                      # weighted: 21% O2 + 79% N2
    K_sg  = 1.7415e-4                                 # Sutton-Graves constant for Earth air

    # convert velocity from km/s to m/s for all calculations
    vel_ms = vel_inf * 1e3

    q_wall = np.zeros(len(altitudes))   # output heat flux array [W/m²]

    for i in range(len(q_wall)):

        if M_inf[i] >= 1.0:
            # ──────────────────────────────────────────────────────────────
            # FAY-RIDDELL EQUATION
            # Valid for M ≥ 1 (supersonic/hypersonic continuum flow)
            # All subscript _e values are at the boundary layer edge
            # (immediately behind the normal shock)
            # ──────────────────────────────────────────────────────────────

            # post-shock temperature [K] via Rankine-Hugoniot relation
            T_e = T_inf[i] * (1 + (2*gamma*(M_inf[i]**2 - 1)) / (gamma + 1)) \
                           * (2 + (gamma - 1)*M_inf[i]**2) / ((gamma + 1)*M_inf[i]**2)

            # post-shock pressure [Pa] via Rankine-Hugoniot relation
            P_e = P_inf[i] * (1 + (2*gamma / (gamma + 1)) * (M_inf[i]**2 - 1))

            # wall pressure equals edge pressure (no normal pressure gradient)
            P_w = P_e

            # post-shock density [kg/m³] via Rankine-Hugoniot relation
            rho_e = rho_inf[i] * ((gamma + 1)*M_inf[i]**2) / (2 + (gamma - 1)*M_inf[i]**2)

            # wall density [kg/m³] from ideal gas law at wall conditions
            rho_w = P_w / (R_g * T_wall[i])

            # dynamic viscosity at edge and wall via Sutherland's law [Pa·s]
            mu_e = sutherland(T_e,       mu_inf[i], T_inf[i])
            mu_w = sutherland(T_wall[i], mu_inf[i], T_inf[i])

            # specific heat at edge and wall conditions [J/kg·K]
            Cp_e = specific_heat(T_e)
            Cp_w = specific_heat(T_wall[i])

            # Prandtl number at edge — evaluated at edge conditions
            Pr = Pr_air(T_e, mu_e, Cp_e)

            # Lewis number at edge — governs dissociation energy transport
            Le = Le_air(T_e, mu_e, Cp_e)

            # stagnation enthalpy [J/kg] = static enthalpy + kinetic energy
            h_0e = Cp_e * T_e + 0.5 * vel_ms[i]**2

            # wall enthalpy [J/kg]
            h_w = Cp_w * T_wall[i]

            # Newtonian stagnation velocity gradient [1/s]
            dV_dx = (1/R) * np.sqrt(np.maximum(2*(P_e - P_inf[i]) / rho_e, 0.0))

            # Fay-Riddell equation [W/m²]
            q_wall[i] = (0.763
                         * Pr**(-0.6)
                         * (rho_e * mu_e)**0.4
                         * (rho_w * mu_w)**0.1
                         * dV_dx**0.5
                         * (h_0e - h_w)
                         * (1 + (Le**0.52 - 1) * (h_D / h_0e)))

        else:
            # ──────────────────────────────────────────────────────────────
            # SUTTON-GRAVES EQUATION
            # Used for M < 1 (subsonic) — heat flux is negligible here
            # q = K · √(ρ∞/Rn) · V³
            # ──────────────────────────────────────────────────────────────
            q_wall[i] = K_sg * np.sqrt(rho_inf[i] / R) * vel_ms[i]**3

    return q_wall


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RADIATION EQUILIBRIUM WALL TEMPERATURE ITERATION
#
# The outer wall temperature T_w adjusts until the surface radiates away
# exactly the incoming convective heat flux:
#
#     q_w = ε · σ · T_w⁴   →   T_w = (q_w / ε·σ)^0.25
#
# Solved by fixed-point iteration starting at 300 K cold wall assumption.
# Convergence criterion: max change in T_w across all points < 0.1 K
# ══════════════════════════════════════════════════════════════════════════════
tol_rad = 0.1   # convergence tolerance [K]

# initialise wall temperature at 300 K (cold wall starting guess)
T_w = np.full(len(altitudes), 300.0)
q_w = np.zeros(len(altitudes))

for iteration in range(50):

    # Step 1: compute heat flux with current wall temperature
    q_w = Fay_Riddell(velocities, Rn, rho_free, T_free, P_free, mu_free, Mach_free, T_w)

    # Step 2: update wall temperature from radiation equilibrium
    T_w_new = (q_w / (epsilon_avcoat * sigma)) ** 0.25

    # Step 3: keep subsonic / zero-flux points at 300 K
    mask    = q_w > 0
    T_w_new = np.where(mask, T_w_new, 300.0)

    # Step 4: check convergence
    if np.all(np.abs(T_w_new - T_w) < tol_rad):
        break

    # Step 5: update for next iteration
    T_w = T_w_new


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FINITE DIFFERENCE HEAT CONDUCTION / TPS THICKNESS SIZING
#
# Solves the 1D transient heat equation through the shield thickness:
#
#     ρ · cp · ∂T/∂t = k · ∂²T/∂x²
#
# Boundary conditions:
#   x = 0 (outer surface): Dirichlet BC → T[0] = T_w(t)
#                          Uses the radiation equilibrium wall temperature
#                          directly — already accounts for convection/radiation
#                          balance so no flux term needed here
#   x = L (inner wall):    adiabatic    → dT/dx = 0 (worst case, no heat loss)
#
# Uses bisection to find minimum thickness L such that the inner wall
# temperature never exceeds T_limit.
# Factor of safety applied to the resulting thickness, not the temperature.
# ══════════════════════════════════════════════════════════════════════════════

def find_shield_thickness(T_wall_bc, t, k, rho, cp,
                          T_init, T_limit, N, L_min, L_max, tol, FS):
    """
    Find minimum heat shield thickness to keep inner wall below T_limit.
    Uses prescribed outer surface temperature (Dirichlet BC) from radiation
    equilibrium iteration. Bisects on thickness L.

    Parameters
    ----------
    T_wall_bc : array [K]     outer surface temperature history (T_w from FR)
    t         : array [s]     time points
    k         : float [W/m·K] material thermal conductivity
    rho       : float [kg/m³] material density
    cp        : float [J/kg·K] material specific heat
    T_init    : float [K]     initial temperature everywhere
    T_limit   : float [K]     max allowable inner wall temperature
    N         : int           number of nodes through thickness
    L_min     : float [m]     minimum thickness in bisection search
    L_max     : float [m]     maximum thickness in bisection search
    tol       : float [m]     bisection convergence tolerance
    FS        : float         factor of safety applied to final thickness

    Returns
    -------
    L_design    : float [m]        minimum thickness × FS
    T_inner     : array [K]        inner wall temperature history
    T_surf_out  : array [K]        outer surface temperature history
    T_profile   : array (Nt, N)    full temperature field vs time
    """

    # build interpolant for outer surface temperature
    # allows sub-step evaluation during time integration
    T_w_interp = interp1d(t, T_wall_bc,
                          kind='linear',
                          bounds_error=False,
                          fill_value=(T_wall_bc[0], T_wall_bc[-1]))

    def run_diffusion(L):
        """
        Explicit finite difference solver for a given thickness L.
        Outer surface prescribed as T_w(t) — Dirichlet BC.
        Inner wall adiabatic — Neumann BC.
        """
        dx    = L / (N - 1)             # node spacing [m]
        alpha = k / (rho * cp)           # thermal diffusivity [m²/s]
        dt    = np.diff(t)               # time step sizes [s]

        T          = np.full(N, T_init, dtype=float)
        T_inner    = np.zeros(len(t))
        T_surf_out = np.zeros(len(t))
        T_profile  = np.zeros((len(t), N))

        # store initial conditions
        T_inner[0]    = T_init
        T_surf_out[0] = float(T_w_interp(t[0]))
        T_profile[0]  = T

        for n in range(len(t) - 1):

            # per-step Fourier number — subdivide if needed for stability
            F_n  = alpha * dt[n] / dx**2
            sub  = int(np.ceil(F_n / 0.4)) if F_n > 0.4 else 1
            dt_n = dt[n] / sub

            for s in range(sub):

                t_sub = t[n] + s * dt_n   # current sub-step time
                F     = alpha * dt_n / dx**2
                T_new = T.copy()

                # ── outer surface — prescribed T_w (Dirichlet BC) ──────────
                # T_w already accounts for convection/radiation balance
                # from the Fay-Riddell radiation equilibrium iteration
                T_new[0] = float(T_w_interp(t_sub))

                # ── interior nodes — standard explicit FD update ───────────
                # T_new[i] = T[i] + F·(T[i+1] - 2·T[i] + T[i-1])
                for i in range(1, N - 1):
                    T_new[i] = T[i] + F * (T[i+1] - 2*T[i] + T[i-1])

                # ── inner wall — adiabatic (Neumann BC) ───────────────────
                # no heat escapes through back face — conservative assumption
                # ghost node: T[N] = T[N-2]
                T_new[-1] = T[-1] + 2*F * (T[-2] - T[-1])

                T = T_new

            T_inner[n+1]    = T[-1]
            T_surf_out[n+1] = T[0]
            T_profile[n+1]  = T

        return T_inner, T_surf_out, T_profile

    def exceeds_limit(L):
        """Returns True if inner wall exceeds T_limit for this thickness."""
        T_inner, _, _ = run_diffusion(L)
        return np.max(T_inner) > T_limit

    # ── bisection on thickness ─────────────────────────────────────────────
    # converges to minimum L that keeps inner wall below T_limit
    lo, hi = L_min, L_max
    while (hi - lo) > tol:
        mid = (lo + hi) / 2.0
        if exceeds_limit(mid):
            lo = mid   # too thin
        else:
            hi = mid   # thick enough — try thinner

    # minimum thickness that just meets the limit
    L_min_req = hi

    # apply factor of safety to thickness — not to temperature
    L_design = L_min_req * FS

    # run final diffusion at design thickness for output arrays
    T_inner, T_surface, T_profile = run_diffusion(L_design)

    return L_design, T_inner, T_surface, T_profile


# ── Run the thickness solver ───────────────────────────────────────────────────
T_limit = 422.0   # max allowable inner wall temperature [K]
FS      = 1.5     # factor of safety applied to thickness

L_req, T_inner, T_surface, T_profile = find_shield_thickness(
    T_wall_bc = T_w,
    t         = times,
    k         = k_avcoat,
    rho       = rho_avcoat,
    cp        = cp_avcoat,
    T_init    = 300.0,
    T_limit   = T_limit,
    N         = 50,
    L_min     = 0.005,
    L_max     = 0.30,
    tol       = 0.001,
    FS        = FS)

print(f"Design thickness (incl. FS={FS}): {L_req*100:.2f} cm")
print(f"Peak inner wall temp:             {np.max(T_inner):.1f} K")
print(f"Peak outer surface temp:          {np.max(T_surface):.1f} K")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PLOTS
# ══════════════════════════════════════════════════════════════════════════════

color_alt   = '#185FA5'
color_vel   = '#993C1D'
color_rho   = '#6A0DAD'
color_temp  = '#CC7700'
color_pres  = '#1A7A4A'
color_mach  = '#C2185B'
color_qw    = '#E63946'
color_tw    = '#FF6F00'


# ── Figure 1 — Altitude and Speed vs Time ─────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(16, 6))
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

ax1.set_xlabel('Elapsed Time (s)', fontsize=12)
ax1.set_ylabel('Altitude (km)', color=color_alt, fontsize=12)
ax1.plot(times, altitudes, color=color_alt, linewidth=2.0, label='Altitude (km)')
ax1.tick_params(axis='y', labelcolor=color_alt)
ax1.set_ylim(0, 140)
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.5)
ax1.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)

ax1b = ax1.twinx()
ax1b.set_ylabel('Speed (km/s)', color=color_vel, fontsize=12)
ax1b.plot(times, velocities, color=color_vel, linewidth=2.0, linestyle='--', label='Speed (km/s)')
ax1b.tick_params(axis='y', labelcolor=color_vel)
ax1b.set_ylim(0, 12)
ax1b.yaxis.set_minor_locator(ticker.AutoMinorLocator())

all_lines  = ax1.get_legend_handles_labels()[0]  + ax1b.get_legend_handles_labels()[0]
all_labels = ax1.get_legend_handles_labels()[1]  + ax1b.get_legend_handles_labels()[1]
ax1.legend(all_lines, all_labels, loc='upper right', fontsize=9, framealpha=0.85)
ax1.set_xlim(times[0], times[-1])
ax1.set_title('Artemis II Reentry — Altitude and Speed vs Time\n', fontsize=11, pad=8)
plt.tight_layout()
plt.savefig('fig1_altitude_speed.png', dpi=150, bbox_inches='tight')
print("Saved fig1_altitude_speed.png")
plt.show()


# ── Figure 2 — Freestream Air Properties vs Time ──────────────────────────────
fig2, ax2 = plt.subplots(figsize=(16, 6))
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')

ax2.set_xlabel('Elapsed Time (s)', fontsize=12)
ax2.set_ylabel('Density (kg/m³)', color=color_rho, fontsize=12)
ax2.plot(times, rho_free, color=color_rho, linewidth=1.5, linestyle='-.', label='Density (kg/m³)')
ax2.tick_params(axis='y', labelcolor=color_rho)
ax2.set_yscale('log')
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.5)
ax2.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)

ax2b = ax2.twinx()
ax2b.set_ylabel('Temperature (K)', color=color_temp, fontsize=12)
ax2b.plot(times, T_free, color=color_temp, linewidth=1.5, linestyle=':', label='Temperature (K)')
ax2b.tick_params(axis='y', labelcolor=color_temp)
ax2b.yaxis.set_minor_locator(ticker.AutoMinorLocator())

ax2c = ax2.twinx()
ax2c.spines['right'].set_position(('outward', 70))
ax2c.set_ylabel('Pressure (Pa)', color=color_pres, fontsize=12)
ax2c.plot(times, P_free, color=color_pres, linewidth=1.5,
          linestyle=(0, (3, 1, 1, 1)), label='Pressure (Pa)')
ax2c.tick_params(axis='y', labelcolor=color_pres)
ax2c.set_yscale('log')
ax2c.yaxis.set_minor_locator(ticker.AutoMinorLocator())

ax2d = ax2.twinx()
ax2d.spines['right'].set_position(('outward', 150))
ax2d.set_ylabel('Speed of Sound (m/s)', color=color_mach, fontsize=12)
ax2d.plot(times, Speed_Sound, color=color_mach, linewidth=1.5,
          linestyle=(0, (5, 1)), label='Speed of Sound (m/s)')
ax2d.tick_params(axis='y', labelcolor=color_mach)
ax2d.yaxis.set_minor_locator(ticker.AutoMinorLocator())

all_lines  = (ax2.get_legend_handles_labels()[0]  + ax2b.get_legend_handles_labels()[0] +
              ax2c.get_legend_handles_labels()[0]  + ax2d.get_legend_handles_labels()[0])
all_labels = (ax2.get_legend_handles_labels()[1]  + ax2b.get_legend_handles_labels()[1] +
              ax2c.get_legend_handles_labels()[1]  + ax2d.get_legend_handles_labels()[1])
ax2.legend(all_lines, all_labels, loc='lower right', fontsize=9, framealpha=0.85)
ax2.set_xlim(times[0], times[-1])
ax2.set_title('Freestream Air Properties vs Time (US Standard Atmosphere 1976)', fontsize=11, pad=8)
plt.tight_layout()
plt.savefig('fig2_freestream.png', dpi=150, bbox_inches='tight')
print("Saved fig2_freestream.png")
plt.show()


# ── Figure 3 — Heat Flux and Wall Temperature vs Time ─────────────────────────
fig3, ax3 = plt.subplots(figsize=(16, 6))
fig3.patch.set_facecolor('white')
ax3.set_facecolor('white')

ax3.set_xlabel('Elapsed Time (s)', fontsize=12)
ax3.set_ylabel('Heat Flux (W/m²)', color=color_qw, fontsize=12)
ax3.plot(times, q_w, color=color_qw, linewidth=2.0, label='Heat Flux $\\dot{q}_w$ (W/m²)')
ax3.tick_params(axis='y', labelcolor=color_qw)
ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax3.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.5)
ax3.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)

ax3b = ax3.twinx()
ax3b.set_ylabel('Wall Temperature (K)', color=color_tw, fontsize=12)
ax3b.plot(times, T_w, color=color_tw, linewidth=1.5, linestyle='--', label='Wall Temp $T_w$ (K)')
ax3b.tick_params(axis='y', labelcolor=color_tw)
ax3b.yaxis.set_minor_locator(ticker.AutoMinorLocator())

all_lines  = ax3.get_legend_handles_labels()[0]  + ax3b.get_legend_handles_labels()[0]
all_labels = ax3.get_legend_handles_labels()[1]  + ax3b.get_legend_handles_labels()[1]
ax3.legend(all_lines, all_labels, loc='upper right', fontsize=9, framealpha=0.85)
ax3.set_xlim(times[0], times[-1])
ax3.set_title('Stagnation Point Heat Flux and Wall Temperature vs Time\n'
              'Radiation equilibrium wall temperature', fontsize=11, pad=8)
plt.tight_layout()
plt.savefig('fig3_heat_flux.png', dpi=150, bbox_inches='tight')
print("Saved fig3_heat_flux.png")
plt.show()


# ── Figure 4 — 3D Temperature Profile Through Shield vs Time ──────────────────
x_nodes = np.linspace(0, L_req * 100, 50)   # depth in cm

step      = max(1, len(times) // 10000)
t_plot    = times[::step]
prof_plot = gaussian_filter(T_profile[::step, :], sigma=1.5)

X, Y = np.meshgrid(x_nodes, t_plot)

fig4 = plt.figure(figsize=(14, 8))
fig4.patch.set_facecolor('white')
ax4  = fig4.add_subplot(111, projection='3d')

surf = ax4.plot_surface(X, Y, prof_plot,
                        cmap='jet',
                        linewidth=0,
                        antialiased=True,
                        alpha=0.92)

fig4.colorbar(surf, ax=ax4, shrink=0.5, aspect=10,
              label='Temperature (K)', pad=0.1)

ax4.set_xlabel('Depth into shield (cm)', fontsize=11, labelpad=10)
ax4.set_ylabel('Elapsed time (s)',        fontsize=11, labelpad=10)
ax4.set_zlabel('Temperature (K)',         fontsize=11, labelpad=10)
ax4.set_title(f'Heat Shield Temperature Profile Through Thickness vs Time\n'
              f'Design thickness = {L_req*100:.2f} cm (FS={FS})  |  '
              f'Inner wall limit = {T_limit:.0f} K',
              fontsize=12, pad=15)

ax4.view_init(elev=30, azim=-60)

plt.tight_layout()
plt.savefig('fig4_temp_profile_3d.png', dpi=150, bbox_inches='tight')
print("Saved fig4_temp_profile_3d.png")
plt.show()
