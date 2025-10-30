# Current Context

## Physical and Chemical Problem Domain

### Astrochemical Networks

Astrochemical networks model the time evolution of chemical species in astronomical environments (molecular clouds, star-forming regions, protoplanetary disks). The governing equations form a system of coupled ordinary differential equations:

$$\frac{dx_i}{dt} = \sum_{j,k} k_{jk} x_j x_k n_H - x_i \sum_k k_{ik} x_k n_H + S_i - D_i$$

where:
- $x_i$ = fractional abundance of species $i$ relative to H nuclei (dimensionless)
- $k_{jk}$ = rate coefficient for reaction $j + k \rightarrow$ products (cm³ s⁻¹)
- $n_H$ = total H nuclei density (cm⁻³)
- $S_i$ = source terms (e.g., photodissociation, cosmic ray ionization)
- $D_i$ = destruction terms (e.g., freeze-out onto dust grains)

### Physical Complexity

**Species:** 333 chemical species including:
- Atoms: H, He, C, N, O, S, Si, etc.
- Molecules: H₂, CO, H₂O, CH₄, NH₃, etc.
- Ions: H⁺, H₃⁺, C⁺, HCO⁺, etc.
- Grain surface species: #H₂O, #CO, #CH₄, etc.
- Grain bulk species: @H₂O, @CO, etc.

**Abundance Range:** $x_i \in [10^{-20}, 1.0]$ (20 orders of magnitude)

**Rate Coefficients:** Temperature-dependent via Arrhenius/Kooij formulas:
$$k(T) = \alpha \left(\frac{T}{300}\right)^\beta \exp\left(-\frac{\gamma}{T}\right)$$

**Physical Parameters:**
- $\rho = n_H$ (density): $\sim 10^4$ to $10^9$ cm⁻³
- $T_{\text{gas}}$ (temperature): $\sim 10$ to $10^2$ K
- $\chi$ (radiation field): $0.01$ to $100$ Habing units
- $A_V$ (visual extinction): $0.1$ to $10^3$ magnitudes

### Mathematical Challenge

The system is **stiff** because:
1. Rate coefficients span many orders of magnitude ($10^{-18}$ to $10^{-9}$ cm³ s⁻¹)
2. Species timescales differ drastically (microseconds to megayears)
3. Abundances vary across 20 orders of magnitude

This requires implicit ODE solvers with adaptive timesteps, making direct integration computationally expensive.

### Conservation Constraints

**Elemental Conservation:** Total elemental abundances must remain constant:
$$\sum_i S_{ei} x_i = E_e \quad \forall \text{ elements } e$$

where $S_{ei}$ is the stoichiometric matrix (number of element $e$ atoms in species $i$) and $E_e$ is the total elemental abundance.

For example, carbon conservation:
$$x_{\text{C}} + x_{\text{CO}} + x_{\text{CH}_4} + \ldots = E_{\text{C}}$$

### Training Data Structure

**Hydrodynamic Trajectories:** Lagrangian tracers follow fluid parcels through simulations

For tracer $j$:
- Physical sequence: $\{\mathbf{p}_{j,t}\}_{t=0}^{T}$ where $\mathbf{p}_{j,t} = [\log_{10}\rho, \log_{10}\chi, \log_{10}A_V, \log_{10}T]$
- Chemical sequence: $\{\mathbf{x}_{j,t}\}_{t=0}^{T}$ where $\mathbf{x}_{j,t} \in \mathbb{R}^{333}$

**Dataset Size:**
- 9,989 Lagrangian tracers
- 298 timesteps per tracer
- Total: ~2.97 million (physical, chemical) state pairs

### Learning Objective

Replace expensive UCLCHEM ODE integration:
$$\mathbf{x}_{j,t+1} = \text{UCLCHEM}(\mathbf{x}_{j,t}, \mathbf{p}_{j,t}, \Delta t)$$

with fast neural network surrogate:
$$\mathbf{x}_{j,t+1} \approx f_\theta(\mathbf{x}_{j,t}, \mathbf{p}_{j,t})$$

where $f_\theta$ provides $10^3$ to $10^5\times$ speedup while maintaining $<1\%$ error and satisfying conservation constraints.
