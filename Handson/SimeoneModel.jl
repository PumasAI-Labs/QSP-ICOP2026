#=
================================================================================
SESSION 1: Introduction & Pumas Model Building
================================================================================

Learning Objectives:
1. Understand the Simeoni PK-PD model structure and biological motivation
2. Build a Pumas @model with proper parameter domains
3. Understand @param, @random, @pre, @init, @vars, @dynamics, @derived blocks

Workshop Duration: 60 minutes

Reference: Simeoni et al. (2004) Cancer Research 64:1094-1101
           "Predictive Pharmacokinetic-Pharmacodynamic Modeling of Tumor
            Growth Kinetics in Xenograft Models"
=#

# ==============================================================================
# SETUP
# ==============================================================================

# Load required packages
using Pumas
using AlgebraOfGraphics, CairoMakie
using DataFramesMeta
using PumasUtilities
using CSV

# Set AlgebraOfGraphics theme
set_aog_theme!()

# ==============================================================================
# PART 1: THE SIMEONI MODEL - BIOLOGICAL BACKGROUND (10 min)
# ==============================================================================

#=
The Simeoni model describes tumor growth kinetics and drug effects:

1. UNPERTURBED GROWTH (Control):
   - Early stage: Exponential growth (unlimited nutrients)
   - Late stage: Linear growth (nutrient-limited)
   - Smooth transition between phases

   Equation:
   dw/dt = λ₀·w / [1 + (λ₀/λ₁·w)^ψ]^(1/ψ)

   where:
   - λ₀ = exponential growth rate (day⁻¹)
   - λ₁ = linear growth rate (g/day)
   - w = tumor weight (g)
   - ψ = smoothing parameter (~20)

2. DRUG-PERTURBED GROWTH (Treated):
   - Drug damages proliferating cells
   - Damaged cells progress through transit compartments to death
   - 4 compartments: x₁ (proliferating), x₂, x₃, x₄ (damaged)

   Equations:
   dx₁/dt = growth - k₂·c(t)·x₁
   dx₂/dt = k₂·c(t)·x₁ - k₁·x₂
   dx₃/dt = k₁·(x₂ - x₃)
   dx₄/dt = k₁·(x₃ - x₄)

   where:
   - k₂ = drug potency (ng⁻¹·mL·day⁻¹)
   - k₁ = transit rate constant (day⁻¹)
   - c(t) = plasma drug concentration
=#

# ==============================================================================
# PART 2: BUILDING THE PK MODEL (15 min)
# ==============================================================================

#=
We'll build the model in steps, explaining each @block.

STEP 1: Define the model skeleton
=#

# This is the simplest possible model - just the structure
model_skeleton = @model begin
    @param begin
        # Parameters go here
    end
    @dynamics begin
        # ODEs go here
    end
    @derived begin
        # Observations go here
    end
end

#=
STEP 2: Add PK parameters

The @param block defines:
- Population (typical) parameters with `∈ RealDomain`
- Inter-individual variability (IIV) with `∈ PDiagDomain`
- Residual error parameters
=#

# 2-compartment PK model
simeoni_pk_model = @model begin
    @param begin
        # Typical values (population parameters) - micro-rate constants
        tvk_el ∈ RealDomain(lower=0.0)     # Elimination rate constant (day⁻¹)
        tvk12  ∈ RealDomain(lower=0.0)     # Central→Peripheral rate constant (day⁻¹)
        tvk21  ∈ RealDomain(lower=0.0)     # Peripheral→Central rate constant (day⁻¹)
        tvVc   ∈ RealDomain(lower=0.0)     # Central volume (L)

        # Inter-individual variability (variance on log scale)
        # PDiagDomain creates a diagonal covariance matrix
        Ω ∈ PDiagDomain(2)  # Variance for k_el, Vc

        # Residual error (proportional)
        σ_prop ∈ RealDomain(lower=0.0)
    end

    #=
    STEP 3: Add random effects

    The @random block specifies the distribution of random effects (η).
    These capture inter-individual variability.
    =#
    @random begin
        η ~ MvNormal(Ω)  # η is a vector with covariance Ω
    end

    #=
    STEP 4: Transform parameters to individual values

    The @pre block computes individual parameters from population values and random effects.
    Using exp(η) gives log-normal distribution (always positive).
    =#
    @pre begin
        k_el = tvk_el * exp(η[1])  # Individual elimination rate
        k12  = tvk12               # Fixed (no IIV)
        k21  = tvk21               # Fixed (no IIV)
        Vc   = tvVc * exp(η[2])    # Individual central volume
    end

    #=
    STEP 5: Define dynamics (ODEs)

    2-compartment model with micro-rate constants:
    - k_el: elimination rate constant
    - k12: central→peripheral rate constant
    - k21: peripheral→central rate constant
    =#
    @dynamics begin
        Central'    = -k_el * Central - k12 * Central + k21 * Peripheral
        Peripheral' =                   k12 * Central - k21 * Peripheral
    end

    #=
    STEP 6: Define derived quantities and observations

    The @derived block:
    - Calculates quantities from ODE solutions
    - Defines observation distributions for fitting
    - := for deterministic variables
    - ~ for stochastic observations
    =#
    @derived begin
        cp := @. Central / Vc  # Concentration = Amount / Volume
        dv ~ @. Normal(cp, abs(cp) * σ_prop + 1e-10)  # Proportional error
    end
end

# Parameters for PK model (matching MATLAB SimBiology estimated values)
pk_params = (
    tvk_el = 13.5,                      # k_el: Elimination rate constant (day⁻¹)
    tvk12 = 0.26,                       # k12: Central→Peripheral rate (day⁻¹)
    tvk21 = 2.0,                        # k21: Peripheral→Central rate (day⁻¹)
    tvVc = 0.08,                        # Vc: Central volume (L)
    Ω = Diagonal([0.04, 0.04]),         # IIV for k_el, Vc
    σ_prop = 0.1                        # Proportional error
)

println("PK model created successfully!")

# ==============================================================================
# PART 3: BUILDING THE FULL PK-PD MODEL (35 min)
# ==============================================================================

#=
Now we'll extend the PK model to include the Simeoni tumor dynamics.

KEY ADDITIONS:
1. PD parameters (λ₀, λ₁, k₁, k₂, w₀)
2. Initial conditions for tumor compartments
3. @vars block for intermediate calculations
4. Full ODE system for PK + PD
5. @observed block for GSA endpoints
=#

simeoni_pkpd_model = @model begin
    @param begin
        # === PK Parameters (micro-rate constants) ===
        tvk_el ∈ RealDomain(lower=0.0)     # Elimination rate constant (day⁻¹)
        tvk12  ∈ RealDomain(lower=0.0)     # Central→Peripheral rate constant (day⁻¹)
        tvk21  ∈ RealDomain(lower=0.0)     # Peripheral→Central rate constant (day⁻¹)
        tvVc   ∈ RealDomain(lower=0.0)     # Central volume (L)

        # === PD Parameters (Simeoni TGI) ===
        tvlambda0 ∈ RealDomain(lower=0.0)   # Exponential growth rate (day⁻¹)
        tvlambda1 ∈ RealDomain(lower=0.0)   # Linear growth rate (g/day)
        tvk1      ∈ RealDomain(lower=0.0)   # Transit rate constant (day⁻¹)
        tvk2      ∈ RealDomain(lower=0.0)   # Drug potency (ng⁻¹·mL·day⁻¹)
        tvw0      ∈ RealDomain(lower=0.0)   # Initial tumor weight (g)
        psi       ∈ RealDomain(lower=1.0)   # Smoothing parameter (FIXED)

        # === Inter-Individual Variability ===
        Ω_pk ∈ PDiagDomain(2)         # Variance for CL, Vc
        Ω_pd ∈ PDiagDomain(3)         # Variance for λ₀, k₁, k₂

        # === Residual Error ===
        σ_pk ∈ RealDomain(lower=0.0)        # PK proportional error
        σ_pd ∈ RealDomain(lower=0.0)        # PD proportional error
    end

    @random begin
        η_pk ~ MvNormal(Ω_pk)  # PK random effects
        η_pd ~ MvNormal(Ω_pd)  # PD random effects
    end

    @pre begin
        # === Individual PK Parameters ===
        k_el = tvk_el * exp(η_pk[1])
        k12  = tvk12
        k21  = tvk21
        Vc   = tvVc * exp(η_pk[2])

        # === Individual PD Parameters ===
        lambda0 = tvlambda0 * exp(η_pd[1])
        lambda1 = tvlambda1                   # Fixed (not estimated with IIV)
        k1      = tvk1 * exp(η_pd[2])
        k2      = tvk2 * exp(η_pd[3])
        w0      = tvw0
    end

    #=
    STEP: Define initial conditions

    The @init block sets starting values for state variables.
    - PK: Start with no drug
    - PD: All tumor in proliferating compartment (x1)
    =#
    @init begin
        Central    = 0.0
        Peripheral = 0.0
        x1 = w0       # All tumor initially proliferating
        x2 = 0.0      # No damaged cells
        x3 = 0.0
        x4 = 0.0
    end

    #=
    STEP: Define intermediate variables

    The @vars block computes values used in ODEs.
    These are calculated at EACH TIME STEP during integration.

    IMPORTANT: Intermediate calculations should go in @vars, NOT @dynamics!
    =#
    @vars begin
        # Drug concentration (ng/mL) - safe division to avoid NaN
        cp = (Central / max(Vc, 1e-10)) * 1000

        # Total tumor weight (sum of all compartments)
        tumor = x1 + x2 + x3 + x4

        # Simeoni growth term with exponential→linear transition
        # When w is small: growth ≈ λ₀·x₁ (exponential)
        # When w is large: growth ≈ λ₁ (linear)
        growth = lambda0 * x1 / (1.0 + (lambda0 / lambda1 * tumor)^psi)^(1.0 / psi)
    end

    #=
    STEP: Define the ODEs

    The @dynamics block specifies the differential equations.
    Variables from @vars can be used here.
    =#
    @dynamics begin
        # === PK: 2-Compartment Model (micro-rate constants) ===
        Central'    = -k_el * Central - k12 * Central + k21 * Peripheral
        Peripheral' = k12 * Central - k21 * Peripheral

        # === PD: Simeoni TGI Transit Compartment Model ===
        # x₁: Proliferating (drug-sensitive) cells
        x1' = growth - k2 * cp * x1

        # x₂, x₃, x₄: Damaged cells progressing through transit
        x2' = k2 * cp * x1 - k1 * x2
        x3' = k1 * (x2 - x3)
        x4' = k1 * (x3 - x4)
    end

    @derived begin
        # Observations with proportional error
        dv_pk ~ @. Normal(cp, abs(cp) * σ_pk + 1e-10)
        dv_pd ~ @. Normal(tumor, abs(tumor) * σ_pd + 1e-10)

        # Explicitly capture growth rate for diagnostics
        growth_rate := @. growth
    end

    #=
    STEP: Define endpoints for GSA

    The @observed block provides scalar endpoints that can be used
    for Global Sensitivity Analysis (Session 5).

    NOTE: Commented out for basic simulations as scalar outputs
    cause DataFrame conversion issues. Uncomment for GSA analysis.
    =#
    #=
    @observed begin
        final_tumor = @. last(tumor)      # Tumor at end of simulation
        max_tumor   = @. maximum(tumor)  # Maximum tumor during simulation
    end
    =#
end

println("Full PK-PD model created successfully!")

# ==============================================================================
# PART 4: QUICK TEST - SIMULATE THE MODEL
# ==============================================================================

#=
Let's verify our model works by running a quick simulation.
=#

# Create subjects for all dose levels
# Body weight: 20 g = 0.020 kg
# Doses: Control (0), 45 mg/kg (0.9 mg), 60 mg/kg (1.2 mg)
mouse_weight_kg = 0.020  # 25g mouse

# Time grid with hourly sampling day 13-14 for PK visualization
time_grid = sort(unique(vcat(0.0:0.5:50.0, 13.0:(1/24):14.0)))

# Create dosage regimens for each dose level
dr_control = DosageRegimen(0.0, time = 13.0, cmt = 1)   # Control: no drug
dr_45mg = DosageRegimen(0.9, time = 13.0, cmt = 1)      # 45 mg/kg
dr_60mg = DosageRegimen(1.2, time = 13.0, cmt = 1)      # 60 mg/kg

# Create subjects for each dose level
subject_control = Subject(id = 1, events = dr_control, time = time_grid)
subject_45mg = Subject(id = 2, events = dr_45mg, time = time_grid)
subject_60mg = Subject(id = 3, events = dr_60mg, time = time_grid)

# Define parameters for simulation
# Parameters from Simeoni et al. (2004) - converted to consistent units
# PK rates converted from 1/hour to 1/day (×24), Volume is per kg body weight
test_params = (
    # PK Parameters (matching MATLAB SimBiology estimated values)
    tvk_el = 13.5,          # k_el: Elimination rate constant (day⁻¹)
    tvk12 = 0.26,           # k12: Central→Peripheral rate (day⁻¹)
    tvk21 = 2.0,            # k21: Peripheral→Central rate (day⁻¹)
    tvVc = 0.08,            # Vc: Central volume (L)
    # PD Parameters (Simeoni TGI)
    tvlambda0 = 0.146,     # λ₀: Exponential growth rate (day⁻¹)
    tvlambda1 = 0.334,     # λ₁: Linear growth rate (g/day)
    tvk1 = 0.469,          # K₁: Transit rate constant (day⁻¹)
    tvk2 = 8.0e-4,         # K₂: Killing rate (mL/ng/day) - adjusted to match MATLAB
    tvw0 = 0.085,          # w₀: Initial tumor weight (g)
    psi = 20.0,            # Ψ: Growth switch (dimensionless)
    # IIV - SET TO NEAR-ZERO for simulation qualification
    # (ensures all subjects use identical typical values)
    Ω_pk = Diagonal([1e-10, 1e-10]),
    Ω_pd = Diagonal([1e-10, 1e-10, 1e-10]),
    σ_pk = 0.1,
    σ_pd = 0.1
)

# Run simulations for all dose levels
println("\nRunning simulations for all dose levels...")
println("NOTE: IIV variances set to near-zero to ensure all subjects use identical typical values")

sim_control = simobs(simeoni_pkpd_model, subject_control, test_params)
sim_control_df = DataFrame(sim_control)
@rtransform!(sim_control_df, :dose_group = "Control")
println("Control simulation complete!")

sim_45mg = simobs(simeoni_pkpd_model, subject_45mg, test_params)
sim_45mg_df = DataFrame(sim_45mg)
@rtransform!(sim_45mg_df, :dose_group = "45 mg/kg")
println("45 mg/kg simulation complete!")

sim_60mg = simobs(simeoni_pkpd_model, subject_60mg, test_params)
sim_60mg_df = DataFrame(sim_60mg)
@rtransform!(sim_60mg_df, :dose_group = "60 mg/kg")
println("60 mg/kg simulation complete!")

# ==============================================================================
# VISUALIZE THE SIMULATION RESULTS WITH OBSERVED DATA
# ==============================================================================

println("\nCreating visualization...")

# --- Read observed data ---
pk_obs = CSV.read("data/PKdata.csv", DataFrame)
pd_control_obs = CSV.read("data/PDdata_control.csv", DataFrame)
pd_alldoses_obs = CSV.read("data/PDdata_AllDoses.csv", DataFrame)

# =============================================================================
# PK PLOT: 45 mg/kg simulation vs observed (Day 13-15)
# =============================================================================
pk_sim_df = @chain sim_45mg_df begin
    @select(:time, :cp)
    @rsubset(:time >= 13.0 && :time <= 15.0)
    @rtransform(:type = "Simulation PK")
end

pk_obs_df = @chain pk_obs begin
    @select(:time = :Time_day_13, :cp = :Conc_ng_mL)
    @rsubset(!ismissing(:cp))
    @rtransform(:type = "Data PK")
end

plt_pk_sim = data(pk_sim_df) *
    mapping(:time => "Time (days)", :cp => "Concentration (ng/mL)") *
    visual(Lines, linewidth = 2, color = :black, label = "Simulation PK")

plt_pk_obs = data(pk_obs_df) *
    mapping(:time => "Time (days)", :cp => "Concentration (ng/mL)") *
    visual(Scatter, markersize = 10, color = :transparent, marker = :circle, strokewidth = 2, strokecolor = :black, label = "Data PK")

fig_pk = draw(plt_pk_sim + plt_pk_obs;
    figure = (size = (550, 450),),
    axis = (title = "CPT11 PK simulated based on Simeoni et al.\nparameters, 45 mg/kg",
            yscale = log10,
            xlabel = "Time (days)",
            ylabel = "Concentration (ng/mL)",
            xgridvisible = true,
            ygridvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(5),
            yminorticks = IntervalsBetween(9)),
            legend = (; show = false)
)

# Add manual legend for PK plot (inside plot, top right)
Legend(fig_pk.figure[1, 1],
    [MarkerElement(marker = :circle, color = :transparent, strokecolor = :black, strokewidth = 2, markersize = 10),
     LineElement(color = :black, linewidth = 2)],
    ["Data PK", "Simulation PK"],
    framevisible = true,
    padding = (5, 5, 5, 5),
    halign = :right,
    valign = :top,
    margin = (10, 10, 10, 10),
    tellwidth = false,
    tellheight = false
)
fig_pk
# =============================================================================
# TUMOR PLOT: All dose levels - Control, 45 mg/kg, 60 mg/kg
# =============================================================================

# --- Prepare simulation data for all doses ---
tumor_sim_control = @chain sim_control_df begin
    @select(:time, :tumor, :dose_group)
end

tumor_sim_45mg = @chain sim_45mg_df begin
    @select(:time, :tumor, :dose_group)
end

tumor_sim_60mg = @chain sim_60mg_df begin
    @select(:time, :tumor, :dose_group)
end

# --- Prepare observed data ---
# Control observed data
tumor_obs_control = @chain pd_control_obs begin
    rename!("Time (days)" => "time", "TumorWeight (g)" => "tumor")
    @rsubset(!ismissing(:tumor))
    @rtransform(:dose_group = "Control")
end

# 45 mg/kg observed data (Group 2 in AllDoses)
tumor_obs_45mg = @chain pd_alldoses_obs begin
    @rsubset(:Group == 2)
    rename!("Time (days)" => "time", "TumorWeight (g)" => "tumor")
    @rsubset(!ismissing(:tumor))
    @rtransform(:dose_group = "45 mg/kg")
end

# 60 mg/kg observed data (Group 3 in AllDoses)
tumor_obs_60mg = @chain pd_alldoses_obs begin
    @rsubset(:Group == 3)
    rename!("Time (days)" => "time", "TumorWeight (g)" => "tumor")
    @rsubset(!ismissing(:tumor))
    @rtransform(:dose_group = "60 mg/kg")
end

# --- Create layered tumor plot matching PowerPoint style ---
# Control: black
plt_tumor_sim_ctrl = data(tumor_sim_control) *
    mapping(:time, :tumor) *
    visual(Lines, linewidth = 2, color = :black)

plt_tumor_obs_ctrl = data(tumor_obs_control) *
    mapping(:time, :tumor) *
    visual(Scatter, markersize = 10, color = :transparent, marker = :circle, strokewidth = 2, strokecolor = :black)

# 45 mg/kg: blue
plt_tumor_sim_45 = data(tumor_sim_45mg) *
    mapping(:time, :tumor) *
    visual(Lines, linewidth = 2, color = :blue)

plt_tumor_obs_45 = data(tumor_obs_45mg) *
    mapping(:time, :tumor) *
    visual(Scatter, markersize = 10, color = :transparent, marker = :circle, strokewidth = 2, strokecolor = :blue)

# 60 mg/kg: red
plt_tumor_sim_60 = data(tumor_sim_60mg) *
    mapping(:time, :tumor) *
    visual(Lines, linewidth = 2, color = :red)

plt_tumor_obs_60 = data(tumor_obs_60mg) *
    mapping(:time, :tumor) *
    visual(Scatter, markersize = 10, color = :transparent, marker = :circle, strokewidth = 2, strokecolor = :red)

# Combine all layers
fig_tumor = draw(
    plt_tumor_sim_ctrl + plt_tumor_obs_ctrl +
    plt_tumor_sim_45 + plt_tumor_obs_45 +
    plt_tumor_sim_60 + plt_tumor_obs_60;
    figure = (size = (700, 500),),
    axis = (title = "Tumor growth inhibition (Control and Treatment arm)",
            xlabel = "Time (days)",
            ylabel = "Tumor weight (g)",
            limits = (0, 50, 0, 12),
            xgridvisible = true,
            ygridvisible = true,
            xminorgridvisible = true,
            yminorgridvisible = true,
            xminorticks = IntervalsBetween(5),
            yminorticks = IntervalsBetween(4),
            xticks = 0:10:50,
            yticks = 0:2:12),
            legend = (; show = false)
)

# Add manual legend for tumor plot (inside plot, top left - matching MATLAB style)
Legend(fig_tumor.figure[1, 1],
    [MarkerElement(marker = :circle, color = :transparent, strokecolor = :black, strokewidth = 2, markersize = 10),
     LineElement(color = :black, linewidth = 2),
     MarkerElement(marker = :circle, color = :transparent, strokecolor = :blue, strokewidth = 2, markersize = 10),
     LineElement(color = :blue, linewidth = 2),
     MarkerElement(marker = :circle, color = :transparent, strokecolor = :red, strokewidth = 2, markersize = 10),
     LineElement(color = :red, linewidth = 2)],
    ["Data for Control", "Simulation for Control",
     "Data for 45 mg/kg", "Simulation for 45 mg/kg",
     "Data for 60 mg/kg", "Simulation for 60 mg/kg"],
    framevisible = true,
    padding = (5, 5, 5, 5),
    halign = :left,
    valign = :top,
    margin = (10, 10, 10, 10),
    tellwidth = false,
    tellheight = false
)
fig_tumor
# =============================================================================
# Save figures
# =============================================================================
mkpath("outputs")
save("outputs/01_pk_simulation.png", fig_pk)
save("outputs/01_tumor_alldoses.png", fig_tumor)
println("Figures saved: outputs/01_pk_simulation.png, outputs/01_tumor_alldoses.png")

# ==============================================================================
# SUMMARY
# ==============================================================================

println("\n" * "="^60)
println("SESSION 1 COMPLETE: Pumas Model Building")
println("="^60)
println("""

Key Takeaways:
1. @param: Define population parameters, IIV, and error terms
2. @random: Specify random effect distributions
3. @pre: Transform parameters from population to individual level
4. @init: Set initial conditions for ODEs
5. @vars: Compute intermediate variables (used in @dynamics)
6. @dynamics: Define the differential equations
7. @derived: Calculate observed quantities and observation distributions
8. @observed: Define endpoints for GSA

Models Created:
- simeoni_pk_model: 2-compartment PK only
- simeoni_pkpd_model: Full PK-PD with Simeoni TGI dynamics

Next: Session 2 - Parameter Estimation
""")
