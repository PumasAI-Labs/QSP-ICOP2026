#=
================================================================================
PKPD Parameter Estimation for Simeoni Model
================================================================================

This script estimates PD parameters using the NaivePooled approach:
- Uses PK parameters fixed from PKestimation.jl
- Estimates Simeoni tumor growth inhibition (TGI) parameters
- Data from Control, 45 mg/kg, and 60 mg/kg dose groups

Reference: Simeoni et al. (2004) Cancer Research 64:1094-1101
=#

# ==============================================================================
# SETUP
# ==============================================================================

using Pumas
using AlgebraOfGraphics, CairoMakie
using DataFramesMeta
using PumasUtilities
using CSV

set_aog_theme!()

# ==============================================================================
# PK PARAMETERS (FROM PKestimation.jl)
# ==============================================================================

#=
These values should be updated after running PKestimation.jl
The values below are from the MATLAB SimBiology model as initial reference
=#
pk_estimates = (tvk_el = 13.686221601894077,
 tvk12 = 0.27060837593286097,
 tvk21 = 1.4879817703974072,
 tvVc = 0.07789461388217106,
 σ_prop = 0.012786649461480986,)

# ==============================================================================
# DATA PREPARATION
# ==============================================================================

println("\n" * "="^60)
println("LOADING PD DATA")
println("="^60)

# Read PD data files
pd_control_raw = CSV.read("data/PDdata_control.csv", DataFrame)
pd_alldoses_raw = CSV.read("data/PDdata_AllDoses.csv", DataFrame)

# --- Prepare Control data (ID = 1, no drug) ---
pd_control = @chain pd_control_raw begin
    rename("Time (days)" => "time", "TumorWeight (g)" => "tumor_obs")
    @rtransform(:id = "Control", :evid = 0, :cmt = 2, :amt = 0.0)  # CMT=2 for PD observations
    @rtransform :evid = :time == 13.0 ? 1 : :evid
    @rtransform :cmt = :time == 13.0 ? 1 : missing
    @rtransform :amt = :time == 13.0 ? 0.0 : missing
    @select(:id, :time, :tumor_obs, :evid, :cmt, :amt, :Group)
end

# --- Prepare 45 mg/kg data (ID = 2) ---
pd_45mg = @chain pd_alldoses_raw begin
    @rsubset(:Group == 2)
    rename("Time (days)" => "time", "TumorWeight (g)" => "tumor_obs")
    @rtransform(:id = "45mg", :evid = 0, :cmt = 2, :amt = 0.0)
    @rtransform :evid = :time == 13.0 ? 1 : :evid
    @rtransform :cmt = :time == 13.0 ? 1 : missing
    @rtransform :amt = :time == 13.0 ? (:Dose)*10 : missing
    @select(:id, :time, :tumor_obs, :evid, :cmt, :amt, :Group)
end

# --- Prepare 60 mg/kg data (ID = 3) ---
pd_60mg = @chain pd_alldoses_raw begin
    @rsubset(:Group == 3)
    rename("Time (days)" => "time", "TumorWeight (g)" => "tumor_obs")
    @rtransform(:id = "60mg", :evid = 0, :cmt = 2, :amt = 0.0)
    @rtransform :evid = :time == 13.0 ? 1 : :evid
    @rtransform :cmt = :time == 13.0 ? 1 : missing
    @rtransform :amt = :time == 13.0 ? :Dose : missing
    @select(:id, :time, :tumor_obs, :evid, :cmt, :amt, :Group)
end

# Combine all data
pd_data_full = vcat(
    pd_control,
    pd_45mg,
    pd_60mg
)
sort!(pd_data_full, [:id, :time])

# Create Pumas population
pd_population = read_pumas(
    pd_data_full,
    id = :id,
    time = :time,
    observations = [:tumor_obs],
    evid = :evid,
    cmt = :cmt,
    amt = :amt
)

println("\nPopulation created with $(length(pd_population)) subject(s)")

# ==============================================================================
# PKPD MODEL DEFINITION (No Random Effects for NaivePooled)
# ==============================================================================

pkpd_model = @model begin
    @param begin
        # === PK Parameters (will be FIXED) ===
        tvk_el ∈ RealDomain(lower=0.0)     # Elimination rate constant (day⁻¹)
        tvk12  ∈ RealDomain(lower=0.0)     # Central→Peripheral rate constant (day⁻¹)
        tvk21  ∈ RealDomain(lower=0.0)     # Peripheral→Central rate constant (day⁻¹)
        tvVc   ∈ RealDomain(lower=0.0)     # Central volume (L)

        # === PD Parameters (to be ESTIMATED) ===
        tvlambda0 ∈ RealDomain(lower=0.0)   # Exponential growth rate (day⁻¹)
        tvlambda1 ∈ RealDomain(lower=0.0)   # Linear growth rate (g/day)
        tvk1      ∈ RealDomain(lower=0.0)   # Transit rate constant (day⁻¹)
        tvk2      ∈ RealDomain(lower=0.0)   # Drug potency (ng⁻¹·mL·day⁻¹)
        tvw0      ∈ RealDomain(lower=0.0)   # Initial tumor weight (g)

        # psi is FIXED at 20.0 (smoothing parameter)
        psi ∈ RealDomain(lower=1.0)

        # === Residual Error ===
        σ_pd ∈ RealDomain(lower=0.0)        # PD proportional error
    end

    @pre begin
        # === PK Parameters (fixed) ===
        k_el = tvk_el
        k12  = tvk12
        k21  = tvk21
        Vc   = tvVc

        # === PD Parameters (estimated) ===
        lambda0 = tvlambda0
        lambda1 = tvlambda1
        k1      = tvk1
        k2      = tvk2
        w0      = tvw0
    end

    @init begin
        # PK: Start with no drug (dose will be added by events)
        Central    = 0.0
        Peripheral = 0.0
        # PD: All tumor initially in proliferating compartment
        x1 = w0       # Proliferating cells
        x2 = 0.0      # Damaged cells (transit 1)
        x3 = 0.0      # Damaged cells (transit 2)
        x4 = 0.0      # Damaged cells (transit 3)
    end

    @vars begin
        # Drug concentration (ng/mL) - safe division
        cp = (Central / max(Vc, 1e-10)) * 1000

        # Total tumor weight (sum of all compartments)
        tumor = x1 + x2 + x3 + x4

        # Simeoni growth term with exponential→linear transition
        growth = lambda0 * x1 / (1.0 + (lambda0 / lambda1 * tumor)^psi)^(1.0 / psi)
    end

    @dynamics begin
        # === PK: 2-Compartment Model ===
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
        # Tumor weight observation with proportional error
        tumor_obs ~ @. Normal(tumor, abs(tumor) * σ_pd + 1e-10)
    end
end

println("\nPKPD model defined with Simeoni TGI dynamics")

# ==============================================================================
# INITIAL PARAMETER VALUES
# ==============================================================================

# Initial values from Simeoni et al. (2004) / MATLAB SimBiology
init_params = (
    # PK parameters (fixed)
    tvk_el = pk_estimates.tvk_el,
    tvk12  = pk_estimates.tvk12,
    tvk21  = pk_estimates.tvk21,
    tvVc   = pk_estimates.tvVc,

    # PD parameters (to estimate)
    tvlambda0 = 0.146,     # λ₀: Exponential growth rate (day⁻¹)
    tvlambda1 = 0.334,     # λ₁: Linear growth rate (g/day)
    tvk1      = 0.469,     # k₁: Transit rate constant (day⁻¹)
    tvk2      = 8.0e-4,    # k₂: Killing rate (mL/ng/day)
    tvw0      = 0.085,     # w₀: Initial tumor weight (g)
    psi       = 20.0,      # Ψ: Growth switch (FIXED)

    # Residual error
    σ_pd = 0.15            # Proportional error
)


# ==============================================================================
# MODEL FITTING (with Fixed PK Parameters)
# ==============================================================================

println("\n" * "="^60)
println("FITTING PKPD MODEL")
println("="^60)

println("\nFitting with fixed PK parameters...")
println("Parameters fixed: tvk_el, tvk12, tvk21, tvVc, psi")

fit_pkpd = fit(
    pkpd_model,
    pd_population,
    init_params,
    NaivePooled();
    constantcoef = (
        :tvk_el,
        :tvk12 ,
        :tvk21 ,
        :tvVc  ,
        :psi 
    )
)

println("PKPD model fit complete!")

# ==============================================================================
# PARAMETER ESTIMATES
# ==============================================================================

println("\n" * "="^60)
println("ESTIMATED PD PARAMETERS")
println("="^60)

# Get coefficient estimates
coeftable(fit_pkpd) |> simple_table
# Parameter inference (confidence intervals)
println("\n--- Parameter Inference ---")
infer_result = infer(fit_pkpd)
coeftable(infer_result) |> simple_table

# ==============================================================================
# MODEL DIAGNOSTICS
# ==============================================================================

println("\n" * "="^60)
println("MODEL DIAGNOSTICS")
println("="^60)

# Inspect results
inspect_result = inspect(fit_pkpd)
inspect_df = DataFrame(inspect_result)

# Model metrics
println("\n--- Model Fit Metrics ---")
metrics = metrics_table(fit_pkpd)
simple_table(metrics)
# ==============================================================================
# DIAGNOSTIC PLOTS
# ==============================================================================

println("\nCreating diagnostic plots...")

# Goodness-of-fit plots
observations_vs_predictions(inspect_result)

# Subject fits (individual prediction plots for all 3 dose groups)
subj_plots = subject_fits(inspect_result, 
                        separate = true,
                        ipred_linewidth = 2, 
                        markersize = 8,
            axis = (;
                    xlabel = "Time (day)",
                    ylabel = "Tumor Weight (g)",
                    limits = (0, 50, 0, 12),
                    xgridvisible = true,
                    ygridvisible = true,
                    xminorgridvisible = true,
                    yminorgridvisible = true,
                    xminorticks = IntervalsBetween(5),
                    yminorticks = IntervalsBetween(4),
                    xticks = 0:10:50,
                    yticks = 0:2:12),
            legend = (; position = :bottom)
)

# ==============================================================================
# COMPARISON WITH LITERATURE VALUES
# ==============================================================================


println("\n" * "="^60)
println("PKPD ESTIMATION COMPLETE")
println("="^60)

