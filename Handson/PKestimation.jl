#=
================================================================================
PK Parameter Estimation for Simeoni Model
================================================================================

This script estimates PK parameters using the NaivePooled approach:
- Compares additive vs proportional error models
- Uses data from PKdata.csv (45 mg/kg dose, 7 observations)
- Outputs parameter estimates for use in PKPDestimation.jl

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
using SummaryTables
set_aog_theme!()

# ==============================================================================
# DATA PREPARATION
# ==============================================================================

println("Loading PK data...")

# Read PK data
pk_raw = CSV.read("data/PKdata.csv", DataFrame)

# Prepare data for Pumas
# - Time_day: time relative to dose (starts at 0)
# - Conc_ng_mL: concentration observations
# - Dose: 45 mg/kg = 0.9 mg (mouse weight 0.025 kg × 45 mg/kg × 1000 μg/mg / 1000 = 1.125 mg)
#   However, based on MATLAB model, dose is 0.9 mg
pk_data = @chain pk_raw begin
    rename(:Time_day => :time, :Conc_ng_mL => :dv, :Dose_45_mg_kg => :amt)
    @transform :id = 1
    @rtransform :evid = :time == 0.0 ? 1 : 0
    @rtransform :cmt = :time == 0.0 ? 1 : missing
    @transform :id = 1
    select(:id, :time, :dv, :evid, :cmt , :amt, :Group)
end

println("PK data prepared:")
first(pk_data, 10) 
# Create Pumas population
pk_population = read_pumas(
    pk_data,
    id = :id,
    time = :time,
    observations = [:dv],
    evid = :evid,
    cmt = :cmt,
    amt = :amt
)

println("\nPopulation created with $(length(pk_population)) subject(s)")
println("Number of observations: $(nobs(pk_population))")

# ==============================================================================
# MODEL DEFINITIONS (No Random Effects for NaivePooled)
# ==============================================================================

#=
Model 1: Additive Error
DV ~ Normal(cp, σ_add)
- Constant variance regardless of concentration
- Better for data with similar measurement uncertainty across range
=#
pk_model_additive = @model begin
    @param begin
        # PK Parameters (micro-rate constants)
        tvk_el ∈ RealDomain(lower=0.0)     # Elimination rate constant (day⁻¹)
        tvk12  ∈ RealDomain(lower=0.0)     # Central→Peripheral rate constant (day⁻¹)
        tvk21  ∈ RealDomain(lower=0.0)     # Peripheral→Central rate constant (day⁻¹)
        tvVc   ∈ RealDomain(lower=0.0)     # Central volume (L)

        # Residual error (additive)
        σ_add ∈ RealDomain(lower=0.0)
    end

    @pre begin
        k_el = tvk_el
        k12  = tvk12
        k21  = tvk21
        Vc   = tvVc
    end

    @dynamics begin
        Central'    = -k_el * Central - k12 * Central + k21 * Peripheral
        Peripheral' = k12 * Central - k21 * Peripheral
    end

    @derived begin
        cp := @. Central / Vc * 1000  # Convert to ng/mL
        dv ~ @. Normal(cp, σ_add)     # Additive error
    end
end

#=
Model 2: Proportional Error
DV ~ Normal(cp, |cp| × σ_prop)
- Variance proportional to prediction
- Common for concentration data spanning several orders of magnitude
=#
pk_model_proportional = @model begin
    @param begin
        # PK Parameters (micro-rate constants)
        tvk_el ∈ RealDomain(lower=0.0)     # Elimination rate constant (day⁻¹)
        tvk12  ∈ RealDomain(lower=0.0)     # Central→Peripheral rate constant (day⁻¹)
        tvk21  ∈ RealDomain(lower=0.0)     # Peripheral→Central rate constant (day⁻¹)
        tvVc   ∈ RealDomain(lower=0.0)     # Central volume (L)

        # Residual error (proportional)
        σ_prop ∈ RealDomain(lower=0.0)
    end

    @pre begin
        k_el = tvk_el
        k12  = tvk12
        k21  = tvk21
        Vc   = tvVc
    end

    @dynamics begin
        Central'    = -k_el * Central - k12 * Central + k21 * Peripheral
        Peripheral' = k12 * Central - k21 * Peripheral
    end

    @derived begin
        cp := @. Central / Vc * 1000  # Convert to ng/mL
        dv ~ @. Normal(cp, abs(cp) * σ_prop + 1e-10)  # Proportional error
    end
end

println("\nModels defined:")
println("  1. pk_model_additive (additive error)")
println("  2. pk_model_proportional (proportional error)")

# ==============================================================================
# INITIAL PARAMETER ESTIMATES
# ==============================================================================

# Initial values based on Simeoni et al. (2004) / MATLAB SimBiology
init_params_add = (
    tvk_el = 13.5,      # Elimination rate constant (day⁻¹)
    tvk12  = 0.26,      # Central→Peripheral rate (day⁻¹)
    tvk21  = 2.0,       # Peripheral→Central rate (day⁻¹)
    tvVc   = 0.08,      # Central volume (L)
    σ_add  = 500.0      # Additive error (ng/mL)
)

init_params_prop = (
    tvk_el = 13.5,      # Elimination rate constant (day⁻¹)
    tvk12  = 0.26,      # Central→Peripheral rate (day⁻¹)
    tvk21  = 2.0,       # Peripheral→Central rate (day⁻¹)
    tvVc   = 0.08,      # Central volume (L)
    σ_prop = 0.2        # Proportional error (fraction)
)

# ==============================================================================
# MODEL FITTING
# ==============================================================================

println("\n" * "="^60)
println("FITTING MODELS")
println("="^60)

# Fit additive error model
println("\nFitting additive error model...")
fit_additive = fit(
    pk_model_additive,
    pk_population,
    init_params_add,
    NaivePooled()
)
println("Additive model fit complete!")

# Fit proportional error model
println("\nFitting proportional error model...")
fit_proportional = fit(
    pk_model_proportional,
    pk_population,
    init_params_prop,
    NaivePooled()
)
println("Proportional model fit complete!")

# ==============================================================================
# MODEL COMPARISON
# ==============================================================================

println("\n" * "="^60)
println("MODEL COMPARISON")
println("="^60)

# Compare models using metrics
metrics_comparison = outerjoin(
    rename(metrics_table(fit_additive),:Value => :Additive),
    rename(metrics_table(fit_proportional),:Value => :Proportional),
    on = :Metric,
)   
simple_table(metrics_comparison)

# ==============================================================================
# PARAMETER ESTIMATES & INFERENCE
# ==============================================================================

simple_table(
    compare_estimates(;
                Additive = fit_additive, 
                Proportional = fit_proportional)
)

# Parameter inference (confidence intervals)
println("\n--- Parameter Inference ---")
infer_result = infer(fit_proportional)

# ==============================================================================
# MODEL DIAGNOSTICS
# ==============================================================================

println("\n" * "="^60)
println("MODEL DIAGNOSTICS")
println("="^60)

# Inspect results (predictions and residuals)
inspect_result = inspect(fit_proportional)
inspect_df = DataFrame(inspect_result)

# ==============================================================================
# DIAGNOSTIC PLOTS
# ==============================================================================

println("\nCreating diagnostic plots...")

# Subject fits (individual prediction plots)
subj_plots = subject_fits(inspect_result,
            axis = (yscale = log10, ylabel = "Concentration (ng/mL)", 
            xlabel = "Time (day)",
            title = "PK Subject Fits - Proportional Error Model")
            
)

println("\n" * "="^60)
println("PK ESTIMATION COMPLETE")
println("="^60)
