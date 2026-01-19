#=
================================================================================
GLOBAL SENSITIVITY ANALYSIS (GSA) - Simeoni Model
================================================================================

This script performs Global Sensitivity Analysis on the Simeoni TGI model:
- Sobol variance-based sensitivity indices
- First-order (S1) and Total-order (ST) indices
- Explores entire parameter space (not just local perturbations)
- Captures parameter interactions
- 3 dose groups: Control, 45 mg/kg, 60 mg/kg

Reference:
- Simeoni et al. (2004) Cancer Research 64:1094-1101
- Saltelli et al. "Global Sensitivity Analysis: The Primer"
=#

# ==============================================================================
# PART 1: SETUP
# ==============================================================================

using Pumas
using AlgebraOfGraphics, CairoMakie
using DataFramesMeta
using PumasUtilities
using CSV
using GlobalSensitivity

set_aog_theme!()

println("="^60)
println("GLOBAL SENSITIVITY ANALYSIS - Simeoni Model")
println("="^60)

# ==============================================================================
# PART 2: GSA MODEL DEFINITION
# ==============================================================================

#=
Model with @observed block for scalar endpoints required by GSA.
Based on simeoni_pkpd_model but simplified (no random effects).
=#

gsa_model = @model begin
    @param begin
        # === PK Parameters (FIXED for GSA) ===
        tvk_el ∈ RealDomain(lower=0.0)     # Elimination rate constant (day⁻¹)
        tvk12  ∈ RealDomain(lower=0.0)     # Central→Peripheral rate constant (day⁻¹)
        tvk21  ∈ RealDomain(lower=0.0)     # Peripheral→Central rate constant (day⁻¹)
        tvVc   ∈ RealDomain(lower=0.0)     # Central volume (L)

        # === PD Parameters (for GSA) ===
        tvlambda0 ∈ RealDomain(lower=0.0)   # Exponential growth rate (day⁻¹)
        tvlambda1 ∈ RealDomain(lower=0.0)   # Linear growth rate (g/day)
        tvk1      ∈ RealDomain(lower=0.0)   # Transit rate constant (day⁻¹)
        tvk2      ∈ RealDomain(lower=0.0)   # Drug potency (ng⁻¹·mL·day⁻¹)
        tvw0      ∈ RealDomain(lower=0.0)   # Initial tumor weight (g)
        psi       ∈ RealDomain(lower=1.0)   # Smoothing parameter
    end

    @pre begin
        # Direct parameter mapping (no random effects)
        k_el    = tvk_el
        k12     = tvk12
        k21     = tvk21
        Vc      = tvVc
        lambda0 = tvlambda0
        lambda1 = tvlambda1
        k1      = tvk1
        k2      = tvk2
        w0      = tvw0
    end

    @init begin
        Central    = 0.0
        Peripheral = 0.0
        x1 = w0       # All tumor initially proliferating
        x2 = 0.0      # No damaged cells
        x3 = 0.0
        x4 = 0.0
    end

    @vars begin
        # Drug concentration (ng/mL)
        cp = (Central / max(Vc, 1e-10)) * 1000

        # Total tumor weight
        tumor = x1 + x2 + x3 + x4

        # Simeoni growth term
        growth = lambda0 * x1 / (1.0 + (lambda0 / lambda1 * tumor)^psi)^(1.0 / psi)
    end

    @dynamics begin
        # PK: 2-Compartment Model
        Central'    = -k_el * Central - k12 * Central + k21 * Peripheral
        Peripheral' = k12 * Central - k21 * Peripheral

        # PD: Simeoni TGI Transit Compartment Model
        x1' = growth - k2 * cp * x1
        x2' = k2 * cp * x1 - k1 * x2
        x3' = k1 * (x2 - x3)
        x4' = k1 * (x3 - x4)
    end

    @derived begin
        # Tumor weight output (for time-series)
        tumor_obs = @. tumor
    end

    # Scalar endpoints for GSA (required for Sobol analysis)
    @observed begin
        final_tumor = last(tumor_obs)
    end
end

println("\nGSA model defined with @observed block for scalar endpoints")

# ==============================================================================
# PART 3: DATA PREPARATION
# ==============================================================================

# Time configuration
time_req = 30.0  # Sensitivity endpoint (day 30)
time_grid = sort(unique([collect(0.0:0.5:50.0); time_req]))

# Dosage regimens (drug administered at day 13 when tumor reaches ~1g)
dr_control = DosageRegimen(0.0, time = 13.0, cmt = 1)   # Control: no drug
dr_45mg = DosageRegimen(0.9, time = 13.0, cmt = 1)      # 45 mg/kg
dr_60mg = DosageRegimen(1.2, time = 13.0, cmt = 1)      # 60 mg/kg

# Create subjects for each dose group
subject_control = Subject(id = "Control", events = dr_control, time = time_grid)
subject_45mg = Subject(id = "45mgkg", events = dr_45mg, time = time_grid)
subject_60mg = Subject(id = "60mgkg", events = dr_60mg, time = time_grid)

dose_groups = [
    (name = "Control", subject = subject_control),
    (name = "45 mg/kg", subject = subject_45mg),
    (name = "60 mg/kg", subject = subject_60mg)
]

println("\nSubjects created for 3 dose groups: Control, 45 mg/kg, 60 mg/kg")

# ==============================================================================
# PART 4: BASELINE PARAMETERS
# ==============================================================================

#=
Parameters from PKPDestimation.jl (estimated values):
- PK parameters: Fixed from PK estimation
- PD parameters: Estimated from PKPD fit
=#

baseline_params = (
    # PK Parameters (from PKestimation.jl)
    tvk_el = 13.686,
    tvk12 = 0.271,
    tvk21 = 1.488,
    tvVc = 0.078,

    # PD Parameters (from PKPDestimation.jl / Simeoni et al.)
    tvlambda0 = 0.12441,     # λ₀: Exponential growth rate (day⁻¹)
    tvlambda1 = 0.34511,     # λ₁: Linear growth rate (g/day)
    tvk1 = 0.81905,          # k₁: Transit rate constant (day⁻¹)
    tvk2 = 0.00074522,       # k₂: Drug potency (mL/ng/day)
    tvw0 = 0.085,            # w₀: Initial tumor weight (g)
    psi = 20.0               # ψ: Growth transition smoothing
)

# Parameters to analyze (matching MATLAB GSA_SimBiScript.m)
gsa_param_names = [:tvlambda0, :tvlambda1, :psi, :tvk1, :tvk2]

# Display names for plots (matching MATLAB style)
param_display_names = Dict(
    :tvlambda0 => "lambda_0",
    :tvlambda1 => "lambda_1",
    :psi => "phi",
    :tvk1 => "k1",
    :tvk2 => "k2"
)

println("\nBaseline parameters set")
println("Parameters for GSA: $(join([param_display_names[p] for p in gsa_param_names], ", "))")

# ==============================================================================
# PART 5: GSA PARAMETER RANGES
# ==============================================================================

#=
Define parameter ranges for GSA sampling.
Using ±50% bounds around baseline values for comparability with LSA.
The Pumas.gsa function requires:
- p_range_low: NamedTuple of lower bounds
- p_range_high: NamedTuple of upper bounds
=#

# Lower bounds (50% of baseline)
p_range_low = (
    tvlambda0 = baseline_params.tvlambda0 * 0.5,
    tvlambda1 = baseline_params.tvlambda1 * 0.5,
    psi       = baseline_params.psi * 0.5,
    tvk1      = baseline_params.tvk1 * 0.5,
    tvk2      = baseline_params.tvk2 * 0.5,
)

# Upper bounds (150% of baseline)
p_range_high = (
    tvlambda0 = baseline_params.tvlambda0 * 1.5,
    tvlambda1 = baseline_params.tvlambda1 * 1.5,
    psi       = baseline_params.psi * 1.5,
    tvk1      = baseline_params.tvk1 * 1.5,
    tvk2      = baseline_params.tvk2 * 1.5,
)

# Parameters to keep constant (PK + w0)
constant_params = (:tvk_el, :tvk12, :tvk21, :tvVc, :tvw0)

println("\nParameter ranges for GSA (±50% of baseline):")
for param in gsa_param_names
    println("  $(param_display_names[param]): [$(round(p_range_low[param], sigdigits=3)), $(round(p_range_high[param], sigdigits=3))]")
end

# ==============================================================================
# PART 6: RUN GSA (Sobol Analysis)
# ==============================================================================

println("\n" * "="^60)
println("RUNNING GLOBAL SENSITIVITY ANALYSIS")
println("="^60)
println("\nMethod: Sobol variance-based sensitivity indices")
println("Endpoint: Tumor weight at day $time_req")

# Number of samples for Sobol analysis
n_samples = 1000  # Increase for more accurate results

println("\nNumber of samples: $n_samples")
println("Bootstrap replicates for CI: 100")

# Storage for GSA results
gsa_results = DataFrame(
    dose_group = String[],
    parameter = String[],
    S1 = Float64[],        # First-order Sobol index
    ST = Float64[],        # Total-order Sobol index
    S1_CI_low = Float64[], # Confidence interval lower bound
    S1_CI_high = Float64[],
    ST_CI_low = Float64[],
    ST_CI_high = Float64[]
)

n_groups = length(dose_groups)
for (group_idx, group) in enumerate(dose_groups)
    println("\n" * "="^50)
    println("[Step $group_idx/$n_groups] GSA for $(group.name)")
    println("="^50)

    # Run Sobol sensitivity analysis
    println("  [1/3] Running Sobol analysis ($n_samples samples, 100 bootstrap replicates)...")
    println("        This may take a few minutes...")

    gsa_result = Pumas.gsa(
        gsa_model,
        group.subject,
        baseline_params,
        Sobol(nboot = 100),         # GSA method with bootstrap for CI
        [:final_tumor],             # Variables to analyze (from @observed)
        p_range_low,                # Lower bounds for varied parameters
        p_range_high;               # Upper bounds for varied parameters
        constantcoef = constant_params,
        samples = n_samples
    )
    println("        Done!")

    # SobolOutput structure:
    # - first_order: DataFrame with S1 values (columns: dv_name, tvlambda0, tvlambda1, ...)
    # - first_order_conf_int: DataFrame with S1 CIs (columns: dv_name, param Min CI, param Max CI, ...)
    # - total_order: DataFrame with ST values
    # - total_order_conf_int: DataFrame with ST CIs
    # Row 1 corresponds to :final_tumor

    println("  [2/3] Extracting Sobol indices from results...")

    # Extract DataFrames
    s1_df = gsa_result.first_order
    s1_ci_df = gsa_result.first_order_conf_int
    st_df = gsa_result.total_order
    st_ci_df = gsa_result.total_order_conf_int

    println("  [3/3] Processing $(length(gsa_param_names)) parameters...")

    # Store results - iterate over the parameters we're varying
    for (param_idx, param) in enumerate(gsa_param_names)
        display_name = param_display_names[param]
        param_str = String(param)

        # Get Sobol indices from DataFrames (row 1 = final_tumor)
        s1_val = s1_df[1, param]
        st_val = st_df[1, param]

        # Get confidence intervals from separate CI DataFrames
        s1_ci_low = s1_ci_df[1, Symbol(param_str * " Min CI")]
        s1_ci_high = s1_ci_df[1, Symbol(param_str * " Max CI")]
        st_ci_low = st_ci_df[1, Symbol(param_str * " Min CI")]
        st_ci_high = st_ci_df[1, Symbol(param_str * " Max CI")]

        push!(gsa_results, (
            dose_group = group.name,
            parameter = display_name,
            S1 = s1_val,
            ST = st_val,
            S1_CI_low = s1_ci_low,
            S1_CI_high = s1_ci_high,
            ST_CI_low = st_ci_low,
            ST_CI_high = st_ci_high
        ))

        println("        [$param_idx/$(length(gsa_param_names))] $(rpad(display_name, 10)): S1 = $(round(s1_val, digits=3)), ST = $(round(st_val, digits=3))")
    end

    println("\n  Completed $(group.name)!")
end

println("\n" * "="^60)
println("GSA COMPLETE")
println("="^60)

# ==============================================================================
# PART 7: GSA VISUALIZATION
# ==============================================================================

println("\nCreating GSA visualization...")

# Determine consistent parameter ordering across all panels
# Use average ST across all dose groups
param_avg_sensitivity = @chain gsa_results begin
    groupby(:parameter)
    @combine(:avg_ST = mean(:ST))
    sort(:avg_ST)  # Ascending order (least sensitive at top)
end
consistent_param_order = param_avg_sensitivity.parameter

# Create figure with 3 panels
fig = Figure(size = (1400, 600), fontsize = 12)

# Add overall title
Label(fig[0, 1:3], "Global Sensitivity Analysis (Sobol Indices)", fontsize = 18, font = :bold)

for (idx, group) in enumerate(dose_groups)
    # Filter and reorder data
    group_data = @chain gsa_results begin
        @rsubset(:dose_group == group.name)
    end
    group_data = group_data[indexin(consistent_param_order, group_data.parameter), :]

    # Y positions
    y_pos = collect(1:nrow(group_data))
    bar_width = 0.35

    # Create axis
    ax = Axis(fig[1, idx],
        title = "Sobol Indices for $(group.name)\n(Tumor at day $time_req)",
        xlabel = "Sensitivity Index",
        ylabel = idx == 1 ? "Parameters" : "",
        yticks = (y_pos, group_data.parameter),
        yticklabelsize = 12,
        xgridvisible = true,
        ygridvisible = false
    )
    xlims!(ax, 0, 1.1)

    # Plot S1 (first-order) bars
    barplot!(ax, y_pos .- bar_width/2, group_data.S1,
             direction = :x, color = :steelblue, width = bar_width, label = "S1 (First-order)")

    # Plot ST (total-order) bars
    barplot!(ax, y_pos .+ bar_width/2, group_data.ST,
             direction = :x, color = :darkorange, width = bar_width, label = "ST (Total-order)")

    # Add error bars for confidence intervals
    errorbars!(ax, group_data.S1, y_pos .- bar_width/2,
               group_data.S1 .- group_data.S1_CI_low,
               group_data.S1_CI_high .- group_data.S1,
               direction = :x, color = :black, whiskerwidth = 5)

    errorbars!(ax, group_data.ST, y_pos .+ bar_width/2,
               group_data.ST .- group_data.ST_CI_low,
               group_data.ST_CI_high .- group_data.ST,
               direction = :x, color = :black, whiskerwidth = 5)
end

# Add legend at bottom
Legend(fig[2, 1:3],
    [PolyElement(color = :steelblue), PolyElement(color = :darkorange)],
    ["S1 (First-order: direct effect)", "ST (Total-order: including interactions)"],
    orientation = :horizontal,
    framevisible = false
)

fig

# ==============================================================================
# PART 8: SAVE OUTPUTS
# ==============================================================================

println("\nSaving outputs...")

mkpath("outputs")

# Save Sobol plot
save("outputs/gsa_sobol_plots.png", fig, px_per_unit = 2)
println("  Saved: outputs/gsa_sobol_plots.png")

# Save Sobol indices table
CSV.write("outputs/gsa_sobol_indices.csv", gsa_results)
println("  Saved: outputs/gsa_sobol_indices.csv")

# ==============================================================================
# PART 9: RESULTS SUMMARY
# ==============================================================================

println("\n" * "="^60)
println("GSA RESULTS SUMMARY")
println("="^60)

for group in dose_groups
    println("\n--- $(group.name) ---")
    group_data = @chain gsa_results begin
        @rsubset(:dose_group == group.name)
        sort(:ST, rev = true)  # Most sensitive first by total effect
    end

    println("  Parameters ranked by total-order sensitivity (ST):")
    for row in eachrow(group_data)
        interaction = row.ST - row.S1
        println("    $(rpad(row.parameter, 10)): S1 = $(lpad(round(row.S1, digits=3), 5)), " *
                "ST = $(lpad(round(row.ST, digits=3), 5)), " *
                "Interaction = $(round(interaction, digits=3))")
    end
end

# ==============================================================================
# INTERPRETATION
# ==============================================================================

println("\n" * "="^60)
println("GSA INTERPRETATION")
println("="^60)

println("""

Understanding Sobol Indices:

1. FIRST-ORDER INDEX (S1):
   - Fraction of output variance due to parameter alone
   - S1 = 0.5 means parameter explains 50% of variance directly
   - Higher S1 → parameter has strong direct effect

2. TOTAL-ORDER INDEX (ST):
   - Includes direct effect + all interactions with other parameters
   - ST ≥ S1 always (equality when no interactions)
   - ST - S1 = contribution from parameter interactions

3. KEY INSIGHTS:
   - If ST ≈ S1: Parameter acts independently
   - If ST >> S1: Parameter has significant interactions
   - Sum of all S1 ≈ 1 if model is additive
   - Sum of all ST > 1 indicates interactions

4. COMPARISON WITH LSA:
   - LSA: Local perturbations (±50%) around baseline
   - GSA: Explores entire parameter space uniformly
   - GSA captures non-linear effects and interactions
   - Both should identify similar "most sensitive" parameters

5. BIOLOGICAL INSIGHTS:
   - lambda_0: Dominant driver of tumor growth variability
   - k2: Important for treatment response (treated groups only)
   - Interactions reveal coupled effects between growth and drug action
""")

println("\n" * "="^60)
println("GLOBAL SENSITIVITY ANALYSIS COMPLETE")
println("="^60)
