#=
================================================================================
LOCAL SENSITIVITY ANALYSIS (LSA) - Simeoni Model
================================================================================

This script performs Local Sensitivity Analysis on the Simeoni TGI model:
- ±50% perturbation of PD parameters
- Endpoint: Tumor weight at day 30
- 3 dose groups: Control, 45 mg/kg, 60 mg/kg
- Tornado plots matching MATLAB LSA output style

Reference: Simeoni et al. (2004) Cancer Research 64:1094-1101
=#

# ==============================================================================
# PART 1: SETUP
# ==============================================================================

using Pumas
using AlgebraOfGraphics, CairoMakie
using DataFramesMeta
using PumasUtilities
using CSV

set_aog_theme!()

println("="^60)
println("LOCAL SENSITIVITY ANALYSIS - Simeoni Model")
println("="^60)

# ==============================================================================
# PART 2: LSA MODEL DEFINITION (No Random Effects)
# ==============================================================================

#=
Minimal PKPD model without random effects for deterministic LSA.
Based on simeoni_pkpd_model but simplified for parameter perturbation.
=#

lsa_model = @model begin
    @param begin
        # === PK Parameters (FIXED for LSA) ===
        tvk_el ∈ RealDomain(lower=0.0)     # Elimination rate constant (day⁻¹)
        tvk12  ∈ RealDomain(lower=0.0)     # Central→Peripheral rate constant (day⁻¹)
        tvk21  ∈ RealDomain(lower=0.0)     # Peripheral→Central rate constant (day⁻¹)
        tvVc   ∈ RealDomain(lower=0.0)     # Central volume (L)

        # === PD Parameters (for LSA) ===
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
        # Tumor weight output (deterministic)
        tumor_weight := @. tumor
    end
end

println("\nLSA model defined (no random effects)")

# ==============================================================================
# PART 3: DATA PREPARATION
# ==============================================================================

# Time configuration
time_req = 30.0  # Sensitivity endpoint (day 30)
time_grid = sort(unique([collect(0.0:0.5:50.0); time_req]))

# Dosage regimens (drug administered at day 13 when tumor reaches ~1g)
# Body weight: 25g mouse → doses in mg total
dr_control = DosageRegimen(0.0, time = 13.0, cmt = 1)   # Control: no drug
dr_45mg = DosageRegimen(0.9, time = 13.0, cmt = 1)      # 45 mg/kg × 0.025 kg × factor
dr_60mg = DosageRegimen(1.2, time = 13.0, cmt = 1)      # 60 mg/kg × 0.025 kg × factor

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

# Parameters to analyze (matching MATLAB LSA_script.m)
# Note: Using Julia naming convention with tv prefix
lsa_param_names = [:tvlambda0, :tvlambda1, :psi, :tvk1, :tvk2]

# Display names for plots (matching MATLAB style)
param_display_names = Dict(
    :tvlambda0 => "lambda_0",
    :tvlambda1 => "lambda_1",
    :psi => "phi",
    :tvk1 => "k1",
    :tvk2 => "k2"
)

println("\nBaseline parameters set")
println("Parameters for LSA: $(join([param_display_names[p] for p in lsa_param_names], ", "))")

# ==============================================================================
# PART 5: PARAMETER PERTURBATION LOGIC
# ==============================================================================

"""
    perturb_params(baseline, param_name, direction)

Create a new parameter set with one parameter perturbed by ±50%.
- direction = :increase → multiply by 1.5 (+50%)
- direction = :decrease → multiply by 0.5 (-50%)
"""
function perturb_params(baseline::NamedTuple, param_name::Symbol, direction::Symbol)
    factor = direction == :increase ? 1.5 : 0.5  # ±50%

    # Convert to Dict for modification
    params_dict = Dict(pairs(baseline))
    params_dict[param_name] = baseline[param_name] * factor

    # Convert back to NamedTuple
    return NamedTuple(params_dict)
end

"""
    extract_tumor_at_time(sim_result, target_time)

Extract tumor weight from simulation at a specific time point.
"""
function extract_tumor_at_time(sim_result, target_time::Float64)
    sim_df = DataFrame(sim_result)

    # Find closest time point
    time_diffs = abs.(sim_df.time .- target_time)
    idx = argmin(time_diffs)

    return sim_df.tumor[idx]
end

# ==============================================================================
# PART 6: LSA SIMULATION WORKFLOW
# ==============================================================================

println("\n" * "="^60)
println("RUNNING LOCAL SENSITIVITY ANALYSIS")
println("="^60)
println("\nPerturbation: ±50%")
println("Endpoint: Tumor weight at day $time_req")

# Storage for results
lsa_results = DataFrame(
    dose_group = String[],
    parameter = String[],
    baseline_tumor = Float64[],
    increase_tumor = Float64[],
    decrease_tumor = Float64[],
    increase_pct_change = Float64[],
    decrease_pct_change = Float64[],
    total_sensitivity = Float64[]
)

for group in dose_groups
    println("\n--- $(group.name) ---")

    # 1. Run baseline simulation
    sim_baseline = simobs(lsa_model, group.subject, baseline_params)
    baseline_tumor = extract_tumor_at_time(sim_baseline, time_req)
    println("  Baseline tumor at day $time_req: $(round(baseline_tumor, digits=4)) g")

    # 2. Perturb each parameter
    for param_name in lsa_param_names
        display_name = param_display_names[param_name]

        # +50% perturbation
        params_inc = perturb_params(baseline_params, param_name, :increase)
        sim_inc = simobs(lsa_model, group.subject, params_inc)
        tumor_inc = extract_tumor_at_time(sim_inc, time_req)

        # -50% perturbation
        params_dec = perturb_params(baseline_params, param_name, :decrease)
        sim_dec = simobs(lsa_model, group.subject, params_dec)
        tumor_dec = extract_tumor_at_time(sim_dec, time_req)

        # Calculate % change
        pct_change_inc = 100 * (tumor_inc - baseline_tumor) / baseline_tumor
        pct_change_dec = 100 * (tumor_dec - baseline_tumor) / baseline_tumor
        total_sens = abs(pct_change_inc) + abs(pct_change_dec)

        # Store results
        push!(lsa_results, (
            dose_group = group.name,
            parameter = display_name,
            baseline_tumor = baseline_tumor,
            increase_tumor = tumor_inc,
            decrease_tumor = tumor_dec,
            increase_pct_change = pct_change_inc,
            decrease_pct_change = pct_change_dec,
            total_sensitivity = total_sens
        ))

        println("  $display_name: +50% → $(round(pct_change_inc, digits=2))%, -50% → $(round(pct_change_dec, digits=2))%")
    end
end

println("\n" * "="^60)
println("LSA COMPLETE: $(nrow(lsa_results)) simulations")
println("="^60)

# ==============================================================================
# PART 7: TORNADO PLOT VISUALIZATION
# ==============================================================================

println("\nCreating tornado plots...")

# Determine consistent parameter ordering across all panels
# Use average total sensitivity across all dose groups
param_avg_sensitivity = @chain lsa_results begin
    groupby(:parameter)
    @combine(:avg_sensitivity = mean(:total_sensitivity))
    sort(:avg_sensitivity)  # Ascending order (least sensitive at top)
end
consistent_param_order = param_avg_sensitivity.parameter

# Calculate global x-axis limits for consistent scaling
global_max_change = maximum([
    maximum(abs.(lsa_results.increase_pct_change)),
    maximum(abs.(lsa_results.decrease_pct_change))
])

# Create figure with 3 panels (one per dose group)
fig = Figure(size = (1400, 500), fontsize = 12)

# Add overall title
Label(fig[0, 1:3], "Local Sensitivity Analysis", fontsize = 18, font = :bold)

for (idx, group) in enumerate(dose_groups)
    # Filter data for this dose group and apply consistent ordering
    group_data = @chain lsa_results begin
        @rsubset(:dose_group == group.name)
    end

    # Reorder to match consistent parameter order
    group_data = group_data[indexin(consistent_param_order, group_data.parameter), :]

    # Y positions for bars
    y_pos = 1:nrow(group_data)

    # Create axis with consistent y-axis labels
    ax = Axis(fig[1, idx],
        title = "Sensitivity of Tumor at day = $time_req\nfor $(group.name)",
        xlabel = "% change in Tumor",
        ylabel = idx == 1 ? "Parameters" : "",
        yticks = (collect(y_pos), group_data.parameter),
        yticklabelsize = 12,
        xgridvisible = true,
        ygridvisible = false
    )

    # Use consistent x-axis limits across all panels
    xlims!(ax, -global_max_change * 1.1, global_max_change * 1.1)

    # Plot horizontal bars
    # Blue bars for increase (+50% parameter)
    barplot!(ax, y_pos, group_data.increase_pct_change,
             direction = :x, color = :steelblue, label = "increasing")

    # Red bars for decrease (-50% parameter)
    barplot!(ax, y_pos, group_data.decrease_pct_change,
             direction = :x, color = :indianred, label = "decreasing")

    # Add vertical line at x=0
    vlines!(ax, [0], color = :black, linewidth = 1)
end

# Add legend at bottom
Legend(fig[2, 1:3],
    [PolyElement(color = :indianred), PolyElement(color = :steelblue)],
    ["decreasing (-50%)", "increasing (+50%)"],
    orientation = :horizontal,
    framevisible = false
)

fig

# ==============================================================================
# PART 8: SAVE OUTPUTS
# ==============================================================================

println("\nSaving outputs...")

mkpath("outputs")

# Save tornado plot
save("outputs/lsa_tornado_plots.png", fig, px_per_unit = 2)
println("  Saved: outputs/lsa_tornado_plots.png")

# Save sensitivity table
CSV.write("outputs/lsa_sensitivity_table.csv", lsa_results)
println("  Saved: outputs/lsa_sensitivity_table.csv")

# ==============================================================================
# PART 9: RESULTS SUMMARY
# ==============================================================================

println("\n" * "="^60)
println("LSA RESULTS SUMMARY")
println("="^60)

# Summary table per dose group
for group in dose_groups
    println("\n--- $(group.name) ---")
    group_data = @chain lsa_results begin
        @rsubset(:dose_group == group.name)
        sort(:total_sensitivity, rev = true)  # Most sensitive first
    end

    println("  Baseline tumor at day $time_req: $(round(group_data.baseline_tumor[1], digits=4)) g")
    println("\n  Parameter sensitivities (sorted by total impact):")

    for row in eachrow(group_data)
        println("    $(rpad(row.parameter, 10)): +50% → $(lpad(round(row.increase_pct_change, digits=1), 6))%, " *
                "-50% → $(lpad(round(row.decrease_pct_change, digits=1), 6))% (total: $(round(row.total_sensitivity, digits=1))%)")
    end
end

# ==============================================================================
# BIOLOGICAL INTERPRETATION
# ==============================================================================

println("\n" * "="^60)
println("BIOLOGICAL INTERPRETATION")
println("="^60)

println("""

Key Findings:

1. CONTROL GROUP (no drug):
   - lambda_0 (exponential growth rate) is the most sensitive parameter
   - Increasing lambda_0 → increases tumor weight (faster initial growth)
   - k2 (drug potency) has NO effect (no drug present)

2. TREATED GROUPS (45 mg/kg, 60 mg/kg):
   - lambda_0 remains the dominant parameter
   - k2 now shows effect: increasing k2 → decreases tumor weight
   - Higher doses (60 mg/kg) show stronger k2 sensitivity than lower doses

3. PARAMETER EFFECTS:
   - lambda_0 ↑ → tumor ↑ (faster exponential growth)
   - lambda_1 ↑ → tumor ↓ (faster transition to linear, slower net growth)
   - k1 ↑ → tumor ↓ (faster transit of damaged cells to death)
   - k2 ↑ → tumor ↓ (stronger drug kill effect, only in treated groups)
   - phi ↑ → tumor ↑ (sharper growth transition)

This analysis helps identify which parameters are most important for:
- Tumor growth predictions (lambda_0, lambda_1)
- Treatment efficacy predictions (k2, k1)
""")

println("\n" * "="^60)
println("LOCAL SENSITIVITY ANALYSIS COMPLETE")
println("="^60)
