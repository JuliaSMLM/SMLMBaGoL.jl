# Optimality Sweep + Speed Test
# ==============================
# Runs systematic d/σ sweep across (n, μ) conditions and speed benchmark.
#
# Run with: julia --threads=auto --project=examples examples/nmer_stats.jl

using Pkg; Pkg.activate(@__DIR__)
using SMLMBaGoL
using CairoMakie   # activates BaGoLMakieExt

# =============================================================================
# Optimality sweep: d/σ curves for n∈{2,8}, μ∈{5,10}
# =============================================================================

output_dir = joinpath(@__DIR__, "output", "optimality")
rm(output_dir; force=true, recursive=true)
mkpath(output_dir)

sweep = run_optimality_sweep(;
    n_values = [2, 8],
    mu_values = [5.0, 10.0],
    d_sigma_range = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    n_trials = 25,
    n_iterations = 8000,
    burn_in = 2000,
    fixed_sigma = 0.010,
    shape = 5.0
)

write_sweep(sweep; output_dir)
plot_sweep(sweep; output_dir)

# =============================================================================
# Speed test
# =============================================================================

speed = run_speed_test(;
    n_locs_range = [100, 500, 1000, 5000],
    n_repeats = 3,
    n_iterations = 5000,
    burn_in = 1000
)

write_speed(speed; output_dir)
plot_speed(speed; output_dir)

println("\nResults in $output_dir")
