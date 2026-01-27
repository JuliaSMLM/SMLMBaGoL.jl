# Partitioned BaGoL Demo
# ======================
# Demonstrates parallel processing of large datasets using spatial partitioning.
# Precision-weighted DBSCAN clusters localizations, then runs BaGoL in parallel.
#
# Run with: julia --threads=auto --project=examples examples/08_partitioned_demo.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMData
using CairoMakie
using Random
using Statistics

const OUTPUT_DIR = joinpath(@__DIR__, "output")
mkpath(OUTPUT_DIR)

println("="^60)
println("Partitioned BaGoL Demo")
println("="^60)
println("Threads available: $(Threads.nthreads())")

Random.seed!(2024)

# =============================================================================
# 1. GENERATE LARGE SIMULATED DATASET
# =============================================================================
println("\n--- Generating simulated data ---")

# Create multiple well-separated clusters to demonstrate partitioning
n_clusters = 6
emitters_per_cluster = 3
locs_per_emitter = 8
σ_loc = 0.008  # 8 nm localization precision
cluster_spacing = 0.300  # 300 nm between clusters
emitter_spacing = 0.040  # 40 nm within clusters

all_locs = SMLMData.Emitter2DFit[]
all_true_positions = Tuple{Float64, Float64}[]

let loc_id = 1
    for cluster_idx in 1:n_clusters
        # Arrange clusters in a 2x3 grid
        row = (cluster_idx - 1) ÷ 3
        col = (cluster_idx - 1) % 3
        cluster_cx = 0.1 + col * cluster_spacing
        cluster_cy = 0.1 + row * cluster_spacing

        # Create emitters within cluster (triangle pattern)
        for emitter_idx in 1:emitters_per_cluster
            θ = 2π * (emitter_idx - 1) / emitters_per_cluster - π/2
            r = emitter_spacing / (2 * sin(π / emitters_per_cluster))
            ex = cluster_cx + r * cos(θ)
            ey = cluster_cy + r * sin(θ)
            push!(all_true_positions, (ex, ey))

            # Generate localizations for this emitter
            for _ in 1:locs_per_emitter
                x = ex + randn() * σ_loc
                y = ey + randn() * σ_loc
                σ = σ_loc * (0.9 + 0.2 * rand())
                push!(all_locs, SMLMData.Emitter2DFit(
                    x, y, 1000.0, 10.0, σ, σ, 0.0, 50.0, 1.0, 1, 1, 0, loc_id
                ))
                loc_id += 1
            end
        end
    end
end

total_emitters = n_clusters * emitters_per_cluster
total_locs = length(all_locs)
println("Generated $total_locs localizations from $total_emitters emitters")
println("  $n_clusters clusters × $emitters_per_cluster emitters/cluster × $locs_per_emitter locs/emitter")

# =============================================================================
# 2. RUN STANDARD BAGOL (for comparison)
# =============================================================================
println("\n--- Running standard BaGoL (single chain) ---")

camera = SMLMData.IdealCamera(64, 64, 0.1)
smld_input = SMLMData.BasicSMLD(all_locs, camera, 100, 1)

result_standard, diag_standard = run_bagol(
    smld_input;
    nsigma = Inf,  # Disable partitioning (all locs in one cluster)
    λ_K = Float64(total_emitters),
    n_iterations = 10000,
    burn_in = 2000,
    verbose = true
)

println("\nStandard BaGoL MAP-N: $(diag_standard.n_emitters) (true: $total_emitters)")

# =============================================================================
# 3. RUN PARTITIONED BAGOL VIA UNIFIED API
# =============================================================================
println("\n--- Running partitioned BaGoL (parallel via SMLD interface) ---")

# Always partitions by default with nsigma=3.0
result_partitioned, diag_partitioned = run_bagol(
    smld_input;
    nsigma = 3.0,              # DBSCAN threshold in sigma units
    max_partition_size = 100,  # Target max per partition
    sync_interval = 500,       # Global hierarchical sync every 500 iters
    n_iterations = 5000,       # Total iterations
    burn_in = 1000,
    verbose = true
)

println("\n--- Partitioned Results ---")
println("Partitioned MAP-N: $(diag_partitioned.n_emitters) (true: $total_emitters)")

# =============================================================================
# 4. COMPARE RESULTS
# =============================================================================
println("\n" * "="^60)
println("COMPARISON")
println("="^60)
println("True emitters:          $total_emitters")
println("Standard BaGoL MAP-N:   $(diag_standard.n_emitters)")
println("Partitioned BaGoL MAP-N: $(diag_partitioned.n_emitters)")

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
println("\n--- Generating visualization ---")

fig = Figure(size=(1200, 500))

# Plot 1: Input localizations with true positions
ax1 = Axis(fig[1, 1], title="Input Localizations",
           xlabel="x (μm)", ylabel="y (μm)", aspect=DataAspect())

scatter!(ax1, [loc.x for loc in all_locs], [loc.y for loc in all_locs],
         color=(:gray, 0.3), markersize=3)
scatter!(ax1, [p[1] for p in all_true_positions], [p[2] for p in all_true_positions],
         marker='x', color=:black, markersize=12, label="True emitters")
axislegend(ax1, position=:rt, framevisible=false)

# Plot 2: Standard vs Partitioned results
ax2 = Axis(fig[1, 2], title="Standard vs Partitioned BaGoL",
           xlabel="x (μm)", ylabel="y (μm)", aspect=DataAspect())

scatter!(ax2, [p[1] for p in all_true_positions], [p[2] for p in all_true_positions],
         marker='x', color=:black, markersize=15, label="True ($total_emitters)")
scatter!(ax2, [e.x for e in result_standard.emitters], [e.y for e in result_standard.emitters],
         color=(:blue, 0.7), markersize=10, label="Standard ($(diag_standard.n_emitters))")
scatter!(ax2, [e.x for e in result_partitioned.emitters],
         [e.y for e in result_partitioned.emitters],
         color=(:red, 0.7), markersize=8, marker=:diamond,
         label="Partitioned ($(diag_partitioned.n_emitters))")
axislegend(ax2, position=:rt, framevisible=false)

# Plot 3: K posterior comparison
ax3 = Axis(fig[1, 3], title="K Posterior (Standard)",
           xlabel="K (emitter count)", ylabel="Frequency")

barplot!(ax3, 0:(length(diag_standard.posterior_k)-1), diag_standard.posterior_k,
         color=(:blue, 0.5), label="Standard")
vlines!(ax3, [total_emitters], color=:black, linestyle=:dash, linewidth=2, label="True K")
axislegend(ax3, position=:rt, framevisible=false)

save(joinpath(OUTPUT_DIR, "partitioned_demo.png"), fig)
println("Saved: $(joinpath(OUTPUT_DIR, "partitioned_demo.png"))")

println("\n" * "="^60)
println("Demo complete!")
println("="^60)
