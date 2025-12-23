# N-mer Demo
# ===========
# Tests cluster resolution on simulated n-mers (dimers, trimers).
#
# Run with: julia --project=examples examples/02_nmer_demo.jl

using Pkg
Pkg.activate(@__DIR__)

using SMLMBaGoL
using SMLMData
using Random
using CairoMakie
using Statistics

const OUTPUT_DIR = joinpath(@__DIR__, "output")

println("="^60)
println("N-mer Demo")
println("="^60)

Random.seed!(123)

# -----------------------------------------------------------------------------
# Generate n-mers
# -----------------------------------------------------------------------------
n_dimers = 4
n_trimers = 2
emitter_spacing = 0.040  # 40 nm within clusters
cluster_spacing = 0.200  # 200 nm between clusters
σ_loc = 0.008
locs_per_emitter = 8

function generate_nmer(cx, cy, n, spacing, σ_loc, n_locs, start_id)
    locs = SMLMData.Emitter2DFit[]
    positions = Tuple{Float64, Float64}[]

    # Arrange emitters in line for dimer, triangle for trimer
    for i in 1:n
        if n == 2
            ex = cx + (i == 1 ? -spacing/2 : spacing/2)
            ey = cy
        else
            θ = 2π * (i-1) / n - π/2
            r = spacing / (2 * sin(π / n))
            ex = cx + r * cos(θ)
            ey = cy + r * sin(θ)
        end
        push!(positions, (ex, ey))

        for j in 1:n_locs
            x = ex + randn() * σ_loc
            y = ey + randn() * σ_loc
            σ = σ_loc * (0.9 + 0.2 * rand())
            push!(locs, SMLMData.Emitter2DFit(x, y, 1000.0, 10.0, σ, σ, 50.0, 1.0, 1, 1, 0, start_id))
            start_id += 1
        end
    end
    return locs, positions, start_id
end

all_locs = SMLMData.Emitter2DFit[]
all_positions = Tuple{Float64, Float64}[]
loc_id = 1

for i in 1:(n_dimers + n_trimers)
    n = i <= n_dimers ? 2 : 3
    row = (i - 1) ÷ 3
    col = (i - 1) % 3
    cx = 0.1 + col * cluster_spacing
    cy = 0.1 + row * cluster_spacing

    locs, pos, new_id = generate_nmer(cx, cy, n, emitter_spacing, σ_loc, locs_per_emitter, loc_id)
    global loc_id = new_id
    append!(all_locs, locs)
    append!(all_positions, pos)
end

total_emitters = n_dimers * 2 + n_trimers * 3
println("\nGenerated $(length(all_locs)) localizations from $total_emitters emitters")
println("  $(n_dimers) dimers, $(n_trimers) trimers")

# -----------------------------------------------------------------------------
# Run BaGoL
# -----------------------------------------------------------------------------
println("\nRunning BaGoL...")

chain = run_bagol(
    all_locs;
    τ = 0.010,
    λ_K = Float64(total_emitters),
    n_iterations = 10000,
    burn_in = 2000,
    verbose = true
)

result = estimate_mapn(chain)

println("\n" * "="^60)
println("Results")
println("="^60)
println("True emitters: $total_emitters")
println("MAP-N: $(result.n_emitters)")

# Visualization with full chain samples
fig = plot_bagol(chain, result, all_locs;
    true_positions=all_positions,
    save_path=joinpath(OUTPUT_DIR, "nmer_result.png"))
println("\nSaved: $(joinpath(OUTPUT_DIR, "nmer_result.png"))")
