using SMLMBaGoL
using Random
using StatsBase
using SMLMData

# Set random seed for reproducibility
Random.seed!(123)

# Create synthetic data
println("Creating synthetic data...")

# Generate n-mer ground truth
N_EMITTERS = 6
DIAMETER = 100.0  # nm
PHOTONS = 10_000.0

# Create emitters in a circle
angles = range(0, 2π, length=N_EMITTERS+1)[1:end-1]
radius = DIAMETER / 2
ground_truth_positions = [(radius * cos(θ), radius * sin(θ)) for θ in angles]
ground_truth_emitters = [Emitter2D(x, y, PHOTONS) for (x, y) in ground_truth_positions]

# Generate synthetic localizations
observations_per_emitter = 50
τ = 50.0  # nm linking precision
σ_loc = 25.0  # nm localization precision

localizations = Localization2D[]
for (emitter_idx, emitter) in enumerate(ground_truth_emitters)
    n_obs = observations_per_emitter
    for _ in 1:n_obs
        # Sample true position from emitter
        x_true = emitter.x + randn() * τ
        y_true = emitter.y + randn() * τ
        
        # Observed position with localization error
        x_obs = x_true + randn() * σ_loc
        y_obs = y_true + randn() * σ_loc
        
        push!(localizations, Localization2D(x_obs, y_obs, σ_loc, σ_loc, 1))
    end
end

println("Generated $(length(localizations)) localizations from $(length(ground_truth_emitters)) emitters")

# Set up priors
spatial_prior = UniformSpatialPrior(-200.0, 200.0, -200.0, 200.0)
count_prior = HierarchicalNegBinomialPrior(
    50.0,     # μ = 50 localizations per emitter expected
    2.0,      # κ = 2 concentration
    τ^2,      # τ² = 2500 nm² (50 nm)²
    (1.0, 0.02),   # μ_hyperprior (a₀, b₀) for Gamma prior on μ
    (1.0, 0.5),    # κ_hyperprior (c₀, d₀) for Gamma prior on κ  
    (3.0, 7500.0)  # τ²_hyperprior (a_τ, b_τ) for InverseGamma prior on τ²
)

# Create initial state with a few emitters
initial_positions = [
    (0.0, 0.0),
    (50.0, 0.0),
    (-50.0, 0.0)
]
initial_emitters = [Emitter2D(x, y, PHOTONS) for (x, y) in initial_positions]

# Initialize allocations randomly
initial_allocations = rand(1:length(initial_emitters), length(localizations))

# Sample initial latent positions
τ² = 50.0^2
initial_latent_positions = Vector{Tuple{Float64,Float64}}(undef, length(localizations))
for (i, loc) in enumerate(localizations)
    emitter = initial_emitters[initial_allocations[i]]
    prec_x = 1/τ² + 1/loc.σx^2
    prec_y = 1/τ² + 1/loc.σy^2
    post_mean_x = (emitter.x/τ² + loc.x/loc.σx^2) / prec_x
    post_mean_y = (emitter.y/τ² + loc.y/loc.σy^2) / prec_y
    
    latent_x = post_mean_x + randn() / sqrt(prec_x)
    latent_y = post_mean_y + randn() / sqrt(prec_y)
    
    initial_latent_positions[i] = (latent_x, latent_y)
end

# Create initial state (will compute log_likelihood internally)
initial_state_temp = BaGoLState(
    initial_emitters,
    localizations,
    initial_allocations,
    initial_latent_positions,
    spatial_prior,
    count_prior,
    τ²,
    0.0  # Dummy value, will be recomputed
)

# Compute actual log_likelihood
initial_state = BaGoLState(
    initial_emitters,
    localizations,
    initial_allocations,
    initial_latent_positions,
    spatial_prior,
    count_prior,
    τ²,
    log_likelihood(initial_state_temp)
)

println("\nInitial state:")
println("- $(length(initial_state.emitters)) emitters")
println("- Log-likelihood: $(initial_state.log_likelihood)")

# Create RJMCMC chain using initialize_chain
chain = initialize_chain(
    localizations,
    Emitter2D{Float64},
    spatial_prior,
    count_prior;
    initial_K = length(initial_emitters),
    burn_in = 0,
    thin = 1
)

# Test birth move
println("\n=== Testing Birth Move ===")
for i in 1:5
    proposed_state = propose_move(Birth, initial_state, chain)
    if !isnothing(proposed_state)
        log_α = log_acceptance_ratio(Birth, initial_state, proposed_state, chain)
        α = exp(log_α)
        println("Birth $i: $(length(initial_state.emitters)) → $(length(proposed_state.emitters)) emitters")
        println("  Log-likelihood change: $(proposed_state.log_likelihood - initial_state.log_likelihood)")
        println("  Log acceptance ratio: $log_α (α = $α)")
        
        # Check if any localizations were allocated to the new emitter
        new_emitter_idx = length(proposed_state.emitters)
        n_allocated = count(==(new_emitter_idx), proposed_state.allocations)
        println("  Localizations allocated to new emitter: $n_allocated")
    else
        println("Birth $i: Failed (returned nothing)")
    end
end

# Test death move on a state with more emitters
test_state = chain.current_state
for i in 1:5
    proposed = propose_move(Birth, test_state, chain)
    if !isnothing(proposed) && rand() < 0.8  # Accept most births for testing
        global test_state = proposed
    end
end

println("\n=== Testing Death Move ===")
println("Test state has $(length(test_state.emitters)) emitters")

for i in 1:5
    proposed_state = propose_move(Death, test_state, chain)
    if !isnothing(proposed_state)
        log_α = log_acceptance_ratio(Death, test_state, proposed_state, chain)
        α = exp(log_α)
        println("Death $i: $(length(test_state.emitters)) → $(length(proposed_state.emitters)) emitters")
        println("  Log-likelihood change: $(proposed_state.log_likelihood - test_state.log_likelihood)")
        println("  Log acceptance ratio: $log_α (α = $α)")
        
        # Check that all localizations are still allocated
        unallocated = count(==(0), proposed_state.allocations)
        println("  Unallocated localizations: $unallocated")
    else
        println("Death $i: Failed (returned nothing)")
    end
end

println("\n=== Testing Reversibility ===")
# Test that birth followed by death can recover similar state
original_state = test_state
birth_state = propose_move(Birth, original_state, chain)
if !isnothing(birth_state)
    death_state = propose_move(Death, birth_state, chain)
    if !isnothing(death_state)
        println("Original: $(length(original_state.emitters)) emitters, LL = $(original_state.log_likelihood)")
        println("After birth: $(length(birth_state.emitters)) emitters, LL = $(birth_state.log_likelihood)")
        println("After death: $(length(death_state.emitters)) emitters, LL = $(death_state.log_likelihood)")
        
        # Check acceptance ratios satisfy detailed balance
        log_α_birth = log_acceptance_ratio(Birth, original_state, birth_state, chain)
        log_α_death = log_acceptance_ratio(Death, birth_state, death_state, chain)
        println("\nLog α (birth): $log_α_birth")
        println("Log α (death): $log_α_death")
        println("Sum should be close to 0 for detailed balance: $(log_α_birth + log_α_death)")
    end
end

println("\nTest completed!")