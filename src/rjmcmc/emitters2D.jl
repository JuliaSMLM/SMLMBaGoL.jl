
mutable struct Emitter2D{T} <: AbstractEmitter
    μ_y::T
    μ_x::T
end

struct Localization2D{T} <: AbstractObservation
    y::T
    x::T
    σ_y::T
    σ_x::T
end

function add_emitter(θ::Params, new_emitter::Emitter2D)
    θ_new = Params(θ.emitters)
    push!(θ_new.emitters, new_emitter)
    return θ_new
end

function remove_emitter(θ::Params{T}, id::Int) where T <: Emitter2D
    θ_new = Params(θ.emitters)
    deleteat!(θ_new.emitters, id)
    return θ_new
end

function p_z_given_y(obs::Localization2D, emitter::Emitter2D)
    return pdf(Normal(emitter.μ_x, emitter.σ_x), obs.x) *
        pdf(Normal(emitter.μ_y, emitter.σ_y), obs.y)
end

function gibbs_mu!(μ_vals::Vector{T}, id::Int, x_vals::Vector{T}, sigma_vals::Vector{T}, z_vals::Vector{Int}) where T<:Real     
    # If there are no observations of this id, then sample from the localizations 
    # TODO Should sample from prior instead.
    mask = z_vals .== id
    if !any(mask)
        μ_vals[id] = x_vals[rand(1:length(z_vals))]
        return
    end

    # Otherwise, sample from the posterior
    # Modeling as a normal distribution with known variance
    xs = view(x_vals, mask)
    sigs = view(sigma_vals, mask)
    a = sum(xs ./ (sigs .^ 2))
    b = sum(1 ./ (sigs .^ 2))
    x_mle = a/b
    x_se = 1 / sqrt(b)
    μ_vals[id] = randn(T) * x_se + x_mle
end

function gibbs_mu!(emitter::Emitter2D, ŷ::Observations, z::Allocations, id::Int) 
    gibbs_mu!(emitter.μ_x, ŷ.x, ŷ.σ_x, z.idx, id)
    gibbs_mu!(emitter.μ_y, ŷ.y, ŷ.σ_y, z.idx, id)        
end

function build_prior_y(obs::Observations)
    # Make a Mixture Model Distribution from Observations
    means = [[obs.y, obs.x] for obs in obs.ŷ]
    covs = [[obs.σ_y^2, 0.0; 0.0, obs.σ_x^2] for obs in obs.ŷ]
    components = [MvNormal(mean, cov) for (mean, cov) in zip(means, covs)]
    weights = [1.0 for obs in obs.ŷ]./length(obs.ŷ)
    return MixtureModel(components, weights)
end


