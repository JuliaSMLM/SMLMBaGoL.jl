
mutable struct Emitter2D{T} <: AbstractEmitter where T<:Real
    x::T
    y::T
end
function Emitter2D(coords::Vector{<:Real})
    return Emitter2D(coords[1], coords[2])
end
function Emitter2D{T}(coords::Vector{T}) where T<:Real
    return Emitter2D(coords[2], coords[1])
end


struct Localization2D{T} <: AbstractObservation
    x::T
    y::T
    σ_x::T
    σ_y::T
end


function log_p_z_given_y(loc::Localization2D, emitter::Emitter2D)
    return logpdf(Normal(loc.x, loc.σ_x), emitter.x) +
        logpdf(Normal(loc.y, loc.σ_y), emitter.y)
end


function move!(emitter::Emitter2D, obs::Observations, z::Allocations, id_emitter::Int, prior_y::Distributions.Distribution)
    # handle the case where there are no observations for this emitter
    if !has_allocation(z, id_emitter)
        # sample from the prior
        emitter.y, emitter.x = rand(prior_y)
        return
    end
    
    # build a prior distribution from the observations for this emitter
    locs = [obs.ŷ[i] for i in 1:length(obs) if z.idx[i] == id_emitter]
    prior_y = build_prior_y(locs)
    # sample from the prior
    emitter.y, emitter.x = rand(prior_y)
end



function build_prior_y(obs::Observations)
    # Make a Mixture Model Distribution from Observations
    means = [[obs.y, obs.x] for obs in obs.ŷ]
    covs = [[obs.σ_y^2 0.0; 0.0 obs.σ_x^2] for obs in obs.ŷ]
    components = [MvNormal(mean, cov) for (mean, cov) in zip(means, covs)]
    weights = [1.0 for obs in obs.ŷ]./length(obs.ŷ)
    return MixtureModel(components, weights)
end

function build_prior_y(locs::Vector{Localization2D{T}}) where T<:Real
    # Make a Mixture Model Distribution from Observations
    means = [[obs.y, obs.x] for obs in locs]
    covs = [[obs.σ_y^2 0.0; 0.0 obs.σ_x^2] for obs in locs]
    components = [MvNormal(mean, cov) for (mean, cov) in zip(means, covs)]
    weights = [1.0 for obs in locs]./length(locs)
    return MixtureModel(components, weights)
end



# https://juliastats.org/Distributions.jl/stable/convolution/
function convolve_distribution(distribution::UnivariateDistribution, k::Int)
    convolved_distribution = distribution
    for _ in 2:k
        convolved_distribution = convolve(convolved_distribution, distribution)
    end
    return convolved_distribution
end

function build_prior_k(obs::Observations, prior_λ::Distributions.UnivariateDistribution)
    max_k = length(obs.ŷ)
    N = max_k
    probabilities = zeros(max_k + 1)
    for k in 0:max_k
        convolved_distribution = convolve_distribution(prior_λ, k)
        probabilities[k + 1] = pdf(convolved_distribution, N)
    end
    probabilities ./= sum(probabilities)  # Normalize to make it a valid probability distribution
    return Categorical(probabilities)
end


function gen_emitters2D(n::Int, prior_y::Distributions.Distribution)
    return Params([Emitter2D(rand(prior_y)) for i in 1:n])
end

function gen_observations2D(prior_λ, emitters::Params ; photons::Float64 = 1000.0)
    
    p = Exponential(photons)
    σ_PSF = 100.0
    
    # make empty arrays
    y = Float64[]
    x = Float64[]
    σ_y = Float64[]
    σ_x = Float64[]

    for emitter in emitters.emitters
        n = Int(round((rand(prior_λ))))
        
        σ_temp = σ_PSF./sqrt.(rand(p, n))/10
        
        y_temp = [rand(Normal(emitter.y,σ_temp[i]))  for i in 1:n]
        x_temp = [rand(Normal(emitter.x,σ_temp[i]))  for i in 1:n]
        
        append!(y, y_temp)
        append!(x, x_temp)
        append!(σ_y, σ_temp)
        append!(σ_x, σ_temp)

    end
    
    return Observations(Localization2D.(x, y, σ_x, σ_y))
end


