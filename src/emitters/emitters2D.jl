
mutable struct Emitter2D{T} <: AbstractEmitter where {T<:Real}
    x::T
    y::T
end
function Emitter2D(coords::Vector{<:Real})
    return Emitter2D(coords[1], coords[2])
end
function Emitter2D{T}(coords::Vector{T}) where {T<:Real}
    return Emitter2D(coords[2], coords[1])
end
function Emitter2D{T}(emitter::Emitter2D{T}) where T
    return Emitter2D{T}(emitter.x, emitter.y)
end

struct Localization2D{T} <: AbstractObservation
    x::T
    y::T
    σ_x::T
    σ_y::T
end
# Constructor for Localization2D using type of Emitter2D
function Localization2D{T}(emitter::Emitter2D{T}, σ_x::T, σ_y::T) where T
    return Localization2D(emitter.x, emitter.y, σ_x, σ_y)
end


function log_p_z_given_y(loc::Localization2D, emitter::Emitter2D)
    return logpdf(Normal(loc.x, loc.σ_x), emitter.x) +
           logpdf(Normal(loc.y, loc.σ_y), emitter.y)
end


function move!(emitter::Emitter2D, obs::Observations, z::Allocations, id_emitter::Int, prior_y::Distributions.Distribution)
    # handle the case where there are no observations for this emitter
    
    # Use mean and variance of Observations to sample from a Normal distribution
    x_sum = 0.0
    y_sum = 0.0
    x_var = 0.0
    y_var = 0.0
    for i in 1:length(obs)
        if z.idx[i] == id_emitter
            x_sum += obs.ŷ[i].x * obs.ŷ[i].σ_x^(-2)
            y_sum += obs.ŷ[i].y * obs.ŷ[i].σ_y^(-2)
            x_var += obs.ŷ[i].σ_x^(-2)
            y_var += obs.ŷ[i].σ_y^(-2)
        end
    end

    emitter.x = rand(Normal(x_sum / x_var, sqrt(1 / x_var)))
    emitter.y = rand(Normal(y_sum / y_var, sqrt(1 / y_var)))
end

function build_prior_y(obs::Observations) 
    # Make a Mixture Model Distribution from Observations
    means = [[obs.y, obs.x] for obs in obs.ŷ]
    covs = [[obs.σ_y^2 0.0; 0.0 obs.σ_x^2] for obs in obs.ŷ]
    components = [MvNormal(mean, cov) for (mean, cov) in zip(means, covs)]
    weights = [1.0 for obs in obs.ŷ] ./ length(obs.ŷ)
    return MixtureModel(components, weights)
end

function build_prior_y(locs::Vector{Localization2D{T}}) where {T<:Real}
    # Make a Mixture Model Distribution from Observations
    means = [[obs.y, obs.x] for obs in locs]
    covs = [[obs.σ_y^2 0.0; 0.0 obs.σ_x^2] for obs in locs]
    components = [MvNormal(mean, cov) for (mean, cov) in zip(means, covs)]
    weights = [1.0 for obs in locs] ./ length(locs)
    return MixtureModel(components, weights)
end

function gen_emitter!(emitter::Emitter2D{T}, prior_y::Distributions.Distribution) where T <: Real
    emitter.x, emitter.y = rand(prior_y)
end

function gen_emitters(ET::Type{<:SMLMBaGoL.Emitters.Emitter2D{T}}, n::Int, prior_y::Distributions.Distribution) where T <: Real
    return [Emitter2D{T}(rand(prior_y)) for i in 1:n]
end


function gen_observations(prior_λ, emitters::Vector{SMLMBaGoL.Emitters.Emitter2D{T}}; 
    photons::Float64=1000.0, min_photons::Int=200) where T <: Real

    p = Exponential(photons)
    p = Truncated(p, min_photons, Inf)
    σ_PSF = 100.0

    # make empty arrays
    y = Float64[]
    x = Float64[]
    σ_y = Float64[]
    σ_x = Float64[]

    for emitter in emitters
        n = Int(round((rand(prior_λ))))

        σ_temp = σ_PSF ./ sqrt.(rand(p, n))

        y_temp = [rand(Normal(emitter.y, σ_temp[i])) for i in 1:n]
        x_temp = [rand(Normal(emitter.x, σ_temp[i])) for i in 1:n]

        append!(y, y_temp)
        append!(x, x_temp)
        append!(σ_y, σ_temp)
        append!(σ_x, σ_temp)

    end

    return Observations(Localization2D.(x, y, σ_x, σ_y))
end

function merge_emitters(emitter1::Emitter2D, emitter2::Emitter2D)
    new_x = (emitter1.x + emitter2.x) / 2
    new_y = (emitter1.y + emitter2.y) / 2
    return Emitter2D(new_x, new_y)
end
