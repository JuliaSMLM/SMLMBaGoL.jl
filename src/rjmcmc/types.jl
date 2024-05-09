import Base: length


struct RJMCMC_Chain
    states::Vector{Params}
    allocations::Vector{Allocations}
end
length(chain::RJMCMC_Chain) = length(chain.states)

struct RJMCMC_ROI
    obs::SMLMBaGoL.Observations
    prior_y::Distributions.Distribution
    prior_k::Distributions.Distribution
    p_jump::Distributions.Categorical    
    emitter_type::Type
    prior_λ::Distributions.Distribution
end


