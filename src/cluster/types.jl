

mutable struct Subregion 
    obs::Observations
    chains::Vector{RJMCMC_Chain}
end

function Subregion(obs::Observations, n_chains::Int)
    chains = Vector{RJMCMC_Chain}(undef, n_chains)
    for i in 1:n_chains
        chains[i] = RJMCMC_Chain([Params(AbstractEmitter[])], [Allocations(Int[])])
    end
    return Subregion(obs, chains)
end



