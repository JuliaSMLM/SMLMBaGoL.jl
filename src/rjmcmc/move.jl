
function move!(θ::Params, ŷ::Observations, z::Allocations)
    # Move all the emitters
    for id in eachindex(θ)
        gibbs_mu!(θ[id], ŷ, z, id)
    end
end




