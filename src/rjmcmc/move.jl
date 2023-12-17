
function move!(θ::Params, obs::Observations, z::Allocations, prior_y::Distributions.Distribution)
    # Move all the emitters
    for id in eachindex(θ.emitters)
        move!(θ.emitters[id], obs, z, id, prior_y)
    end
end




