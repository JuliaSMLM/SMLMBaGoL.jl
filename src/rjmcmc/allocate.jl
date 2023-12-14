


function allocate!(z::Allocations, obs::Observations, θ::Params)
    k = length(θ)
    p = zeros(k)

    for i in length(obs)   
        for j in 1:k
            p[j] = p_z_given_y(obs[i], θ[j])
        end
        z[i] = rand(Categorical(p))
    end
end

function allocate(obs::Observations, θ::Params)
    z = Allocations(zeros(Int, length(obs)))
    allocate!(z, obs, θ)
    return z
end
