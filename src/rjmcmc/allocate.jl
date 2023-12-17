

function has_allocation(z::Allocations, id::Int)
    return id in z.idx
end


function allocate!(z::Allocations, obs::Observations, θ::Params)
    k = length(θ)
    log_p = zeros(k)

    for i in 1:length(obs)   
        for j in 1:k
            log_p[j] = log_p_z_given_y(obs.ŷ[i], θ.emitters[j])
        end
        log_p .-= maximum(log_p)
        log_p .= exp.(log_p) .+ eps()
        log_p ./= sum(log_p) 
        z.idx[i] = rand(Categorical(log_p))
    end
end

function allocate(obs::Observations, θ::Params)
    z = Allocations(zeros(Int, length(obs.ŷ)))
    allocate!(z, obs, θ)
    return z
end
