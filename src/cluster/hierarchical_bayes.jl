
 
function get_n_loc(z_frame::Vector{Int}, emitter_id::Int)
    n = 0 
    for z in z_frame
        if z == emitter_id
            n += 1
        end
    end
    return n
end

function log_likelihood_z(z::Allocations, prior_λ::Distributions.Distribution) 
    llz = 0.0
    for z_frame in z.idx
        for emitter_id in unique(z_frame)
            n_loc = get_n_loc(z_frame, emitter_id)
            llz += logpdf(prior_λ, n_loc)
        end
    end
end

