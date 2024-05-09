


function add!(post::Posterior2D, chain::RJMCMC_Chain)
    for state in chain.states
        for emitter in state.emitters
            x_idx = Int(round(emitter.x / post.pixelsize))
            y_idx = Int(round(emitter.y / post.pixelsize))
            # check if the emitter is within the image
            if x_idx > 0 && x_idx <= post.x_size && y_idx > 0 && y_idx <= post.y_size
                post.post_arr[y_idx, x_idx] += 1
            end
        end
    end
end
