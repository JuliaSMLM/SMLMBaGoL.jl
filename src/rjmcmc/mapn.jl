
function find_mapn(chain::RJMCMC_Chain)
    n_vec = length.(chain.states)
    n_map = mode(n_vec)
    return n_map
end

function find_mapn_ref_state(chain::RJMCMC_Chain)
    n_map = find_mapn(chain)

    # build a histogram image of mapn subchain

    # find how many states have n_map emitters
    n_states = 0
    for state in chain.states
        if length(state) == n_map
            n_states += 1
        end
    end
    x = zeros(n_states * n_map)
    y = zeros(n_states * n_map)
    for state in chain.states
        if length(state) != n_map
            continue
        end
        for (i, emitter) in enumerate(state.emitters)
            x[(n_states-1)*n_map+i] = emitter.x
            y[(n_states-1)*n_map+i] = emitter.y
        end
    end

    # make a histogram of the x and y values
    hist_data = fit(Histogram, (x, y))

    w = hist_data.weights / sum(hist_data.weights)

    p_state = 0
    best_state = 0
    for state in chain.states
        if length(state) != n_map
            continue
        end

        p = 1.0
        for emitter in state.emitters
            # find histogram bin for emitter
            xbin = findfirst(hist_data.edges[1] .<= emitter.x)
            ybin = findfirst(hist_data.edges[2] .<= emitter.y)

            if isnothing(xbin) || isnothing(ybin)
                println("Emitter out of bounds")
                println(emitter.x, " ", emitter.y)
                println("bin ranges: ", hist_data.edges[1][1], " ", hist_data.edges[1][end], " ", hist_data.edges[2][1], " ", hist_data.edges[2][end])
            end
            p *= w[xbin, ybin]
        end

        if p > p_state
            p_state = p
            best_state = state
        end
    end

    # get x,y for best state
    x = Float64[]
    y = Float64[]
    for emitter in chain.states[best_state].emitters
        push!(x, emitter.x)
        push!(y, emitter.y)
    end

    return best_state, x, y
end