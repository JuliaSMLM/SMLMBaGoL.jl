
function find_mapn(chain::RJMCMC_Chain)
    n_vec = length.(chain.states)
    n_map = mode(n_vec)
    return n_map, n_vec
end

function find_mapn_ref_state(chain::RJMCMC_Chain)
    n_map,  = find_mapn(chain)

    # build a histogram image of mapn subchain

    # find how many states have n_map emitters
    n_states = 0
    for state in chain.states
        if length(state) == n_map
            n_states += 1
        end
    end

    # make a list of x and y values for each state
    x = zeros(n_states * n_map)
    y = zeros(n_states * n_map)
    cnt = 0
    for state in chain.states
        if length(state) != n_map
            continue
        end
        cnt += 1
        for (i, emitter) in enumerate(state.emitters)
            x[(cnt-1)*n_map+i] = emitter.x
            y[(cnt-1)*n_map+i] = emitter.y
        end
    end

    # make a histogram of the x and y values
    hist_data = fit(Histogram, (x, y))
    w = hist_data.weights / sum(hist_data.weights)

    p_state = 0
    best_state = chain.states[1]
    for state in chain.states
        if length(state) != n_map
            continue
        end
        p = 1.0
        for emitter in state.emitters
            # find histogram bin for emitter
            xbin = findfirst(hist_data.edges[1] .<= emitter.x)
            ybin = findfirst(hist_data.edges[2] .<= emitter.y)
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
    for emitter in best_state.emitters
        push!(x, emitter.x)
        push!(y, emitter.y)
    end

    return best_state
end


function euclidean_distance(emitter1::Emitter2D, emitter2::Emitter2D)
    return sqrt((emitter1.x - emitter2.x)^2 + (emitter1.y - emitter2.y)^2)
end

function fill_cost_matrix!(cost_matrix, emitters1, emitters2)
    n = size(cost_matrix, 1)
    for i in 1:n
        for j in 1:n
            cost_matrix[i, j] = euclidean_distance(emitters1[i], emitters2[j])
        end
    end
    return nothing
end


function sort_mapn_chain!(chain_mapn::RJMCMC_Chain, ref_state)
    # use Hungarian algorithm to make sure the order of emitters is consistent
    # across states

    # make a list of x and y values for each state
    n_map = length(chain_mapn.states[1].emitters)
 
    # storage space for cost matrix
    cost_matrix = zeros(n_map, n_map)

    for state in chain_mapn.states
        # make a cost matrix
        fill_cost_matrix!(cost_matrix, ref_state.emitters, state.emitters)

        # use Hungarian algorithm to find the optimal assignment
        assignment, = hungarian(cost_matrix)

        # re-order the emitters in the state using in place permutation
        permute!(state.emitters, assignment)
    end
end

function sort_mapn_chain!(chain_mapn::RJMCMC_Chain; n_iterate::Int=2)
    # initial reference state
    ref_state = deepcopy(chain_mapn.states[1])
    # initial sort 
    sort_mapn_chain!(chain_mapn, ref_state)

    n_map = length(chain_mapn.states[1].emitters)
    n_states = length(chain_mapn.states)

    for i in 2:n_iterate
        # build a new ref_state using the mean x and y values for each emitter
        for i in 1:n_map
            x = 0.0
            y = 0.0
            for state in chain_mapn.states
                x += state.emitters[i].x
                y += state.emitters[i].y
            end
            ref_state.emitters[i].x = x / n_states
            ref_state.emitters[i].y = y / n_states
        end

        # sort the chain_mapn using the ref_state
        sort_mapn_chain!(chain_mapn, ref_state)
    end
end

function get_mapn_emitters(chain_mapn_sorted::RJMCMC_Chain, obs::Observations)
    # get mean and standard deviation of x and y for each emitter across states
    # store in an Localizations
    n_map = length(chain_mapn_sorted.states[1].emitters)
    n_states = length(chain_mapn_sorted.states)
    
    # storing results in an localizations struct
    loc_type = typeof(obs.ŷ[1])

    # make vector of Localizations to store results
    mapn_coords = Vector{loc_type}(undef, n_map)

    # get number of fields in an emitter (should have method for this)
    fnames = fieldnames(typeof(chain_mapn_sorted.states[1].emitters[1]))
    n_fields = length(fnames)

    # loop over fields, then loop over emitters, then loop over states
    val_temp = zeros(n_states)
    coords = zeros(n_fields, n_map)
    σs = zeros(n_fields, n_map)

    for i in 1:n_fields
        for j in 1:n_map
            for k in 1:n_states
                val_temp[k] = getfield(chain_mapn_sorted.states[k].emitters[j], fnames[i])
            end
            coords[i, j] = mean(val_temp)
            σs[i, j] = std(val_temp)
            end
    end
    
    for i in 1:n_map
        mapn_coords[i] = loc_type(coords[:, i]..., σs[:, i]...)
    end

    return mapn_coords
end
