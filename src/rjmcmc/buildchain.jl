
function take_jumps(θ::Params, z::Allocations, roi::RJMCMC_ROI, n_jumps::Int)
    verbose = false
    # θ = deepcopy(θ)
    θ = Params(θ.emitters)
    for i in 1:n_jumps
        # clean_params!(θ, z)
        # @info "jump number $i, length = $(length(θ)), z.idx = $(z.idx)"
        # Propose a new state
        jump_type = rand(roi.p_jump)
        if verbose
            @info "Jump type: $jump_type"
        end
        accepted = false
        if jump_type == 1
            # move emitters
            move!(θ, roi.obs, z, roi.prior_y)
            accepted = true
            if verbose
                @info "Move accepted"
            end
        elseif jump_type == 2
            # allocate
            allocate!(z, roi.obs, θ)
            accepted = true
            if verbose
                @info "Allocate accepted"
            end
        elseif jump_type == 3
            # add emitter
            θ_test, z_test, id = propose_add_emitter(θ, roi.obs, roi.prior_y)
            if !has_allocation(z_test, id)
                # then don't accept
                p_accept = 0.0
            else
                p_accept = p_accept_add(θ, θ_test, roi.obs, z, z_test, roi.prior_k, roi.prior_λ)
            end
            if rand() < p_accept
                accepted = true
                θ = θ_test
                z = z_test
                if verbose
                    @info "Add accepted"
                end
            end
        elseif jump_type == 4
            # remove emitter
            if length(θ) == 1
                # can't remove the last emitter
                continue
            end
            θ_test, z_test, id = propose_remove_emitter(θ, roi.obs, roi.prior_y)
            # If it has no allocations, then p_accept will be 1
            if !has_allocation(z, id)
                p_accept = 1.0
                # @info "No allocations for id = $id"
            else
                p_accept = p_accept_remove(θ, θ_test, roi.obs, z, z_test, roi.prior_k, roi.prior_λ)
            end
            if rand() < p_accept
                accepted = true
                θ = θ_test
                z = z_test
                if verbose
                    @info "Remove accepted"
                end
            end
        elseif jump_type == 5
            # split emitter
            θ_test, z_test, id = propose_split_emitter(θ, roi.obs, z)
            p_accept = p_accept_split(θ, θ_test, roi.obs, z, z_test, roi.prior_k, roi.prior_λ)
            if rand() < p_accept
                accepted = true
                θ = θ_test
                z = z_test
                if verbose
                    @info "Split accepted"
                end
            end
        elseif jump_type == 6
            # merge emitters
            if length(θ) == 1
                # can't merge when there's only one emitter
                continue
            end
            θ_test, z_test, ids = propose_merge_emitters(θ, roi.obs, z)
            p_accept = p_accept_merge(θ, θ_test, roi.obs, z, z_test, roi.prior_k, roi.prior_λ)
            if rand() < p_accept
                accepted = true
                θ = θ_test
                z = z_test
                if verbose
                    @info "Merge accepted"
                end
            end
        elseif jump_type == 7
            # clean: remove emitters with no allocations
            clean_params!(θ, z)
            accepted = true
        end
        # println("z.idx after jump_type $jump_type evals to $accepted = $(z.idx)")
    end
    return θ, z
end

function buildchain(roi::RJMCMC_ROI, n_burnin::Int, n_jumps::Int; θ::Union{Nothing,Params}=nothing)

    # Initialize the chain
    if isnothing(θ)
        # draw k from prior
        k = rand(roi.prior_k)
        if k == 0
            k = 1
        end
        k=1
        # generate a Params object with k emitters of the correct type (based on chain)
        θ = Params([roi.emitter_type(rand(roi.prior_y)) for i in 1:k])
        z = allocate(roi.obs, θ)
        # Clean up the parameters
        clean_params!(θ, z)
    end

    # Create a chain to store the states and allocations
    chain = RJMCMC_Chain(Vector{Params}(undef, n_jumps), Vector{Allocations}(undef, n_jumps))

    # Allocate
    z = allocate(roi.obs, θ)
   
    # Run a burn-in period without saving the states
    θ, z = take_jumps(θ, z, roi, n_burnin)
    
    # Initialize the chain
    chain.states[1] = deepcopy(θ)
    chain.allocations[1] = deepcopy(z)

    for i in 2:n_jumps
        θ, z = take_jumps(θ, z, roi, 1)
        chain.states[i] = θ
        chain.allocations[i] = deepcopy(z)
    end

    return chain
end
