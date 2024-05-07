

function rjmcmc(obs::SMLMBaGoL.Observations, prior_λ::Distributions.Distribution;
    n_burnin::Int=4000,
    n_jumps::Int=8000)

    prior_k = build_prior_k(obs, prior_λ)
    prior_y = build_prior_y(obs)
    
    ## RJMCMC
    p_jump = Categorical([1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7])
    roi = RJMCMC_ROI(obs, prior_y, prior_k, p_jump, Emitter2D, prior_λ)
    chain, z_chain = buildchain(roi, n_burnin, n_jumps)

    # extract mapn chain:
    chain_mapn = extract_mapn_chain(chain)
    best_state = find_mapn_ref_state(chain)
    sort_mapn_chain!(chain_mapn; n_iterate=3)
    mapn_coords = get_mapn_emitters(chain_mapn, obs)

    chain, z_chain, mapn_coords, chain_mapn, roi
end








