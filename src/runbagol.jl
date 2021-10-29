using SMLMData

function runbagol!(smld::SMLMData.SMLD2D, params::BaGoLParams)
    # Split the data into subregions.
    smld_subregions, _, _ = SMLMBaGoL.gensubregions(smld, 
        params.subregion.roisize, params.subregion.roioverlap)

    # Remove outlier localizations.
    SMLMBaGoL.removeoutliers!(smld_subregions, params.prethresholds)

    # Perform hierarchical clustering on the subregions.
    smld_preclustered = SMLMBaGoL.precluster_hierarchical.(smld_subregions, 
        params.preclustering.maxdist)

    # Prepare some distributions (e.g., priors).
    nloc = SMLMData.length(smld)
    params.mcparams.priork = SMLMBaGoL.prior_kemitters(nloc, 
        params.mcparams.α, params.mcparams.β)
    k = Int.(ceil(nloc / (params.mcparams.α*params.mcparams.β)))
    params.mcparams.priorz = SMLMBaGoL.prior_allocations(nloc, k)
    params.mcparams.priorμ = SMLMBaGoL.prior_positions(smld.datasize)
    params.mcparams.priora = SMLMBaGoL.prior_drift(params.mcparams.σ_a *
                                                   [1.0; 1.0])
    params.mcparams.imdistrib = SMLMBaGoL.imagedistribution(smld, 
        params.mcparams.srmag, params.mcparams.nsigma)

end