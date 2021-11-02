using SMLMData

function runbagol!(smld::SMLMData.SMLD2D, params::SMLMBaGoL.BaGoLParams2D)
    # Split the data into subregions.
    smld_subregions, rois, _ = SMLMBaGoL.gensubregions(smld, 
        params.subregion.roisize, params.subregion.roioverlap)

    # Remove outlier localizations.
    SMLMBaGoL.removeoutliers!(smld_subregions, params.prethresholds)

    # Perform hierarchical clustering on the subregions.
    smld_preclustered = SMLMBaGoL.precluster_hierarchical.(smld_subregions, 
        params.preclustering.maxdist)

    # Perform RJMCMC on each precluster.
    chain = SMLMBaGoL.runRJMCMC!(smld_preclustered, rois, params.mcparams)

    return chain
end