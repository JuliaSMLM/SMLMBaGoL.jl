using SMLMData

"""
    chain = runbagol(smld::SMLMData.SMLD2D, params::SMLMBaGoL.BaGoLParams2D)

Perform a typical BaGoL analysis based on localizations in `smld`.

# Description
This function is intended to be the main entry for users of the SMLMBaGoL.jl
package, meaning that it will run the standard analysis workflow for Bayesian 
Grouping of Localizations (BaGoL).

# Inputs
-`smld`: SMLD2D structure containing localizations.
-`params`: Structure of parameter structures used/updated throughout the BaGoL
           analysis.

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
-`rois`: Subregion ROIs used. (Matrix{Vector{Float64}})
-`roioverlap`: ROI overlap used in subregion generation. (Float64)
"""
function runbagol(smld::SMLMData.SMLD2D, params::SMLMBaGoL.BaGoLParams2D)
    # Split the data into subregions.
    if params.subregion.on
        roioverlap = deepcopy(params.subregion.roioverlap)
        smld_subregions, rois, _ = SMLMBaGoL.gensubregions(smld,
            params.subregion.roisize, roioverlap)
    else
        roioverlap = 0.0
        smld_subregions, rois, _ = SMLMBaGoL.gensubregions(smld,
            Float64.(maximum(smld.datasize)), roioverlap)
    end

    # Remove outlier localizations.
    SMLMBaGoL.removeoutliers!(smld_subregions, params.prethresholds)

    # Perform hierarchical clustering on the subregions.
    if params.preclustering.on
        smld_preclustered = SMLMBaGoL.precluster_hierarchical.(smld_subregions,
            params.preclustering.maxdist)
    else
        smld_preclustered = deepcopy(smld_subregions)
        for ii = 1:prod(size(smld_preclustered))
            smld_preclustered[ii].connectID =
                collect(1:Base.length(smld_preclustered[ii].framenum))
        end
    end

    # Perform RJMCMC on each precluster.
    chain = SMLMBaGoL.runRJMCMC(smld_preclustered, rois, params.mcparams)

    return chain, rois, roioverlap
end