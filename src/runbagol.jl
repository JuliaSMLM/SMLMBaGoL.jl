using SMLMData

"""
    chain = runbagol(smld::SMLMData.SMLD2D;
        subregion_params::SMLMBaGoL.SubregionParams2D = SMLMBaGoL.SubregionParams2D(),
        prethresholds::SMLMBaGoL.PreThreshParams2D = SMLMBaGoL.PreThreshParams2D(),
        preclustering_params::SMLMBaGoL.PreclusterParams2D = SMLMBaGoL.PreclusterParams2D(),
        mcparams::SMLMBaGoL.MCParams2D = SMLMBaGoL.MCParams2D())

Perform a typical BaGoL analysis based on localizations in `smld`.

# Description
This function is intended to be the main entry for users of the SMLMBaGoL.jl
package, meaning that it will run the standard analysis workflow for Bayesian 
Grouping of Localizations (BaGoL).

# Inputs
-`smld`: SMLD2D structure containing localizations.
-`subregion_params`: see typedefinitions.jl
-`prethresholds`: see typedefinitions.jl
-`preclustering_params`: see typedefinitions.jl
-`mcparams`: see typedefinitions.jl

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
-`rois`: Subregion ROIs used. (Matrix{Vector{Float64}})
-`roioverlap`: ROI overlap used in subregion generation. (Float64)
"""
function runbagol(smld::SMLMData.SMLD2D;
    subregion_params::SMLMBaGoL.SubregionParams2D = SMLMBaGoL.SubregionParams2D(),
    prethresholds::SMLMBaGoL.PreThreshParams2D = SMLMBaGoL.PreThreshParams2D(),
    preclustering_params::SMLMBaGoL.PreclusterParams2D = SMLMBaGoL.PreclusterParams2D(),
    mcparams::SMLMBaGoL.MCParams2D = SMLMBaGoL.MCParams2D())

    # Split the data into subregions.
    if subregion_params.on
        roioverlap = deepcopy(subregion_params.roioverlap)
        smld_subregions, rois, _ = SMLMBaGoL.gensubregions(smld,
            subregion_params.roisize, roioverlap)
    else
        roioverlap = 0.0
        smld_subregions, rois, _ = SMLMBaGoL.gensubregions(smld,
            Float64.(maximum(smld.datasize)), roioverlap)
    end

    # Remove outlier localizations.
    SMLMBaGoL.removeoutliers!(smld_subregions, prethresholds)

    # Perform hierarchical clustering on the subregions.
    if preclustering_params.on
        smld_preclustered = SMLMBaGoL.precluster_hierarchical.(smld_subregions,
            preclustering_params.maxdist)
    else
        # If we aren't preclustering, we'll label each localization in each 
        # subregion with the same connectID.  Note that having the same 
        # connectID (1) for all subregions is okay since subregions are treated
        # separately.
        smld_preclustered = deepcopy(smld_subregions)
        for ii = 1:prod(size(smld_preclustered))
            smld_preclustered[ii].connectID =
                ones(Float64, length(smld_preclustered[ii].framenum))
        end
    end

    # Perform RJMCMC on each precluster.
    chain = SMLMBaGoL.runRJMCMC(smld_preclustered, rois, mcparams)

    return chain, rois, roioverlap
end

"""
    chain, λchain, rois, roioverlap = runbagol(
        smld::SMLMData.SMLD2D, hbparams::SMLMBaGoL.HBParams2D;
        subregion_params::SMLMBaGoL.SubregionParams2D = SMLMBaGoL.SubregionParams2D(),
        prethresholds::SMLMBaGoL.PreThreshParams2D = SMLMBaGoL.PreThreshParams2D(),
        preclustering_params::SMLMBaGoL.PreclusterParams2D = SMLMBaGoL.PreclusterParams2D(),
        mcparams::SMLMBaGoL.MCParams2D = SMLMBaGoL.MCParams2D())

Perform a hierarchical BaGoL analysis based on localizations in `smld`.

# Description
This function is intended to be the main entry for users of the SMLMBaGoL.jl
package when running it in a hierarchical Bayes formalism, wherein we construct
certain prior distributions (for now, just the distribution of blinks per 
emitter) during the RJMCMC process.

# Inputs
-`smld`: SMLD2D structure containing localizations.
-`hbparams`: see typedefinitions.jl
-`subregion_params`: see typedefinitions.jl
-`prethresholds`: see typedefinitions.jl
-`preclustering_params`: see typedefinitions.jl
-`mcparams`: see typedefinitions.jl

# Outputs
-`chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
-`λchain`: An array of the λ parameters that were used for each hierarchical 
           sample.
-`rois`: Subregion ROIs used. (Matrix{Vector{Float64}})
-`roioverlap`: ROI overlap used in subregion generation. (Float64)
"""
function runbagol(smld::SMLMData.SMLD2D, hbparams::SMLMBaGoL.HBParams2D;
    subregion_params::SMLMBaGoL.SubregionParams2D = SMLMBaGoL.SubregionParams2D(),
    prethresholds::SMLMBaGoL.PreThreshParams2D = SMLMBaGoL.PreThreshParams2D(),
    preclustering_params::SMLMBaGoL.PreclusterParams2D = SMLMBaGoL.PreclusterParams2D(),
    mcparams::SMLMBaGoL.MCParams2D = SMLMBaGoL.MCParams2D())

    # Split the data into subregions.
    if subregion_params.on
        roioverlap = deepcopy(subregion_params.roioverlap)
        smld_subregions, rois, _ = SMLMBaGoL.gensubregions(smld,
            subregion_params.roisize, roioverlap)
    else
        roioverlap = 0.0
        smld_subregions, rois, _ = SMLMBaGoL.gensubregions(smld,
            Float64.(maximum(smld.datasize)), roioverlap)
    end

    # Remove outlier localizations.
    SMLMBaGoL.removeoutliers!(smld_subregions, prethresholds)

    # Perform hierarchical clustering on the subregions.
    if preclustering_params.on
        smld_preclustered = SMLMBaGoL.precluster_hierarchical.(smld_subregions,
            preclustering_params.maxdist)
    else
        # If we aren't preclustering, we'll label each localization in each 
        # subregion with the same connectID.  Note that having the same 
        # connectID (1) for all subregions is okay since subregions are treated
        # separately.
        smld_preclustered = deepcopy(smld_subregions)
        for ii = 1:prod(size(smld_preclustered))
            smld_preclustered[ii].connectID = 
                ones(Float64, length(smld_preclustered[ii].framenum))
        end
    end

    # Perform RJMCMC on each precluster.
    chain, λchain = SMLMBaGoL.runRJMCMC(smld_preclustered, rois, mcparams, hbparams)

    return chain, λchain, rois, roioverlap
end