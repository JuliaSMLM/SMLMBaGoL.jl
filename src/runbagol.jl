using SMLMData

"""
    chain, rois, roioverlap = runbagol(smld::SMLMData.SMLD2D;
        subregion_params::SMLMBaGoL.SubregionParams2D = SMLMBaGoL.SubregionParams2D(),
        prethresholds::SMLMBaGoL.PreThreshParams2D = SMLMBaGoL.PreThreshParams2D(),
        preclustering_params::SMLMBaGoL.PreclusterParams2D = SMLMBaGoL.PreclusterParams2D(),
        mcparams::SMLMBaGoL.MCParams2D = SMLMBaGoL.MCParams2D())

Perform a typical BaGoL analysis based on localizations in `smld`.

# Description
This function is intended to be a mid-level entry point for the SMLMBaGoL.jl
package.  Specifically, this function performs the standard BaGoL analysis 
with several outputs that may not be useful to the typical user, however in 
some instances they may be needed.  High-level entry points to SMLMBaGoL.jl
will typically use this function.  This method is intended for use when the
distribution of the blinks per emitter is known.  If that distribution is 
unknown, use runbagol(smld::SMLMData.SMLD2D, hbparams::SMLMBaGoL.HBParams2D)
defined below.

# Inputs
- `smld`: SMLD2D structure containing localizations.
- `subregion_params`: see typedefinitions.jl
- `prethresholds`: see typedefinitions.jl
- `preclustering_params`: see typedefinitions.jl
- `mcparams`: see typedefinitions.jl

# Outputs
- `chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
- `rois`: Subregion ROIs used. (Matrix{Vector{Float64}})
- `roioverlap`: ROI overlap used in subregion generation, which might be 
                modified from the user set value. (Float64)
"""
function runbagol(smld::SMLMData.SMLD2D;
    subregion_params::SMLMBaGoL.SubregionParams2D=SMLMBaGoL.SubregionParams2D(),
    prethresholds::SMLMBaGoL.PreThreshParams2D=SMLMBaGoL.PreThreshParams2D(),
    preclustering_params::SMLMBaGoL.PreclusterParams2D=SMLMBaGoL.PreclusterParams2D(),
    mcparams::SMLMBaGoL.MCParams2D=SMLMBaGoL.MCParams2D())

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

    # Isolate the valid portions of the chain.
    validchain = SMLMBaGoL.removeoverlap(chain, smld.datasize, rois, roioverlap)

    # Compute the MAPN result.
    mapnout = SMLMBaGoL.mapn(validchain)

    # Package the parameters into a single structure for user reference.
    params = SMLMBaGoL.BaGoLParams2D(subregion_params, 
        prethresholds,
        preclustering_params,
        mcparams,
        SMLMBaGoL.HBParams2D(; nsamples=mcparams.n_burnin+mcparams.n_chain))

    return chain, validchain, mapnout, rois, roioverlap, params
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
This function is intended to be a mid-level entry point for the SMLMBaGoL.jl
package.  Specifically, this function performs the standard hierarchical Bayes
BaGoL analysis with several outputs that may not be useful to the typical user,
however in some instances they may be needed.  High-level entry points to 
SMLMBaGoL.jl will typically use this function.

# Inputs
- `smld`: SMLD2D structure containing localizations.
- `hbparams`: see typedefinitions.jl
- `subregion_params`: see typedefinitions.jl
- `prethresholds`: see typedefinitions.jl
- `preclustering_params`: see typedefinitions.jl
- `mcparams`: see typedefinitions.jl

# Outputs
- `chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
- `λchain`: An array of the λ parameters that were used for each hierarchical 
            sample.
- `rois`: Subregion ROIs used. (Matrix{Vector{Float64}})
- `roioverlap`: ROI overlap used in subregion generation, which might be 
                modified from the user set value. (Float64)
"""
function runbagol(smld::SMLMData.SMLD2D, hbparams::SMLMBaGoL.HBParams2D;
    subregion_params::SMLMBaGoL.SubregionParams2D=SMLMBaGoL.SubregionParams2D(),
    prethresholds::SMLMBaGoL.PreThreshParams2D=SMLMBaGoL.PreThreshParams2D(),
    preclustering_params::SMLMBaGoL.PreclusterParams2D=SMLMBaGoL.PreclusterParams2D(),
    mcparams::SMLMBaGoL.MCParams2D=SMLMBaGoL.MCParams2D())

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
    chain, λchain = SMLMBaGoL.runRJMCMC(smld_preclustered, rois, roioverlap, mcparams, hbparams)

    # Isolate the valid portions of the chain.
    validchain = SMLMBaGoL.removeoverlap(chain, smld.datasize, rois, roioverlap)

    # Compute the MAPN result.
    mapnout = SMLMBaGoL.mapn(validchain)

    # Package the parameters into a single structure for user reference.
    params = SMLMBaGoL.BaGoLParams2D(subregion_params,
        prethresholds,
        preclustering_params,
        mcparams,
        hbparams)

    return chain, validchain, mapnout, λchain, rois, roioverlap, params
end

"""
    smld_MAPN, posterior_im, params = perform_BaGoL_analysis(smld::SMLMData.SMLD2D;
        subregion_params::SMLMBaGoL.SubregionParams2D = SMLMBaGoL.SubregionParams2D(),
        prethresholds::SMLMBaGoL.PreThreshParams2D = SMLMBaGoL.PreThreshParams2D(),
        preclustering_params::SMLMBaGoL.PreclusterParams2D = SMLMBaGoL.PreclusterParams2D(),
        mcparams::SMLMBaGoL.MCParams2D = SMLMBaGoL.MCParams2D())

Perform a (user-friendly) hierarchical BaGoL analysis based on localizations in `smld`.

# Description
This function is intended to be the main entry for users of the SMLMBaGoL.jl
package when running it with an existing calibration for the number of blinks 
per emitter (i.e., when the Gamma distribution defining the blinks per emitter 
is defined in `mcparams` from calibration data).

# Inputs
- `smld`: SMLD2D structure containing localizations.
- `subregion_params`: see typedefinitions.jl
- `prethresholds`: see typedefinitions.jl
- `preclustering_params`: see typedefinitions.jl
- `mcparams`: see typedefinitions.jl
- `imagezoom`: zoom factor applied to output images. (Default = 20)

# Outputs
- `chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
- `λchain`: An array of the λ parameters that were used for each hierarchical 
            sample.
- `rois`: Subregion ROIs used. (Matrix{Vector{Float64}})
- `roioverlap`: ROI overlap used in subregion generation, which might be 
                modified from the user set value. (Float64)
"""
function perform_BaGoL_analysis(smld::SMLMData.SMLD2D;
    subregion_params::SMLMBaGoL.SubregionParams2D=SMLMBaGoL.SubregionParams2D(),
    prethresholds::SMLMBaGoL.PreThreshParams2D=SMLMBaGoL.PreThreshParams2D(),
    preclustering_params::SMLMBaGoL.PreclusterParams2D=SMLMBaGoL.PreclusterParams2D(),
    mcparams::SMLMBaGoL.MCParams2D=SMLMBaGoL.MCParams2D(),
    imagezoom=20)

    # Perform the standard hierarchical BaGoL analysis.
    chain, validchain, mapnout, rois, roioverlap, params = SMLMBaGoL.runbagol(smld;
        subregion_params=subregion_params,
        prethresholds=prethresholds,
        preclustering_params=preclustering_params,
        mcparams=mcparams)

    # Generate a MAPN SMLMData.SMLM2D() structure.
    smld_MAPN = SMLMBaGoL.convert_mapn_smld(mapnout;
        datasize=smld.datasize,
        nframes=smld.nframes,
        ndatasets=smld.ndatasets)

    # Prepare a posterior distribution image.
    _, _, μcat, _ = SMLMBaGoL.catfields(validchain)
    posterior_im = SMLMData.makehistim(μcat, smld.datasize, imagezoom)
    SMLMData.contraststretch!(posterior_im)

    return smld_MAPN, posterior_im, params
end

"""
    smld_MAPN, posterior_im, params = perform_BaGoL_analysis(
        smld::SMLMData.SMLD2D, hbparams::SMLMBaGoL.HBParams2D;
        subregion_params::SMLMBaGoL.SubregionParams2D = SMLMBaGoL.SubregionParams2D(),
        prethresholds::SMLMBaGoL.PreThreshParams2D = SMLMBaGoL.PreThreshParams2D(),
        preclustering_params::SMLMBaGoL.PreclusterParams2D = SMLMBaGoL.PreclusterParams2D(),
        mcparams::SMLMBaGoL.MCParams2D = SMLMBaGoL.MCParams2D())

Perform a (user-friendly) hierarchical BaGoL analysis based on localizations in `smld`.

# Description
This function is intended to be the main entry for users of the SMLMBaGoL.jl
package when running it in a hierarchical Bayes formalism, wherein we construct
certain prior distributions (for now, just the distribution of blinks per 
emitter) during the RJMCMC process.

# Inputs
- `smld`: SMLD2D structure containing localizations.
- `hbparams`: see typedefinitions.jl
- `subregion_params`: see typedefinitions.jl
- `prethresholds`: see typedefinitions.jl
- `preclustering_params`: see typedefinitions.jl
- `mcparams`: see typedefinitions.jl
- `imagezoom`: zoom factor applied to output images. (Default = 20)

# Outputs
- `chain`: An SMLMBaGoL.BaGoLChain2D RJMCMC chain.
- `λchain`: An array of the λ parameters that were used for each hierarchical 
            sample.
- `rois`: Subregion ROIs used. (Matrix{Vector{Float64}})
- `roioverlap`: ROI overlap used in subregion generation, which might be 
                modified from the user set value. (Float64)
"""
function perform_BaGoL_analysis(smld::SMLMData.SMLD2D, hbparams::SMLMBaGoL.HBParams2D;
    subregion_params::SMLMBaGoL.SubregionParams2D=SMLMBaGoL.SubregionParams2D(),
    prethresholds::SMLMBaGoL.PreThreshParams2D=SMLMBaGoL.PreThreshParams2D(),
    preclustering_params::SMLMBaGoL.PreclusterParams2D=SMLMBaGoL.PreclusterParams2D(),
    mcparams::SMLMBaGoL.MCParams2D=SMLMBaGoL.MCParams2D(),
    imagezoom=20)

    # Perform the standard hierarchical BaGoL analysis.
    chain, validchain, mapnout, λchain, rois, roioverlap, params = SMLMBaGoL.runbagol(smld, hbparams;
        subregion_params=subregion_params,
        prethresholds=prethresholds,
        preclustering_params=preclustering_params,
        mcparams=mcparams)

    # Generate a MAPN SMLMData.SMLM2D() structure.
    smld_MAPN = SMLMBaGoL.convert_mapn_smld(mapnout;
        datasize=smld.datasize,
        nframes=smld.nframes,
        ndatasets=smld.ndatasets)

    # Prepare a posterior distribution image.
    _, _, μcat, _ = SMLMBaGoL.catfields(validchain)
    posterior_im = SMLMData.makehistim(μcat, smld.datasize, imagezoom)
    SMLMData.contraststretch!(posterior_im)

    return smld_MAPN, posterior_im, params
end