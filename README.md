# SMLMBaGoL

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliasmlm.github.io/SMLMBaGoL.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliasmlm.github.io/SMLMBaGoL.jl/dev)
[![Build Status](https://github.com/juliasmlm/SMLMBaGoL.jl/workflows/CI/badge.svg)](https://github.com/juliasmlm/SMLMBaGoL.jl/actions)
[![Coverage](https://codecov.io/gh/**juliasmlm**/SMLMBaGoL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/juliasmlm/SMLMBaGoL.jl)

## Overview
SMLMBaGoL performs Bayesian Grouping of Localizations (BaGoL) on single-molecule localization microscopy (SMLM) data
organized in an SMLMData.SMLD2D structure (https://github.com/JuliaSMLM/SMLMData.jl).  Specifically, SMLMBaGoL groups
raw localizations into association with individual emitters, yielding higher precision localizations of those single 
emitters.  The algorithm(s) implemented are presented in https://doi.org/10.1101/752287.

## Interface
Once an SMLMData.SMLD2D structure is fully populated, the user only needs to run a single high-level method from the
package: `runbagol()`.  All fields of the SMLMData.SMLD2D structure must be populated with either meaningful values 
(e.g., for fields like `x`, `y`, `σ_x`, `σ_y`, and `framenum`, which the algorithm depends on) or by placeholders 
with a meaningful size (e.g., fields like `bg` and `σ_bg`, which may not be available, should be set to something
like `smld.bg = zeros(Float64, length(smld.framenum))` and `σ_bg = fill(Inf64, length(smld.framenum))`).

The hierarchical algorithm (to be used when the localizations per emitter is unknown) can be run on the fully 
populated `smld::SMLMData.SMLD2D` with default parameters as

```
params = SMLMBaGoL.BaGoLParams2D()
chain, validchain, mapnout, λchain, rois, roioverlap = SMLMBaGoL.runbagol(smld, params.hbparams)
```

The output `chain` is an SMLMBaGoL.BaGoLChain2D structure containing the history of proposed states and their 
acceptance (see typedefinitions.jl for organization).  `validchain` contains portions of `chain` that are considered
valid (i.e., the portions of the chain that did not fall into the `roioverlap` region of the `rois`).  `mapnout` 
contains the maximum a posteriori probability of the number of emitters computed from `validchain` 
(see SMLMBaGoL.mapn()).  `λchain` contains the chain of hierarchical parameters that define the number of 
localizations per emitter.  The outputs `rois` and `roioverlap` define the sub-regions used when 
`params.subregion.on = true`.

If the localizations per emitter prior has been characterized (e.g., from calibration data), you can instead use

```
params = SMLMBaGoL.BaGoLParams2D()
params.mcparams.η = 5.0 # shape parameter of the Gamma distribution of localizations per emitter (from calibration)
params.mcparams.γ = 10.0 # scale parameter of the Gamma distribution (from calibration)
chain, validchain, mapnout, rois, roioverlap = SMLMBaGoL.runbagol(smld; mcparams = params.mcparams)
```

where the localizations per emitter prior is defined by `params.mcparams`.

In either usage, several additional keyword arguments define other user-accessible parameters (see 
typedefinitions.jl for descriptions of parameters and parameter structures):

```
params = SMLMBaGoL.BaGoLParams2D()
chain, validchain, mapnout, λchain, rois, roioverlap = SMLMBaGoL.runbagol(smld, params.hbparams; 
    subregion_params = params.subregion,
    prethresholds = params.prethresholds,
    preclustering_params = params.preclustering,
    mcparams = params.mcparams)
```

## Citation
https://doi.org/10.1101/752287
