# SMLMBaGoL

## Overview
SMLMBaGoL performs Bayesian Grouping of Localizations (BaGoL) on single-molecule localization microscopy (SMLM) data
organized in an SMLMData.SMLD2D structure (https://github.com/JuliaSMLM/SMLMData.jl).  Specifically, SMLMBaGoL groups
raw localizations into association with individual emitters, yielding higher precision localizations of those single 
emitters.  The algorithm(s) implemented are presented in https://doi.org/10.1101/752287.

## Interface
Once an SMLMData.SMLD2D structure is fully populated, the user only needs to run a single high-level method from the
package: `perform_BaGoL_analysis()`.  All fields of the SMLMData.SMLD2D structure should be populated with either 
meaningful values (e.g., for fields like `x`, `y`, `σ_x`, `σ_y`, and `framenum`, which the algorithm depends on) or
by placeholders with a meaningful size (e.g., fields like `bg` and `σ_bg`, which may not be available, should be set
to something like `smld.bg = zeros(Float64, length(smld.framenum))` and `σ_bg = fill(Inf64, length(smld.framenum))`).

## Examples

BaGoL can be run on the fully populated `smld::SMLMData.SMLD2D` with default parameters as

### Unknown Localization per Emitter Distribution
The first method will run the hierarchical BaGoL algorithm, which also explores the distribution of blinks per 
emitter.  
```
smld_MAPN, posterior_im, params = SMLMBaGoL.perform_BaGoL_analysis(smld, SMLMBaGoL.HBParams2D())
```

### Known Localization per Emitter Distribution
```
smld_MAPN, posterior_im, params = SMLMBaGoL.perform_BaGoL_analysis(smld)
```

The second method will use the blinks per emitter distribution defined in 
The output `smld_MAPN` is a (partially populated) SMLMData.SMLD2D structure populated with the Maximum A
Posteriori Number of emitters (MAPN) emitter positions.  The output `posterior_im` is the posterior distribution of
emitter positions explored by the algorithm stored as a matrix (which sums to 1.0, i.e., `sum(posterior_im)==1.0`).
The output `params` is a structure packaging the user-accessible parameters used for future reference.

## Extended Interface 

Additional parameters can be passed as keyword arguments if needed.  If they are excluded, default settings are used
instead.  For example, when using the hierarchical BaGoL algorithm:

```
smld_MAPN, posterior_im, params = SMLMBaGoL.perform_BaGoL_analysis(
    smld, SMLMBaGoL.HBParams2D(;
        nsamples=10, # chain length run before each resample of hierarchical parameters
        α=2.0, # shape parameter of Gamma distribution defining the prior on the hyperparameters η and γ.
        θ=10.0, # scale parameter of Gamma distribution defining the prior on the hyperparameters η and γ.
        α_scaling=3000.0, # scale factor used in sampling hyperparameters.
        nthinning=5); # number of thinning iterations made before each hierarchical sample is returned
    subregion_params=SMLMBaGoL.SubregionParams2D(;
        roisize=2.0, # subregion size used if on=true below (pixels)
        roioverlap=0.5, # overlap between subregions (pixels)
        on=true), # split data into subregions and analyze separately if true
    prethresholds=SMLMBaGoL.PreThreshParams2D(;
        maxsigmadev_photons=1.0, # maximum st. devs. from mean photons allowed for localizations
        n_min=1, # minimum number of localizations with `r` allowed for localizations
        r=10.0), # separation threshold related to `n_min` (pixels)
    preclustering_params=SMLMBaGoL.PreclusterParams2D(;
        maxdist=0.15, # maximum distance allowed between localizations in same precluster
        on=false), # if true, preclustering localizations in each subregion
    mcparams=SMLMBaGoL.MCParams2D(;
        σ_a=0.0, # st. dev. of drift term
        η=2.0, # shape parameter of Gamma prior on blinks per emitter
        γ=(length(smld_model) / length(smld_true)) / 2.0, # scale parameter of Gamma prior on blinks per emitter
        imdistrib_mag=20.0, # magnification factor used for internal image distributions
        nsigma=5.0, # num. of st. devs. out to which we plot locs. in image distributions
        p_jump=[1.0; 1.0; 1.0; 1.0] / 4.0, # jump probabilities: [move; reallocate; birth; death]
        n_burnin=7000, # number of burn-in iterations in RJMCMC
        n_chain=3000), # length of chain post burn-in
    imagezoom=100.0) # zoom factor for output posterior image
```

Users needing further control of the algorithm or additional troubleshooting outputs (e.g., the hierarchical
parameters explored in the hierarchical BaGoL algorithm) should look at the mid-level calling function 
`SMLMBaGoL.runbagol()`, which operates similarly to `SMLMBaGoL.perform_BaGoL_analysis()`.







