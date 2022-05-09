# SMLMBaGoL

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliasmlm.github.io/SMLMBaGoL.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliasmlm.github.io/SMLMBaGoL.jl/dev)
[![Build Status](https://github.com/juliasmlm/SMLMBaGoL.jl/workflows/CI/badge.svg)](https://github.com/juliasmlm/SMLMBaGoL.jl/actions)
[![Coverage](https://codecov.io/gh/**juliasmlm**/SMLMBaGoL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/juliasmlm/SMLMBaGoL.jl)

## Overview
SMLMBaGoL performs Bayesian Grouping of Localizations (BaGoL) on single-molecule localization microscopy (SMLM) data
organized in an SMLMData.SMLD structure (https://github.com/JuliaSMLM/SMLMData.jl).  Specifically, SMLMBaGoL groups
raw localizations into association with individual emitters, yielding higher precision localizations of those single 
emitters.  The algorithm(s) implemented are presented in https://doi.org/10.1101/752287.

## Interface
Once an SMLMData.SMLD2D structure is fully populated, the user only needs to run a single high-level method from the
package: `perform_BaGoL_analysis()`.  All fields of the SMLMData.SMLD2D structure should be populated with either 
meaningful values (e.g., for fields like `x`, `y`, `σ_x`, `σ_y`, and `framenum`, which the algorithm depends on) or
by placeholders with a meaningful size (e.g., fields like `bg` and `σ_bg`, which may not be available, should be set
to something like `smld.bg = zeros(Float64, length(smld.framenum))` and `σ_bg = fill(Inf64, length(smld.framenum))`).

BaGoL can be run on the fully populated `smld::SMLMData.SMLD2D` with default parameters.  The method will run the hierarchical BaGoL algorithm, which also explores the distribution of blinks per  

```
smld_MAPN, posterior_im, params = SMLMBaGoL.perform_BaGoL_analysis(smld)
```



The output `posterior_im` is the posterior distribution of
emitter positions explored by the algorithm stored as a matrix (which sums to 1.0, i.e., `sum(posterior_im)==1.0`).
The output `params` is a structure packaging the user-accessible parameters used for future reference.

For extended uses, see documentation. 


## Citation
https://doi.org/10.1101/752287
