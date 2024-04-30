using Distributions
using Base

# This file contains type definitions for types used in SMLMBaGoL.


## Parameter structures.
"""
    SubregionParams

Abstract type defining subregion splitting parameter structures.
"""
abstract type SubregionParams end

"""
    SubregionParams2D(roisize::Real, roioverlap::Real, on::Bool)

Structure of parameters defining subregion splitting of data.

# Description
The SubregionParams2D structure organizes parameters related to subregion
splitting of data.

# Fields
- `roisize`: Size of each subregion (region of interest, or ROI). 
            (Pixels)(Default = 5.0)
- `roioverlap`: Size of the overlap between subregions. (Default = 1.0)(Pixels)
- `on`: Flag indicating data should be split into subregions when using
       runbagol() (Default = true)
"""
mutable struct SubregionParams2D <: SubregionParams
    roisize::Real
    roioverlap::Real
    on::Bool
end
function SubregionParams2D(;
    roisize::Real=5.0,
    roioverlap::Real=1.0,
    on::Bool=true)

    return SubregionParams2D(roisize, roioverlap, on)
end

"""
    PreThreshParams

Abstract type defining preprocessing threshold parameter structures.
"""
abstract type PreThreshParams end

"""
    PreThreshParams2D(maxsigmadev_photons::Real, n_min::Int, r::Real)

Structure of parameters defining preprocessing thresholds.

# Description
The PreThreshParams2D structure organizes parameters related to thresholds
applied to data during preprocessing.

# Fields
- `maxsigmadev_photons`: Maximum number of standard deviations above the mean
                        photons allowed. (Default = Inf64)
- `n_min`: Minimum number of nearest neighbors required within `r`. 
          (Pixels)(Default = 1)
- `r`: Distance defining the region within which `n_min` nearest neighbor 
      localizations must be present. (Pixels)(Default = 10.0)
"""
mutable struct PreThreshParams2D <: PreThreshParams
    maxsigmadev_photons::Real
    n_min::Int
    r::Real
end
function PreThreshParams2D(;
    maxsigmadev_photons::Real=Inf64,
    n_min::Int=0,
    r::Real=10.0)

    return PreThreshParams2D(maxsigmadev_photons, n_min, r)
end

"""
    PreclusterParams

Abstract type defining preclustering parameter structures.
"""
abstract type PreclusterParams end

"""
    PreclusterParams2D(maxdist::Real, on::Bool)

Structure of parameters defining preclustering of localizations in subregions.

# Description
The PreclusterParams2D structure organizes parameters related to the
preclustering of localizations within each subregion.

# Fields
- `maxdist`: Maximum distance from one localization to its nearest-neighbor 
            allowed in each precluster. (Default = 0.15)(Pixels)
- `on`: Flag indicating pre-clustering should be applied when using runbagol().
       (Default = false)
"""
mutable struct PreclusterParams2D <: PreclusterParams
    maxdist::Real
    on::Bool
end
function PreclusterParams2D(;
    maxdist::Real=0.15,
    on::Bool=false)

    return PreclusterParams2D(maxdist, on)
end

"""
    MCParams

Abstract type defining RJMCMC parameters.
"""
abstract type MCParams end

"""
    MCParams2D

RJMCMC type structure specific to BaGoL.

# Description
The MCParams structure organizes parameters, distributions, or other info.
related to RJMCMC.

# Fields
- `η`: Shape parameter of Gamma distribtution for blinks per emitter.
- `γ`: Scale parameter of Gamma distribution for blinks per emitter.
- `σ_a`: Standard deviation of drift velocties. (pixels/frame)
- `n_chain`: Number of iterations for chain generation.
- `n_burnin`: Number of burn-in iterations for the chain.
- `p_jump`: Probability of making each jump type. 
            [move; reallocate; birth; death]
- `imdistrib_mag`: Magnification of the Gaussian SR image used to generate 
                   `imdistrib` with respect to the localization coordinate 
                   system.
- `nsigma`: Number of standard deviations out to which we add a Gaussian at each
            localization in the Gaussian image used to define `imdistrib`.
"""
mutable struct MCParams2D <: MCParams
    η::Real
    γ::Real
    σ_a::Real
    n_chain::Int
    n_burnin::Int
    p_jump::Vector{<:Real}
    imdistrib_mag::Real
    nsigma::Real
end
function MCParams2D(;
    η::Real=1.0,
    γ::Real=1.0,
    σ_a::Real=0.0,
    n_chain::Int=2000,
    n_burnin::Int=3000,
    p_jump::Vector{<:Real}=[0.25; 0.25; 0.25; 0.25],
    imdistrib_mag::Real=20.0,
    nsigma::Real=5.0)

    return MCParams2D(η, γ, σ_a, n_chain, n_burnin, p_jump, imdistrib_mag, nsigma)
end

"""
    HBParams

Abstract type defining hierarchical BaGoL parameters.
"""
abstract type HBParams end

"""
    HBParams2D

Parameters related to 2D hierarchical BaGoL analysis.

# Description
The HBParams2D structure organizes parameters, distributions, or other info.
related to hierarchical BaGoL analysis.

# Fields
- `nsamples`: Length of chains ran before resampling the hyperparameters.
              (e.g., if nsamples=10 and mcparams.n_chain=100, we'll make
              a total of 9 meaningful samples of of the hyperparameters).
- `α`: Shape parameter of Gamma distribution defining the prior on the 
       hyperparameters η and γ.
- `θ`: Scale parameter of Gamma distribtution defining the prior on the 
       hyperparameters η and γ.
- `α_scaling`: Scale factor used in sampling hyperparameters.
- `nthinning`: Number of thinning iterations (i.e., hyperparameters are sampled
              `nthinning` times before returning a value for the chain).
- `on`: If true, run hierarchical Bayes.
"""
mutable struct HBParams2D <: HBParams
    nsamples::Int
    α::Real
    θ::Real
    α_scaling::Real
    nthinning::Int
    on::Bool
end
function HBParams2D(;
    nsamples::Int=10,
    α::Real=2.0,
    θ::Real=10.0,
    α_scaling::Real=3000.0,
    nthinning::Int=5,
    on::Bool=true)

    return HBParams2D(nsamples, α, θ, α_scaling, nthinning, on)
end

"""
    BaGoLParams

Abstract type defining RJMCMC parameters.
"""
abstract type BaGoLParams end

"""
    BaGoLParams2D

Structure of user modified parameters defining the BaGoL workflow.

# Description
This structure organizes the parameter structures used in a typical BaGoL
analysis.  The intention is that this structure plus the data represents a 
complete description of the BaGoL analyses/results.  Other 
parameters used internally will be defined in terms of these parameters and the
data.
"""
mutable struct BaGoLParams2D <: BaGoLParams
    subregion::SubregionParams
    prethresholds::PreThreshParams
    preclustering::PreclusterParams
    mcparams::MCParams
    hbparams::HBParams
end
function BaGoLParams2D()
    return BaGoLParams2D(SubregionParams2D(),
        PreThreshParams2D(),
        PreclusterParams2D(),
        MCParams2D(),
        HBParams2D())
end

"""
    Internals2D

Structure of parameters/distributions modified within the BaGoL analysis.

# Description
This structure organizes some distributions (e.g., priors) and parameters that
might be internally updated within a typical BaGoL analysis/are defined in 
terms of the other user set parameters in BaGoLParams2D.

# Fields
- `roi`: Region of interest for the current set of data. 
         ([ystart; xstart; yend; xend])
- `srimsize`: Resulting size of the Gaussian SR image.
- `area`: Area spanned by the localizations. (pixels^2)
- `jumpdistrib`: Distribution defining the jumps to be made. 
                 (see jumpdistrib.jl)
- `imdistrib`: Approximate emitter distribution defined by a normalized Gaussian
               SR image of the localizations.
- `priork`: Prior distribution on the number of emitters.
- `priorz`: Probability of any given allocation of localizations to emitters.
            (only the probability is stored, assuming all allocations are 
            equally weighted, since returning a distribution seems unfeasible).
- `priorμ`: Prior distributions on the emitter positions. (Not currently used, as 
            I've instead been using `imdistrib`.) ([ydistrib; xdistrib])
- `priora`: Prior distribution on the drift velocities. 
            ([a_ydistrib; a_xdistrib])
"""
mutable struct Internals2D <: BaGoLParams
    roi::Vector{<:Real}
    srimsize::Vector{Int}
    area::Real
    jumpdistrib::Distributions.Distribution
    imdistrib::Distributions.Distribution
    priork::Distributions.Distribution
    priorμ::Vector{Distributions.Distribution}
    priora::Vector{Distributions.Distribution}
    Internals2D() = new()
end

## Data structures.

"""
    State

Abstract type defining a Markov chain state.
"""
abstract type State end

"""
    BaGoLState2D(k::Int,
                 z::Vector{Int},
                 μ::Matrix{<:Real},
                 a::Matrix{<:Real})

Structure of data pertaining to a state in a Markov chain.

# Description
This structure organizes some data retained during RJMCMC.

# Fields
- `k`: Number of emitters in the state.
- `z`: Allocations of localizations to the `k` emitters.
- `μ`: Positions of the `k` emitters. ([y x])
- `a`: Drift velocities of the `k` emitters. ([a_y a_x])
"""
mutable struct BaGoLState2D <: State
    k::Int
    z::Vector{Int}
    μ::Matrix{<:Real}
    a::Matrix{<:Real}
end
BaGoLState2D() = SMLMBaGoL.BaGoLState2D(1,
    Vector{Int}(undef, 1),
    Matrix{Float32}(undef, 2, 1),
    Matrix{Float32}(undef, 2, 1))

"""
    MarkovChain

Abstract type defining a Markov chain.
"""
abstract type MarkovChain end

"""
    BaGoLChain2D(states::Vector{SMLMBaGoL.BaGoLState2D},
                 accept::Vector{Bool},
                 n::Int)

Structure of states forming a Markov chain.

# Description
This structure organizes some data retained during RJMCMC.

# Fields
- `states`: States of the chain at each iteration.
- `accepted`: Jump acceptance at each iteration.
- `n`: Number of states in the chain.
"""
mutable struct BaGoLChain2D <: MarkovChain
    states::Vector{SMLMBaGoL.BaGoLState2D}
    accepted::Vector{Bool}
    n::Int
end
function BaGoLChain2D()
    # Initialize an empty chain.
    return BaGoLChain2D(Vector{SMLMBaGoL.BaGoLState2D}(undef, 0),
        Vector{Bool}(undef, 0),
        0)
end
function BaGoLChain2D(n_chain::Int)
    # Initialize a chain structure of length `n_chain`.
    return BaGoLChain2D(Vector{SMLMBaGoL.BaGoLState2D}(undef, n_chain),
        Vector{Bool}(undef, n_chain),
        n_chain)
end
function BaGoLChain2D(state::SMLMBaGoL.BaGoLState2D, accepted::Bool=true)
    # Initialize a chain structure with the given state.
    return BaGoLChain2D([state], [accepted], 1)
end
function Base.length(chain::SMLMBaGoL.BaGoLChain2D)
    return Base.length(chain.states)
end