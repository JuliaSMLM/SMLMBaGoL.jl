using Distributions
using Base

# This file contains type definitions for types used in SMLMBaGoL.


## Parameter structures.
"""
    SubregionParams

Abstract type defining subregion splitting parameter structures.
"""
abstract type SubregionParams
end

"""
    SubregionParams2D(roisize::Float64, roioverlap::Float64)

Structure of parameters defining subregion splitting of data.

# Description
The SubregionParams2D structure organizes parameters related to subregion
splitting of data.

# Fields
-`roisize`: Size of each subregion (region of interest, or ROI). 
            (Pixels)(Default = 5.0)
-`roioverlap`: Size of the overlap between subregions. (Default = 1.0)(Pixels)
"""
mutable struct SubregionParams2D <: SubregionParams
    roisize::Float64
    roioverlap::Float64
end
function SubregionParams2D()
    return SubregionParams2D(5.0, 1.0)
end

"""
    PreThreshParams

Abstract type defining preprocessing threshold parameter structures.
"""
abstract type PreThreshParams
end

"""
    PreThreshParams2D(maxsigmadev_photons::Float64, n_min::Int, r::Float64)

Structure of parameters defining preprocessing thresholds.

# Description
The PreThreshParams2D structure organizes parameters related to thresholds
applied to data during preprocessing.

# Fields
-`maxsigmadev_photons`: Maximum number of standard deviations above the mean
                        photons allowed. (Default = Inf64)
-`n_min`: Minimum number of nearest neighbors required within `r`. 
          (Pixels)(Default = 1)
-`r`: Distance defining the region within which `n_min` nearest neighbor 
      localizations must be present. (Pixels)(Default = 10.0)
"""
mutable struct PreThreshParams2D <: PreThreshParams
    maxsigmadev_photons::Float64
    n_min::Int
    r::Float64
end
function PreThreshParams2D()
    return PreThreshParams2D(Inf64, 0, 10.0)
end

"""
    PreclusterParams

Abstract type defining preclustering parameter structures.
"""
abstract type PreclusterParams
end

"""
    PreclusterParams2D(maxdist::Float64)

Structure of parameters defining preclustering of localizations in subregions.

# Description
The PreclusterParams2D structure organizes parameters related to the
preclustering of localizations within each subregion.

# Fields
-`maxdist`: Maximum distance from one localization to its nearest-neighbor 
            allowed in each precluster. (Default = 0.15)(Pixels)
"""
mutable struct PreclusterParams2D <: PreclusterParams
    maxdist::Float64
end
function PreclusterParams2D()
    return PreclusterParams2D(0.15)
end

"""
    MCParams

Abstract type defining RJMCMC parameters.
"""
abstract type MCParams
end

"""
    MCParams2D

RJMCMC type structure specific to BaGoL.

# Description
The MCParams structure organizes parameters, distributions, or other info.
related to RJMCMC.

# Fields
-`α`: Shape parameter of Gamma distribution for localizations per emitter.
-`β`: Rate parameter of Gamma distribtution for localizations per emitter.
-`σ_a`: Standard deviation of drift velocties. (pixels/frame)
-`n_chain`: Number of iterations for chain generation.
-`n_burnin`: Number of burn-in iterations for the chain.
-`p_jump`: Probability of making each jump type. 
-`srmag`: Magnification of the Gaussian SR image used to generate `imdistrib`
          with respect to the localization coordinate system.
-`nsigma`: Number of standard deviations out to which we add a Gaussian at each
           localization in the Gaussian image used to define `imdistrib`.
"""
mutable struct MCParams2D <: MCParams
    α::Float64
    β::Float64
    σ_a::Float64
    n_chain::Int
    n_burnin::Int
    p_jump::Vector{Float64}
    srmag::Float64
    nsigma::Float64
    MCParams2D() = new()
end

"""
    BaGoLParams

Abstract type defining RJMCMC parameters.
"""
abstract type BaGoLParams
end

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
end
function BaGoLParams2D()
    return BaGoLParams2D(SubregionParams2D(), 
                         PreThreshParams2D(), 
                         PreclusterParams2D(), 
                         MCParams2D())
end

"""
    Internals2D

Structure of parameters/distributions modified within the BaGoL analysis.

# Description
This structure organizes some distributions (e.g., priors) and parameters that
might be internally updated within a typical BaGoL analysis/are defined in 
terms of the other user set parameters in BaGoLParams2D.

# Fields
-`roi`: Region of interest for the current set of data. 
        ([ystart; xstart; yend; xend])
-`srimsize`: Resulting size of the Gaussian SR image.
-`area`: Area spanned by the localizations. (pixels^2)
-`jumpdistrib`: Distribution defining the jumps to be made. 
                (see jumpdistrib.jl)
-`imdistrib`: Approximate emitter distribution defined by a normalized Gaussian
              SR image of the localizations.
-`priork`: Prior distribution on the number of emitters.
-`priorz`: Probability of any given allocation of localizations to emitters.
           (only the probability is stored, assuming all allocations are 
           equally weighted, since returning a distribution seems unfeasible).
-`priorμ`: Prior distributions on the emitter positions. (Not currently used, as 
           I've instead been using `imdistrib`.) ([ydistrib; xdistrib])
-`priora`: Prior distribution on the drift velocities. 
           ([a_ydistrib; a_xdistrib])
"""
mutable struct Internals2D <: BaGoLParams
    roi::Vector{Float64}
    srimsize::Vector{Int}
    area::Float64
    jumpdistrib::Distributions.Distribution
    imdistrib::Distributions.Distribution
    priork::Distributions.Distribution
    priorz # we probably can't return the full distribution for this prior
    priorμ::Vector{Distributions.Distribution}
    priora::Vector{Distributions.Distribution}
    Internals2D() = new()
end

## Data structures.

"""
    State

Abstract type defining a Markov chain state.
"""
abstract type State
end

"""
    BaGoLState2D(k::Int,
                 z::Vector{Int},
                 μ::Matrix{Float64},
                 a::Matrix{Float64})

Structure of data pertaining to a state in a Markov chain.

# Description
This structure organizes some data retained during RJMCMC.

# Fields
-`k`: Number of emitters in the state.
-`z`: Allocations of localizations to the `k` emitters.
-`μ`: Positions of the `k` emitters. ([y x])
-`a`: Drift velocities of the `k` emitters. ([a_y a_x])
"""
mutable struct BaGoLState2D <: State
    k::Int
    z::Vector{Int}
    μ::Matrix{Float64}
    a::Matrix{Float64}
end
BaGoLState2D() = SMLMBaGoL.BaGoLState2D(1,
    Vector{Int}(undef, 1),
    Matrix{Float64}(undef, 2, 1),
    Matrix{Float64}(undef, 2, 1))

"""
    MarkovChain

Abstract type defining a Markov chain.
"""
abstract type MarkovChain
end

"""
    BaGoLChain2D(states::Vector{SMLMBaGoL.BaGoLState2D},
                 accept::Vector{Bool},
                 n::Int)

Structure of states forming a Markov chain.

# Description
This structure organizes some data retained during RJMCMC.

# Fields
-`states`: States of the chain at each iteration.
-`accepted`: Jump acceptance at each iteration.
-`n`: Number of states in the chain.
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
function BaGoLChain2D(state::SMLMBaGoL.BaGoLState2D, accepted::Bool = true)
    # Initialize a chain structure with the given state.
    return BaGoLChain2D([state], [accepted], 1)
end