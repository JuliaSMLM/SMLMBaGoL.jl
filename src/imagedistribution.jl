using SMLMData
using Distributions
using Base

# This file contains functions/methods useful for defining image distributions.

"""
    imdistrib, imsize = imagedistribution(smld::SMLMData.SMLD2D;
                                          mag::Real = 20.0, 
                                          nsigma::Real = 5.0,
                                          roi::Vector{<:Real} = [1.0; 1.0])

Generate an approximate emitter distribution from `smld` coordinates.

# Description
This method takes the coordinates in `smld`, prepares a Gaussian image from
them (each localization is placed in the image as a Gaussian with standard 
deviation given by the localization error), converts the Gaussian image into a
1D probability mass function, and finally prepares a distribution using the
Distributions package.

# Inputs
- `smld`: SMLD2D structure containing emitter localizations.
- `mag`: Approximate magnfication from data coordinates to SR coordinates. 
         (Default = 20.0)
- `nsigma`: Number of standard deviations from the localization coordinate at
            which we truncate the Gaussians in the image. (Default = 5.0)
- `roi`: Region of interest corresponding to `smld` localizations. 
         (pixels)(Default = [1.0; 1.0])
         ([ystart; xstart; yend; xend] or just [ystart; xstart])

# Outputs
- `imdstrib`: 1D distribution describing the approximate emitter distribution
              estimated from a Gaussian image of `smld` localizations.
- `imsize`: Size of the Gaussian image used to compute `imdistrib`.
"""
function imagedistribution(smld::SMLMData.SMLD2D;
                           mag::Real = 20.0, 
                           nsigma::Real = 5.0, 
                           roi::Vector{<:Real} = [1.0; 1.0])
    # Shift the coordinates in `smld` so that they are in the range defined by 
    # `roi` (i.e., `smld` and `roi` might represent a subregion of a full 
    # image, so we need to shift the coordinates before preparing the Gaussian 
    # image).
    smld = deepcopy(smld)
    smld.y .-= roi[1] - 1.0
    smld.x .-= roi[2] - 1.0

    # Prepare the Gaussian image and then compute the distribution.
    image = SMLMData.makegaussim(smld; mag=mag, nsigma=nsigma)
    
    return SMLMBaGoL.imagedistribution(image), collect(size(image))
end

"""
    imdistrib = imagedistribution(image::Matrix{<:Real})

Convert the provided `image` into a distribution.

# Inputs
- `image`: N-dimensional image stored as a matrix.

# Outputs
- `imdstrib`: 1D distribution describing the input `image`.
"""
function imagedistribution(image::Matrix{<:Real})
    # Normalize the image and prepare a distribution using the Distributions
    # package.
    pmf = image[:] ./ sum(image)
    
    return Distributions.DiscreteNonParametric(1:Base.length(pmf), pmf)
end

"""
    coords, sampleind = samplecoords2D(imdistrib::Distributions.Distribution, 
                                       imrows::Int, 
                                       nsamples::Int)

Sample grid coordinates from the distribution `imdistrib`.

# Inputs
- `imdistrib`: Distribution of coordinates "stacked" from 2D to a 1D 
               distribution.  E.g., a normalized gaussian image stacked into
               a column vector might define the PMF of this distribution.
- `imrows`: Number of rows in the grid.
- `nsamples`: Number of coordinates to sample.

# Outputs
- `coords`: Coordinates sampled from `imdistrib`. (nsamplesx2)([y x])
- `sampleind`: Linear index used to sample `imdistrib`.
"""
function samplecoords2D(imdistrib::Distributions.Distribution, 
                        imrows::Int, 
                        nsamples::Int)
    # Sample the `nsamples` positions.
    sampleind = Vector{Int}(undef, nsamples)
    coords = Matrix{Float32}(undef, nsamples, 2)
    for nn = 1:nsamples
        coords[nn, :], sampleind[nn] = SMLMBaGoL.samplecoords2D(imdistrib, imrows)
    end

    return coords, sampleind
end

"""
    coords, sampleind = samplecoords2D(imdistrib::Distributions.Distribution, 
                                       imrows::Int)

Sample grid coordinates from the distribution `imdistrib`.

# Inputs
- `imdistrib`: Distribution of coordinates "stacked" from 2D to a 1D 
               distribution.  E.g., a normalized gaussian image stacked into
               a column vector might define the PMF of this distribution.
- `imrows`: Number of rows in the grid.

# Outputs
- `coords`: Coordinates sampled from `imdistrib`. (2x1)([y; x])
- `sampleind`: Linear index used to sample `imdistrib`.
"""
function samplecoords2D(imdistrib::Distributions.Distribution, 
                        imrows::Int)
    # Sample a new emitter position, adding uniform random noise to ensure the
    # position can be anywhere within a pixel.
    sampleind = Distributions.rand(imdistrib)
    coords = [mod(sampleind, imrows) + 1.0; ceil(sampleind / imrows)]
    coords .+= Base.rand(2) .- 0.5

    return coords, sampleind
end