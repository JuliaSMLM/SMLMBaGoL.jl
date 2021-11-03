using SMLMData
using Distributions
using Base

"""
    imagedistribution(smld::SMLMData.SMLD2D, 
                      mag::Float64 = 20.0, 
                      nsigma::Float64 = 5.0,
                      roi::Vector{Float64} = [1.0; 1.0])

Generate an approximate emitter distribution from `smld` coordinates.

# Description
This method takes the coordinates in `smld`, prepares a Gaussian image from
them (each localization is placed in the image as a Gaussian with standard 
deviation given by the localization error), converts the Gaussian image into a
1D probability mass function, and finally prepares a distribution using the
Distributions package.

# Inputs:
-`smld`: SMLD2D structure containing emitter localizations.
-`mag`: Approximate magnfication from data coordinates to SR coordinates. 
        (Default = 20.0)
-`nsigma`: Number of standard deviations from the localization coordinate at
           which we truncate the Gaussians in the image. (Default = 5.0)
-`roi`: Region of interest corresponding to `smld` localizations. 
        (pixels)(Default = [1.0; 1.0])
        ([ystart; xstart; yend; xend] or just [ystart; xstart])
"""
function imagedistribution(smld::SMLMData.SMLD2D, 
                           mag::Float64 = 20.0, 
                           nsigma::Float64 = 5.0, 
                           roi::Vector{Float64} = [1.0; 1.0])
    # Shift the coordinates in `smld` so that they are in the range defined by 
    # `roi` (i.e., `smld` and `roi` might represent a subregion of a full 
    # image, so we need to shift the coordinates before preparing the Gaussian 
    # image).
    smld = deepcopy(smld)
    smld.y .-= roi[1] - 1.0
    smld.x .-= roi[2] - 1.0

    # Prepare the Gaussian image and then compute the distribution.
    image = makegaussim(smld, mag, nsigma)
    
    return SMLMBaGoL.imagedistribution(image), collect(size(image))
end

"""
    imagedistribution(image::Matrix{Float64})

Convert the provided `image` into a distribution.

# Inputs:
-`image`: N-dimensional image stored as a matrix.
"""
function imagedistribution(image::Matrix{Float64})
    # Normalize the image and prepare a distribution using the Distributions
    # package.
    pmf = image[:] ./ sum(image)
    
    return Distributions.DiscreteNonParametric(1:Base.length(pmf), pmf)
end

"""
    samplecoords2D(imdistrib::Distributions.Distribution, 
                   imrows::Int, 
                   nsamples::Int)

Sample grid coordinates from the distribution `imdistrib`.

# Inputs
-`imdistrib`: Distribution of coordinates "stacked" from 2D to a 1D 
              distribution.  E.g., a normalized gaussian image stacked into
              a column vector might define the PMF of this distribution.
-`imrows`: Number of rows in the grid.
-`nsamples`: Number of coordinates to sample.
"""
function samplecoords2D(imdistrib::Distributions.Distribution, 
                        imrows::Int, 
                        nsamples::Int)
    # Sample the `nsamples` positions.
    sampleind = Vector{Int}(undef, nsamples)
    coords = Matrix{Float64}(undef, nsamples, 2)
    for nn = 1:nsamples
        coords[nn, :], sampleind[nn] = SMLMBaGoL.samplecoords2D(imdistrib, imrows)
    end

    return coords, sampleind
end

"""
    samplecoords2D(imdistrib::Distributions.Distribution, imrows::Int)

Sample grid coordinates from the distribution `imdistrib`.

# Inputs
-`imdistrib`: Distribution of coordinates "stacked" from 2D to a 1D 
              distribution.  E.g., a normalized gaussian image stacked into
              a column vector might define the PMF of this distribution.
-`imrows`: Number of rows in the grid.
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

"""
    makegaussim(smld::SMLMData.SMLD2D, 
                mag::Float64 = 20.0, 
                nsigma::Float64 = 5.0)

Make a Gaussian image of the localizations in `smld`.

# Description
This function creates an image of the localizations in `smld` by placing a
Gaussian truncated to `nsigma` at the localization coordinates, where the 
standard deviation is given by `smld.σ_x` and `smld.σ_y`.  The image is then
normalized such that it sums to 1.0.  The background signal is not accounted
for in this method.

# Inputs
-`smld`: SMLMData.SMLD2D data structure containing localizations.
-`mag`: Approximate magnfication from data coordinates to SR coordinates. 
        (Default = 20.0)
-`nsigma`: Number of standard deviations from the localization coordinate at
           which we truncate the Gaussian. (Default = 5.0)
"""
function makegaussim(smld::SMLMData.SMLD2D,
                     mag::Float64 = 20.0,
                     nsigma::Float64 = 5.0)
    # Loop through emitters and add them to our output Gaussian image.
    imagesize = Int.(round.(smld.datasize * mag))
    image = zeros(Float64, imagesize[1], imagesize[2])
    for nn = 1:SMLMData.length(smld)
        # Prepare a normal distribution for this emitter.
        distrib = Distributions.MvNormal([smld.y[nn]; smld.x[nn]], 
            [smld.σ_y[nn]^2 0.0; 0.0 smld.σ_x[nn]^2])
        
        # Loop through pixels of the image and add this emitter.
        ystart = max(1, 
            Int(round(mag * (smld.y[nn]-nsigma*smld.σ_y[nn]-0.5))))
        yend = min(imagesize[1], 
            Int(round(mag * (smld.y[nn]+nsigma*smld.σ_y[nn]))))
        xstart = max(1, 
            Int(round(mag * (smld.x[nn]-nsigma*smld.σ_x[nn]-0.5))))
        xend = min(imagesize[2], 
            Int(round(mag * (smld.x[nn]+nsigma*smld.σ_x[nn]))))
        for ii = ystart:yend, jj = xstart:xend
            image[ii, jj] += smld.photons[nn] * 
                Distributions.pdf(distrib, ([ii; jj].-0.5) / mag .+ 0.5)
        end
    end

    # Normalize the image to sum to 1.0.  If any image values are NaN, set them
    # to 0.0.
    nanpixels = isnan.(image)
    if any(nanpixels)
        image[nanpixles] .= 0.0
    end
    image = image ./ sum(image)

    return image
end

"""
    makebinim(coords::Matrix{Float64}, 
              datasize::Vector{Float64},
              mag::Float64 = 20.0)

Make a binary image of the localizations in `coords`.

# Description
This function creates an image of the localizations in `coords` by placing a
hot pixel at the coordinates of each localization.

# Inputs
-`coords`: Localization coordinates. ([y x])
-`datasize`: Size of the data image. ([ysize xsize])
-`mag`: Approximate magnfication from data coordinates to SR coordinates. 
        (Default = 20.0)

# Outputs
-`image`: Matrix{Float64} binary image of localizations.
"""
function makebinim(coords::Matrix{Float64},
                   datasize::Vector{Int},
                   mag::Float64 = 20.0)
    # Loop through localizations and add them to our output binary image.
    imagesize = Int.(round.(datasize * mag))
    image = zeros(Float64, imagesize[1], imagesize[2])
    inds = max.(1.0, (coords.-0.5)*mag)
    inds[:, 1] = min.(imagesize[1], inds[:, 1])
    inds[:, 2] = min.(imagesize[2], inds[:, 2])
    inds = Int.(round.(inds))
    for nn = 1:size(coords, 1)
        image[inds[nn, 1], inds[nn, 2]] = 1.0
    end

    # Normalize the image to sum to 1.0.
    image = image ./ min(sum(image), 1.0)

    return image
end

"""
    makebinim(smld::SMLMData.SMLD2D, 
              mag::Float64 = 20.0)

Make a binary image of the localizations in `smld`.

# Description
This function creates an image of the localizations in `smld` by placing a
hot pixel at the coordinates of each localization.

# Inputs
-`smld`: SMLMData.SMLD2D data structure containing localizations.
-`mag`: Approximate magnfication from data coordinates to SR coordinates. 
        (Default = 20.0)

# Outputs
-`image`: Matrix{Float64} binary image of localizations.
"""
function makebinim(smld::SMLMData.SMLD2D,
                   mag::Float64 = 20.0)
    coords = [smld.y smld.x]
    return makebinim(coords, smld.datasize, mag)
end