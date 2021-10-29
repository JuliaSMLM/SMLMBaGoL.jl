using SMLMData
using Distributions

"""
    imagedistribution(smld::SMLMData.SMLD2D, 
                      mag::Float64 = 20.0, 
                      nsigma::Float64 = 5.0)

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
"""
function imagedistribution(smld::SMLMData.SMLD2D, 
                           mag::Float64 = 20.0, 
                           nsigma::Float64 = 5.0)
    # Prepare the Gaussian image and then compute the distribution.
    image = makegaussim(smld, mag, nsigma)
    
    return SMLMBaGoL.imagedistribution(image)
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
    
    return Distributions.DiscreteNonParametric(1:length(pmf), pmf)
end

"""
    samplecoords(imdistrib::Distributions.Distribution, imrows::Int)

Sample grid coordinates from the distribution `imdistrib`.

# Inputs
-`imdistrib`: Distribution of coordinates "stacked" from 2D to a 1D 
              distribution.  E.g., a normalized gaussian image stacked into
              a column vector might define the PMF of this distribution.
-`imrows`: Number of rows in the grid.
"""
function samplecoords2D(imdistrib::Distributions.Distribution, imrows::Int)
    # Sample a new emitter position, adding uniform random noise to ensure the
    # position can be anywhere within a pixel.
    sampleind = Distributions.rand(imdistrib)
    coords = [mod(sampleind, imrows); ceil(sampleind / imrows)]
    coords .+= rand(2) .- 0.5

    return coords, sampleind
end

"""
    makegaussim(smld::SMLMData.SMLD2D, 
                mag::Float64, 
                nsigma::Float64 = 5.0)

Make a Gaussian image of the localizations in `smld`.

# Description
This function creates an image of the localizations in `smld` by placing a
Gaussian truncated to `nsigma` at the localization coordinates, where the 
standard deviation is given by `smld.σ_x` and `smld.σ_y`.  The image is then
normalized such that it sums to 1.0.

# Inputs
-`smld`: SMLMData.SMLD2D data structure containing localizations.
-`mag`: Approximate magnfication from data coordinates to SR coordinates. 
        (Default = 20.0)
-`nsigma`: Number of standard deviations from the localization coordinate at
           which we truncate the Gaussian. (Default = 5.0)
"""
function makegaussim(smld::SMLMData.SMLD2D, 
                     mag::Float64,
                     nsigma::Float64 = 5.0)
    # Loop through emitters and add them to our output Gaussian image.
    imagesize = Int.(round.(smld.datasize * mag))
    image = zeros(imagesize[1], imagesize[2])
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
            image[ii, jj] = image[ii, jj] +
                Distributions.pdf(distrib, ([ii; jj].-0.5) / mag .+ 0.5)
        end
    end
    image = image ./ sum(image)

    return image
end