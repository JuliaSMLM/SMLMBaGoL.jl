using SMLMData
using Distributions
using Base

# This file contains functions/methods useful for defining image distributions.

"""
    imdistrib, imsize = imagedistribution(smld::SMLMData.SMLD2D, 
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

# Inputs
-`smld`: SMLD2D structure containing emitter localizations.
-`mag`: Approximate magnfication from data coordinates to SR coordinates. 
        (Default = 20.0)
-`nsigma`: Number of standard deviations from the localization coordinate at
           which we truncate the Gaussians in the image. (Default = 5.0)
-`roi`: Region of interest corresponding to `smld` localizations. 
        (pixels)(Default = [1.0; 1.0])
        ([ystart; xstart; yend; xend] or just [ystart; xstart])

# Outputs
-`imdstrib`: 1D distribution describing the approximate emitter distribution
             estimated from a Gaussian image of `smld` localizations.
-`imsize`: Size of the Gaussian image used to compute `imdistrib`.
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
    imdistrib = imagedistribution(image::Matrix{Float64})

Convert the provided `image` into a distribution.

# Inputs
-`image`: N-dimensional image stored as a matrix.

# Outputs
-`imdstrib`: 1D distribution describing the input `image`.
"""
function imagedistribution(image::Matrix{Float64})
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
-`imdistrib`: Distribution of coordinates "stacked" from 2D to a 1D 
              distribution.  E.g., a normalized gaussian image stacked into
              a column vector might define the PMF of this distribution.
-`imrows`: Number of rows in the grid.
-`nsamples`: Number of coordinates to sample.

# Outputs
-`coords`: Coordinates sampled from `imdistrib`. (nsamplesx2)([y x])
-`sampleind`: Linear index used to sample `imdistrib`.
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
    coords, sampleind = samplecoords2D(imdistrib::Distributions.Distribution, 
                                       imrows::Int)

Sample grid coordinates from the distribution `imdistrib`.

# Inputs
-`imdistrib`: Distribution of coordinates "stacked" from 2D to a 1D 
              distribution.  E.g., a normalized gaussian image stacked into
              a column vector might define the PMF of this distribution.
-`imrows`: Number of rows in the grid.

# Outputs
-`coords`: Coordinates sampled from `imdistrib`. (2x1)([y; x])
-`sampleind`: Linear index used to sample `imdistrib`.
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
    image = makegaussim(μ::Matrix{Float64},
                        σ_μ::Matrix{Float64}, 
                        photons::Vector{Float64},
                        datasize::Vector{Int},
                        mag::Float64 = 20.0, 
                        nsigma::Float64 = 5.0)

Make a Gaussian image of the localizations in `μ`.

# Description
This function creates an image of the localizations defined by `μ` and `σ_μ`
in which Gaussians with standard deviations `σ_μ` are added at positions `μ`.

# Inputs
-`μ`: Positions of the Gaussians. (pixels)([y x])
-`σ_μ`: Standard errors of `μ` estimates. (pixels)([y x])
-`photons`: Photons attributed to each row of `μ`.
-`datasize`: Size of the region of data collection. (pixels)([y; x])
-`mag`: Approximate magnfication from data coordinates to SR coordinates. 
        (Default = 20.0)
-`nsigma`: Number of standard deviations from the localization coordinate at
           which we truncate the Gaussian. (Default = 5.0)

# Outputs
-`image`: Matrix{Float64} Gaussian image in which each localization in `smld`
          is plotted as a Gaussian.
"""
function makegaussim(μ::Matrix{Float64},
                     σ_μ::Matrix{Float64}, 
                     photons::Vector{Float64},
                     datasize::Vector{Int},
                     mag::Float64 = 20.0,
                     nsigma::Float64 = 5.0)
    # Loop through emitters and add them to our output Gaussian image.
    imagesize = Int.(round.(datasize * mag))
    image = zeros(Float64, imagesize[1], imagesize[2])
    for nn = 1:size(μ, 1)
        # Prepare a normal distribution for this emitter.
        if !all(σ_μ[nn, :] .> 0.0)
            continue
        end
        Σ = [σ_μ[nn, 1]^2 0.0; 0.0 σ_μ[nn, 2]^2]
        distrib = Distributions.MvNormal(μ[nn, :], Σ)
        
        # Loop through pixels of the image and add this emitter.
        ystart = max(1, 
            Int(round(mag * (μ[nn, 1]-nsigma*σ_μ[nn, 2]-0.5))))
        yend = min(imagesize[1], 
            Int(round(mag * (μ[nn, 1]+nsigma*σ_μ[nn, 2]))))
        xstart = max(1, 
            Int(round(mag * (μ[nn, 2]-nsigma*σ_μ[nn, 1]-0.5))))
        xend = min(imagesize[2], 
            Int(round(mag * (μ[nn, 2]+nsigma*σ_μ[nn, 1]))))
        for ii = ystart:yend, jj = xstart:xend
            image[ii, jj] += photons[nn] * 
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
    if !isapprox(sum(image), 1.0)
        @warn "Image is non-normalizable!  Returning flat image."
        image = ones(Float64, imagesize[1], imagesize[2]) ./ prod(imagesize)
    end

    return image
end

"""
    image = makegaussim(smld::SMLMData.SMLD2D, 
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

# Outputs
-`image`: Matrix{Float64} Gaussian image in which each localization in `smld`
          is plotted as a Gaussian.
"""
function makegaussim(smld::SMLMData.SMLD2D,
                     mag::Float64 = 20.0,
                     nsigma::Float64 = 5.0)
    return SMLMBaGoL.makegaussim([smld.y smld.x], [smld.σ_y smld.σ_x], 
        smld.photons, smld.datasize, mag, nsigma)
end

"""
    image = makebinim(coords::Matrix{Float64}, 
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
    image = image ./ max(sum(image), 1.0)
    if !isapprox(sum(image), 1.0)
        @warn "Image is non-normalizable!  Returning flat image."
        image = ones(Float64, imagesize[1], imagesize[2]) ./ prod(imagesize)
    end

    return image
end

"""
    image = makebinim(smld::SMLMData.SMLD2D, mag::Float64 = 20.0)

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
function makebinim(smld::SMLMData.SMLD2D, mag::Float64 = 20.0)
    return makebinim([smld.y smld.x], smld.datasize, mag)
end

"""
    image = makehistim(coords::Matrix{Float64},
                       datasize::Vector{Int},
                       mag::Float64 = 20.0)

Make a histogram image of the localizations in `coords`.

# Description
This function creates an image of the localizations in `coords` by adding 1.0
to a pixel for each localization present within that pixel.  The final image
is then scaled so that it sums to 1.0.

# Inputs
-`coords`: Localization coordinates. ([y x])
-`datasize`: Size of the data image. ([ysize xsize])
-`mag`: Approximate magnfication from data coordinates to SR coordinates. 
        (Default = 20.0)

# Outputs
-`image`: Matrix{Float64} histogram image of localizations.
"""
function makehistim(coords::Matrix{Float64},
                    datasize::Vector{Int},
                    mag::Float64 = 20.0)
    # Loop through localizations and add them to our output image.
    imagesize = Int.(round.(datasize * mag))
    image = zeros(Float64, imagesize[1], imagesize[2])
    inds = max.(1.0, (coords.-0.5)*mag)
    inds[:, 1] = min.(imagesize[1], inds[:, 1])
    inds[:, 2] = min.(imagesize[2], inds[:, 2])
    inds = Int.(round.(inds))
    for nn = 1:size(coords, 1)
        image[inds[nn, 1], inds[nn, 2]] += 1.0
    end

    # Normalize the image to sum to 1.0.
    image = image ./ max(sum(image), 1.0)
    if !isapprox(sum(image), 1.0)
        @warn "Image is non-normalizable!  Returning flat image."
        image = ones(Float64, imagesize[1], imagesize[2]) ./ prod(imagesize)
    end

    return image
end

"""
    image = makehistim(smld::SMLMData.SMLD2D, mag::Float64 = 20.0)

Make a histogram image of the localizations in `smld`.

# Description
This function creates an image of the localizations in `smld` by adding 1.0
to a pixel for each localization present within that pixel.  The final image
is then scaled so that it sums to 1.0.

# Inputs
-`smld`: SMLMData.SMLD2D data structure containing localizations.
-`mag`: Approximate magnfication from data coordinates to SR coordinates. 
        (Default = 20.0)

# Outputs
-`image`: Matrix{Float64} histogram image of localizations.
"""
function makehistim(smld::SMLMData.SMLD2D, mag::Float64 = 20.0)
    return makehistim([smld.y smld.x], smld.datasize, mag)
end

"""
    image = makecircleim(coords::Matrix{Float64},
                         σ::Vector{Float64},
                         datasize::Vector{Int},
                         mag::Float64 = 20.0)

Make a circle image of the localizations in `coords`.

# Description
This function creates an image of the localizations in `coords` by adding a
circle centered at the locations `coords` with radii `σ`.

# Inputs
-`coords`: Localization coordinates. ([y x])
-`σ`: Standard error of localizations in `coords`. (nlocx1)
-`datasize`: Size of the data image. ([ysize xsize])
-`mag`: Approximate magnfication from data coordinates to SR coordinates. 
        (Default = 20.0)

# Outputs
-`image`: Matrix{Float64} histogram image of localizations.
"""
function makecircleim(coords::Matrix{Float64},
                      σ::Vector{Float64},
                      datasize::Vector{Int},
                      mag::Float64 = 20.0)
    # Rescale the coordinates based on `mag`.
    coords = mag*(coords.-0.5) .+ 0.5
    σ *= mag

    # Loop through localizations and add them to our output image.
    imagesize = Int.(round.(datasize * mag))
    image = zeros(Float64, imagesize[1], imagesize[2])
    for nn = 1:size(coords, 1)
        # If σ[nn] isn't positive, skip this localization.
        if !(σ[nn] > 0.0)
            continue
        end

        # Define the pixel locations that fall along the circle.
        # NOTE: The extra factor of 4 improves circle appearance.
        θ = range(0, 2*pi, length = max(4, Int(ceil(4 * (2*pi*σ[nn])))))
        rows = Int.(round.(coords[nn, 1] .+ σ[nn]*sin.(θ)))
        cols = Int.(round.(coords[nn, 2] .+ σ[nn]*cos.(θ)))
        validind = findall((rows.>=1) .* (rows.<imagesize[1]) .*
            (cols.>=1) .* (cols.<imagesize[1]))

        # Set the pixels of the output image to 1.0 wherever met by the circle.
        for ii in validind
            image[rows[ii], cols[ii]] = 1.0
        end
    end

    # Normalize the image to sum to 1.0.
    image = image ./ max(sum(image), 1.0)
    if !isapprox(sum(image), 1.0)
        @warn "Image is non-normalizable!  Returning flat image."
        image = ones(Float64, imagesize[1], imagesize[2]) ./ prod(imagesize)
    end

    return image
end

"""
    image = makecircleim(smld::SMLMData.SMLD2D, mag::Float64 = 20.0)

Make a circle image of the localizations in `smld`.

# Description
This function creates an image of the localizations in `smld` by adding a
circle for each localization.

# Inputs
-`smld`: SMLMData.SMLD2D data structure containing localizations.
-`mag`: Approximate magnfication from data coordinates to SR coordinates. 
        (Default = 20.0)

# Outputs
-`image`: Matrix{Float64} circle image of localizations.
"""
function makecircleim(smld::SMLMData.SMLD2D, mag::Float64 = 20.0)
    coords = [smld.y smld.x]
    σ = vec(mean([smld.σ_y smld.σ_x], dims = 2))
    return SMLMBaGoL.makecircleim(coords, σ, smld.datasize, mag)
end