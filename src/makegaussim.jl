using SMLMData
using Distributions

"""
    makegaussim(smld::SMLMData.SMLD2D, 
                pxsize::Float64 = 0.1,
                pxsizeGauss::Float64 = 0.01,
                nsigma::Float64 = 5.0)

Make a Gaussian image of the localizations in `smld`.

# Description
This function creates an image of the localizations in `smld` by placing a
Gaussian truncated to `nsigma` at the localization coordinates, where the 
standard deviation is given by `smld.σ_x` and `smld.σ_y`.  The image is then
normalized such that it sums to 1.0.

# Inputs
-`smld`: SMLMData.SMLD2D data structure containing localizations.
-`pxsize`: Pixel size corresponding to the localizations in `smld`. 
           (micrometers)(Default = 0.1 micrometers)
-`pxsizeGauss`: Approximate pixel size of the output image. 
                (micrometers)(Default = 0.01 micrometers)
-`nsigma`: Number of standard deviations from the localization coordinate at
           which we truncate the Gaussian. (Default = 5.0)
"""
function makegaussim(smld::SMLMData.SMLD2D, 
                     pxsize::Float64 = 0.1,
                     pxsizeGauss::Float64 = 0.01,
                     nsigma::Float64 = 5.0)
    # Loop through emitters and add them to our output Gaussian image.
    mag = pxsize / pxsizeGauss
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