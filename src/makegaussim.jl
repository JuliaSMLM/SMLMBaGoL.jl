using SMLMData
using Distributions

function makegaussim(smld::SMLMData.SMLD2D, 
                     pixelsize::Float64 = 5.0, 
                     nsigma::Float64 = 5.0)
    # Loop through emitters and add them to our output Gaussian image.
    imagesize = Int.(round.(pixelsize * smld.datasize))
    image = zeros(imagesize[1], imagesize[2])
    for nn = 1:SMLMData.length(smld)
        # Prepare a normal distribution for this emitter.
        distrib = Distributions.MvNormal([smld.y[nn]; smld.x[nn]], 
            [smld.σ_y[nn]^2 0.0; 0.0 smld.σ_x[nn]^2])
        
        # Loop through pixels of the image and add this emitter.
        ystart = max(1, Int(round(smld.y[nn]-nsigma*smld.σ_y[nn])))
        yend = min(imagesize[1],
            Int(round(pixelsize * (smld.y[nn]+nsigma*smld.σ_y[nn]))))
        xstart = max(1, Int(round(smld.x[nn]-nsigma*smld.σ_x[nn])))
        xend = min(imagesize[2],
            Int(round(pixelsize * (smld.x[nn]+nsigma*smld.σ_x[nn]))))
        for ii = ystart:yend, jj = xstart:xend
            image[ii, jj] = image[ii, jj] +
                Distributions.pdf(distrib, [ii; jj]/pixelsize .+ 0.5)
        end
    end

    return image
end