# This script demonstrates usage of the SMLMBaGoL package.
using Revise
using SMLMBaGoL
using SMLMData
using SMLMSim
using Plots
using Images
using StatsBase

# Simulate some data.
smld_true, smld_model, smld = SMLMSim.sim(;
    ρ=0.1, # patterns / pixel^2
    σ_PSF=1.3, # pixels
    minphotons=50,
    ndatasets=10,
    nframes=1000,
    framerate=1.0, # set to 1.0 to keep things in camera units
    pattern=SMLMSim.Nmer2D(; n=3, d=1.0),
    molecule=SMLMSim.GenericFluor(; q=[0 0.5; 1e-3 0], γ=1e3), # 1 / frame 
    camera=SMLMSim.IdealCamera(; xpixels=16, ypixels=16, pixelsize=1.0) # set pixelsize to 1.0 to keep things in camera units
)

# Perform a standard BaGoL analysis.
smld_MAPN, posterior_im, params = SMLMBaGoL.perform_BaGoL_analysis(
    smld, SMLMBaGoL.HBParams2D(;
        nsamples=10, # chain length run before each resample of hierarchical parameters
        α=2.0, # shape parameter of Gamma distribution defining the prior on the hyperparameters η and γ.
        θ=10.0, # scale parameter of Gamma distribution defining the prior on the hyperparameters η and γ.
        α_scaling=3000.0, # scale factor used in sampling hyperparameters.
        nthinning=10, # number of thinning iterations made before each hierarchical sample is returned
        on=true); # use hierarchical Bayes to update blinks per emitter distribution if set to true
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
        η=1.0, # shape parameter of Gamma prior on blinks per emitter
        γ=(length(smld_model) / length(smld_true)) / 1.0, # scale parameter of Gamma prior on blinks per emitter
        imdistrib_mag=20.0, # magnification factor used for internal image distributions
        nsigma=5.0, # num. of st. devs. out to which we plot locs. in image distributions
        p_jump=[1.0; 1.0; 1.0; 1.0] / 4.0, # jump probabilities: [move; reallocate; birth; death]
        n_burnin=8000, # number of burn-in iterations in RJMCMC
        n_chain=2000), # length of chain post burn-in
    imagezoom=100.0) # zoom factor for output posterior image
Images.save("posterior_im.png", SMLMData.contraststretch(posterior_im))

# Make an overlay SR image of input and MAPN result.
srmag = 100.0
circleim_mapn = SMLMData.makecircleim(smld_MAPN, srmag)
SMLMData.contraststretch!(circleim_mapn)
circleim_in = SMLMData.makecircleim(smld, srmag)
SMLMData.contraststretch!(circleim_in)
overlay_im = RGB.(circleim_in, circleim_mapn, circleim_in)
Images.save("mapn_raw_overlay.png", overlay_im)
