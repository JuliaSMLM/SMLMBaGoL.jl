"""
    smld_to_localizations(smld::SMLMSim.BasicSMLD) -> Vector{Localization2D{Float64}}

Convert an SMLMSim BasicSMLD structure to BaGoL Localization2D format.

This function extracts the localization uncertainties (σ_x, σ_y) directly from 
SMLMSim's Emitter2DFit objects, which contain realistic noise-based uncertainties 
from the apply_noise() step.

# Arguments
- `smld`: SMLMSim BasicSMLD structure containing Emitter2DFit objects with uncertainties

# Returns
- Vector of Localization2D suitable for BaGoL analysis

# Example
```julia
# Create SMLMSim simulation (returns noisy localizations by default)
smld = simulate_static_smlm(density=0.1, minphotons=200)

# Convert to BaGoL format (automatic - uses σ_x, σ_y from SMLMSim)
localizations = smld_to_localizations(smld)

# Run BaGoL analysis
chains = run_bagol(smld)  # Direct dispatch - conversion happens automatically
```
"""
function smld_to_localizations(smld::SMLMSim.BasicSMLD)
    localizations = Localization2D{Float64}[]
    
    for emitter in smld.emitters
        # Extract position and uncertainties from Emitter2DFit
        x = emitter.x
        y = emitter.y
        
        # Use the σ_x, σ_y uncertainties from SMLMSim (already calculated from photon noise)
        # These are the realistic localization precisions from apply_noise()
        σx = emitter.σ_x
        σy = emitter.σ_y
        
        # Use the frame number from the emitter
        frame = emitter.frame
        
        # Create localization with SMLMSim's calculated uncertainties
        loc = Localization2D{Float64}(x, y, σx, σy, frame)
        push!(localizations, loc)
    end
    
    return localizations
end

"""
    simulate_static_smlm(; density=0.1, σ_psf=0.13, minphotons=100, 
                          nframes=1000, framerate=100.0, ndims=2, 
                          zrange=[-0.5, 0.5], npixelsx=128, npixelsy=128, 
                          pixelsize=0.1, return_noisy=true) -> SMLMSim.BasicSMLD

Create a static SMLM simulation using SMLMSim with sensible defaults.

# Arguments
- `density`: Emitter density (emitters per unit area)
- `σ_psf`: PSF standard deviation in micrometers
- `minphotons`: Minimum photon count threshold
- `nframes`: Number of frames to simulate
- `framerate`: Frame rate in Hz
- `ndims`: Number of spatial dimensions (2 or 3)
- `zrange`: Z-range as [min, max] in micrometers for 3D
- `npixelsx`: Number of pixels in x direction (default: 64)
- `npixelsy`: Number of pixels in y direction (default: 32)
- `pixelsize`: Pixel size in micrometers (default: 0.1, giving 6.4μm × 3.2μm field)
- `return_noisy`: Return noisy localizations (true) or true positions (false)

# Returns
- Single BasicSMLD structure ready for BaGoL analysis

# Example
```julia
# Quick simulation with defaults (6.4μm × 3.2μm field)
smld = simulate_static_smlm()

# Custom simulation with larger field (25.6μm × 25.6μm)
smld = simulate_static_smlm(npixelsx=256, npixelsy=256, density=0.2)

# Custom pixel size for different field size (3.2μm × 1.6μm)
smld = simulate_static_smlm(pixelsize=0.05)

# Run BaGoL directly
chains = run_bagol(smld; n_iterations=5000)
```
"""
function simulate_static_smlm(; density=0.1, 
                               σ_psf=0.13, 
                               minphotons=100,
                               nframes=1000, 
                               framerate=100.0, 
                               ndims=2, 
                               zrange=[-0.5, 0.5],
                               npixelsx=64,
                               npixelsy=32,
                               pixelsize=0.1,
                               return_noisy=true)
    
    # Create SMLMSim parameters
    params = SMLMSim.StaticSMLMParams(
        density=density,
        σ_psf=σ_psf,
        minphotons=minphotons,
        ndatasets=1,  # Only need one dataset
        nframes=nframes,
        framerate=framerate,
        ndims=ndims,
        zrange=zrange
    )
    
    # Create camera to define simulation field size
    camera = SMLMSim.IdealCamera(npixelsx, npixelsy, pixelsize)
    
    # Create fluorophore with reasonable blinking rates for target ~10 events
    # For nframes/framerate total time, want ~1 Hz rates for reasonable blinking
    total_time = nframes / framerate  # seconds
    target_events = 10
    k_off = target_events / (2 * total_time)  # ~5 off events
    k_on = target_events / (2 * total_time)   # ~5 on events
    
    fluor = SMLMSim.GenericFluor(photons=1e5, k_off=k_off, k_on=k_on)
    
    # Run simulation - returns (smld_true, smld_model, smld_noisy)
    smld_true, smld_model, smld_noisy = SMLMSim.simulate(params; molecule=fluor, camera=camera)
    
    # Return the noisy dataset by default (has Emitter2DFit with σ_x, σ_y)
    # or the true/model datasets if requested
    if return_noisy
        # Store PSF width in metadata for reference
        smld_noisy.metadata["σ_psf"] = σ_psf
        smld_noisy.metadata["simulation_type"] = "noisy_localizations"
        return smld_noisy
    else
        # Return true positions if specifically requested
        smld_true.metadata["σ_psf"] = σ_psf
        smld_true.metadata["simulation_type"] = "true_positions"
        return smld_true
    end
end

"""
    simulate_nmer_smlmsim(; n=6, diameter=0.050, density=0.1, kwargs...) -> SMLMSim.BasicSMLD

Create an n-mer pattern simulation using SMLMSim's pattern system.

# Arguments
- `n`: Number of emitters in the circular n-mer
- `diameter`: Diameter of the n-mer circle in micrometers
- `density`: Background emitter density
- `kwargs...`: Additional arguments passed to simulate_static_smlm

# Returns
- BasicSMLD structure with n-mer pattern embedded in background emitters

# Example
```julia
# Simulate a 6-mer with 50nm diameter
smld = simulate_nmer_smlmsim(n=6, diameter=0.050)

# Run BaGoL analysis
chains = run_bagol(smld)
```
"""
function simulate_nmer_smlmsim(; n=6, 
                                diameter=0.050, 
                                density=0.1, 
                                kwargs...)
    
    # For now, just return a basic static simulation
    # TODO: Add n-mer pattern support when SMLMSim API is clearer
    println("Note: simulate_nmer_smlmsim currently returns basic static simulation.")
    println("      N-mer pattern support will be added in future version.")
    
    smld = simulate_static_smlm(; density=density, kwargs...)
    
    # Store n-mer metadata for reference
    smld.metadata["nmer_n"] = n
    smld.metadata["nmer_diameter"] = diameter
    smld.metadata["simulation_type"] = "static_placeholder"
    
    return smld
end