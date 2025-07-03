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
                          pixelsize=0.1, return_noisy=true,
                          n_mer=nothing, n_mer_diameter=0.050, n_mer_x=nothing, n_mer_y=nothing) -> SMLMSim.BasicSMLD

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
- `loc_per_emitter`: Target number of localizations per emitter (controls blinking rates, default: 10)
- `return_noisy`: Return noisy localizations (true) or true positions (false)
- `n_mer`: Number of emitters in n-mer pattern (nothing for no pattern)
- `n_mer_diameter`: Diameter of n-mer circle in micrometers (default: 0.050)
- `n_mer_x`: X-center of n-mer pattern (default: center of field)
- `n_mer_y`: Y-center of n-mer pattern (default: center of field)

# Returns
- Single BasicSMLD structure ready for BaGoL analysis

# Example
```julia
# Quick simulation with defaults (6.4μm × 3.2μm field)
smld = simulate_static_smlm()

# Custom simulation with larger field (25.6μm × 25.6μm)
smld = simulate_static_smlm(npixelsx=256, npixelsy=256, density=0.2)

# Simulation with 6-mer pattern (50nm diameter)
smld = simulate_static_smlm(n_mer=6, n_mer_diameter=0.050)

# Simulation with 3-mer pattern (100nm diameter) at specific location
smld = simulate_static_smlm(n_mer=3, n_mer_diameter=0.100, n_mer_x=3.2, n_mer_y=1.6)

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
                               loc_per_emitter=10,
                               tau=0.0,
                               return_noisy=true,
                               n_mer=nothing,
                               n_mer_diameter=0.050,
                               n_mer_x=nothing,
                               n_mer_y=nothing)
    
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
    
    # Create fluorophore with blinking rates to achieve target localizations per emitter
    total_time = nframes / framerate  # seconds
    target_events = loc_per_emitter
    
    # Use framerate as k_off (synchronized with frame acquisition)
    k_off = framerate  # off rate tied to acquisition rate
    
    # k_on based on desired events over total time
    # Some events will span multiple frames, giving 2-3x target localizations, but that's acceptable
    k_on = target_events / total_time  # on rate to achieve target events
    
    fluor = SMLMSim.GenericFluor(photons=1e5, k_off=k_off, k_on=k_on)
    
    # Create pattern if n-mer is requested
    pattern = nothing
    center_x = nothing
    center_y = nothing
    if n_mer !== nothing && n_mer > 0
        # Calculate field center if not specified
        field_width = npixelsx * pixelsize
        field_height = npixelsy * pixelsize
        center_x = n_mer_x === nothing ? field_width / 2 : n_mer_x
        center_y = n_mer_y === nothing ? field_height / 2 : n_mer_y
        
        # Generate n-mer positions manually for precise control
        radius = n_mer_diameter / 2
        angles = range(0, 2π, length=n_mer+1)[1:end-1]  # n equally spaced angles
        x_positions = [center_x + radius * cos(angle) for angle in angles]
        y_positions = [center_y + radius * sin(angle) for angle in angles]
        
        # Create n-mer pattern using SMLMSim's built-in pattern system with explicit positions
        pattern = SMLMSim.Nmer2D(n_mer, n_mer_diameter, x_positions, y_positions)
    end
    
    # Run simulation - returns (smld_true, smld_model, smld_noisy)
    if pattern !== nothing
        smld_true, smld_model, smld_noisy = SMLMSim.simulate(params; molecule=fluor, camera=camera, pattern=pattern)
        
        # Store n-mer metadata
        for smld in [smld_true, smld_model, smld_noisy]
            smld.metadata["n_mer"] = n_mer
            smld.metadata["n_mer_diameter"] = n_mer_diameter
            smld.metadata["n_mer_center_x"] = center_x
            smld.metadata["n_mer_center_y"] = center_y
        end
    else
        smld_true, smld_model, smld_noisy = SMLMSim.simulate(params; molecule=fluor, camera=camera)
    end
    
    # Return the noisy dataset by default (has Emitter2DFit with σ_x, σ_y)
    # or the true/model datasets if requested
    if return_noisy
        # Add additional systematic noise with tau parameter
        if tau > 0.0
            for emitter in smld_noisy.emitters
                # Add additional Gaussian noise to positions (keep sigmas unchanged)
                emitter.x += tau * randn()
                emitter.y += tau * randn()
                # Note: σ_x and σ_y remain as originally calculated (photon-noise limited)
            end
        end
        
        # Store PSF width in metadata for reference
        smld_noisy.metadata["σ_psf"] = σ_psf
        smld_noisy.metadata["simulation_type"] = "noisy_localizations"
        smld_noisy.metadata["tau"] = tau
        return smld_noisy
    else
        # Add additional systematic noise with tau parameter for true positions
        if tau > 0.0
            for emitter in smld_true.emitters
                # Add additional Gaussian noise to positions
                emitter.x += tau * randn()
                emitter.y += tau * randn()
            end
        end
        
        # Return true positions if specifically requested
        smld_true.metadata["σ_psf"] = σ_psf
        smld_true.metadata["simulation_type"] = "true_positions"
        smld_true.metadata["tau"] = tau
        return smld_true
    end
end

"""
    simulate_nmer_smlmsim(; n=6, diameter=0.050, density=0.1, kwargs...) -> SMLMSim.BasicSMLD

Create an n-mer pattern simulation using SMLMSim with background emitters.

# Arguments
- `n`: Number of emitters in the circular n-mer
- `diameter`: Diameter of the n-mer circle in micrometers (default: 0.050)
- `density`: Background emitter density (default: 0.1)
- `kwargs...`: Additional arguments passed to simulate_static_smlm

# Returns
- BasicSMLD structure with n-mer pattern embedded in background emitters

# Example
```julia
# Simulate a 6-mer with 50nm diameter
smld = simulate_nmer_smlmsim(n=6, diameter=0.050)

# Simulate a 3-mer with 100nm diameter and higher background density
smld = simulate_nmer_smlmsim(n=3, diameter=0.100, density=0.5)

# Run BaGoL analysis
chains = run_bagol(smld)
```
"""
function simulate_nmer_smlmsim(; n=6, 
                                diameter=0.050, 
                                density=0.1, 
                                kwargs...)
    
    # Use the new n-mer functionality in simulate_static_smlm
    smld = simulate_static_smlm(; 
        density=density, 
        n_mer=n, 
        n_mer_diameter=diameter, 
        kwargs...)
    
    # Update metadata to indicate this is an n-mer simulation
    smld.metadata["simulation_type"] = "nmer_pattern"
    
    return smld
end