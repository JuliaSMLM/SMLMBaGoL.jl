# Pretty printing for SMLMBaGoL types

import Base: show

# Pretty printing for emitters is handled by SMLMData
# We only add display methods for types specific to SMLMBaGoL

# Pretty printing for Localization2D
function Base.show(io::IO, loc::Localization2D)
    print(io, "Localization2D(pos=($(round(loc.x, digits=3)), $(round(loc.y, digits=3))), σ=($(round(loc.σx, digits=3)), $(round(loc.σy, digits=3))), frame=$(loc.frame))")
end

function Base.show(io::IO, ::MIME"text/plain", loc::Localization2D)
    println(io, "Localization2D:")
    println(io, "  Position: ($(round(loc.x, digits=3)), $(round(loc.y, digits=3))) μm")
    println(io, "  Precision: ($(round(loc.σx, digits=3)), $(round(loc.σy, digits=3))) μm")
    println(io, "  Frame: $(loc.frame)")
end

# Pretty printing for arrays of Emitters
function Base.show(io::IO, ::MIME"text/plain", emitters::Vector{E}) where E<:AbstractEmitter
    n = length(emitters)
    println(io, "$n Emitter$(n == 1 ? "" : "s"):")
    
    if n == 0
        println(io, "  (empty)")
        return
    end
    
    # Show first few emitters
    max_shown = 10
    show_count = min(n, max_shown)
    for i in 1:show_count
        emitter = emitters[i]
        if isa(emitter, Emitter2DFit)
            println(io, "  [$i] ID $(emitter.id): ($(round(emitter.x, digits=3)), $(round(emitter.y, digits=3))) ± ($(round(emitter.σ_x, digits=3)), $(round(emitter.σ_y, digits=3))) μm")
        else
            # Basic Emitter2D doesn't have uncertainties
            println(io, "  [$i]: ($(round(emitter.x, digits=3)), $(round(emitter.y, digits=3))) μm, $(round(emitter.photons, digits=0)) photons")
        end
    end
    
    if n > max_shown
        println(io, "  ... and $(n-max_shown) more")
    end
end

# Pretty printing for arrays of Localizations
function Base.show(io::IO, ::MIME"text/plain", locs::Vector{L}) where L<:AbstractLocalization
    n = length(locs)
    println(io, "$n Localization$(n == 1 ? "" : "s"):")
    
    if n == 0
        println(io, "  (empty)")
        return
    end
    
    # Calculate summary statistics
    x_coords = [loc.x for loc in locs]
    y_coords = [loc.y for loc in locs]
    σx_coords = [loc.σx for loc in locs]
    σy_coords = [loc.σy for loc in locs]
    frames = [loc.frame for loc in locs]
    
    println(io, "  Position range: x ∈ [$(round(minimum(x_coords), digits=3)), $(round(maximum(x_coords), digits=3))], y ∈ [$(round(minimum(y_coords), digits=3)), $(round(maximum(y_coords), digits=3))] μm")
    println(io, "  Precision: σx = $(round(mean(σx_coords), digits=3)) ± $(round(std(σx_coords), digits=3)), σy = $(round(mean(σy_coords), digits=3)) ± $(round(std(σy_coords), digits=3)) μm")
    println(io, "  Frame range: $(minimum(frames)) to $(maximum(frames))")
    
    # Show first few localizations
    show_count = min(n, 3)
    println(io, "  First $(show_count):")
    for i in 1:show_count
        loc = locs[i]
        println(io, "    [$i] ($(round(loc.x, digits=3)), $(round(loc.y, digits=3))) ± ($(round(loc.σx, digits=3)), $(round(loc.σy, digits=3))) μm, frame $(loc.frame)")
    end
    
    if n > 3
        println(io, "    ... and $(n-3) more")
    end
end

# Pretty printing for BaGoLState
function Base.show(io::IO, ::MIME"text/plain", state::BaGoLState)
    n_emitters = length(state.emitters)
    n_locs = length(state.localizations)
    
    println(io, "BaGoLState:")
    println(io, "  Emitters: $n_emitters")
    println(io, "  Localizations: $n_locs")
    println(io, "  Log-likelihood: $(round(state.log_likelihood, digits=2))")
    
    if n_emitters > 0
        println(io, "  Emitter positions:")
        show_count = min(n_emitters, 3)
        for i in 1:show_count
            emitter = state.emitters[i]
            n_allocated = count(==(i), state.allocations)
            if isa(emitter, Emitter2DFit)
                println(io, "    ID $(emitter.id): ($(round(emitter.x, digits=3)), $(round(emitter.y, digits=3))) ± ($(round(emitter.σ_x, digits=3)), $(round(emitter.σ_y, digits=3))) μm ($(n_allocated) localizations)")
            else
                # Basic Emitter2D
                println(io, "    Emitter $i: ($(round(emitter.x, digits=3)), $(round(emitter.y, digits=3))) μm, $(round(emitter.photons, digits=0)) photons ($(n_allocated) localizations)")
            end
        end
        if n_emitters > 3
            println(io, "    ... and $(n_emitters-3) more")
        end
    end
end

# Pretty printing for RJMCMCChain
function Base.show(io::IO, ::MIME"text/plain", chain::RJMCMCChain)
    n_samples = length(chain.samples)
    n_locs = length(chain.localizations)
    current_K = length(chain.current_state.emitters)
    
    println(io, "RJMCMCChain:")
    println(io, "  Localizations: $n_locs")
    println(io, "  Current emitters: $current_K")
    println(io, "  Samples collected: $n_samples")
    println(io, "  Burn-in: $(chain.burn_in)")
    println(io, "  Thinning: $(chain.thin)")
    
    if n_samples > 0
        K_values = [length(sample.emitters) for sample in chain.samples]
        println(io, "  Emitter count range: $(minimum(K_values)) to $(maximum(K_values))")
        println(io, "  Mean emitter count: $(round(mean(K_values), digits=1))")
    end
end