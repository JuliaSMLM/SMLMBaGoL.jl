"""
Validate that a BaGoLState maintains proper Gibbs sampling consistency.
Returns (is_valid, error_messages)
"""
function validate_state_consistency(state::BaGoLState; tolerance=1e-10)
    errors = String[]
    
    # Check 1: All allocations are valid
    for (i, alloc) in enumerate(state.allocations)
        if alloc < 1 || alloc > length(state.emitters)
            push!(errors, "Invalid allocation: localization $i assigned to emitter $alloc")
        end
    end
    
    # Check 2: Latent positions are consistent with model
    for (i, alloc) in enumerate(state.allocations)
        if 1 ≤ alloc ≤ length(state.emitters)
            emitter = state.emitters[alloc]
            loc = state.localizations[i]
            latent = state.latent_positions[i]
            
            # Expected squared distances under the model
            expected_dist_to_emitter² = 2 * state.τ²  # E[||r-s||²] for 2D Gaussian
            expected_dist_to_obs² = 2 * (loc.σx^2 + loc.σy^2)
            
            actual_dist_to_emitter² = (latent[1] - emitter.x)^2 + (latent[2] - emitter.y)^2
            actual_dist_to_obs² = (latent[1] - loc.x)^2 + (latent[2] - loc.y)^2
            
            # Check if distances are reasonable (within 5 std devs)
            if actual_dist_to_emitter² > 25 * expected_dist_to_emitter²
                push!(errors, "Latent position $i too far from emitter $alloc: distance² = $actual_dist_to_emitter², expected ≈ $expected_dist_to_emitter²")
            end
            if actual_dist_to_obs² > 25 * expected_dist_to_obs²
                push!(errors, "Latent position $i too far from observation: distance² = $actual_dist_to_obs², expected ≈ $expected_dist_to_obs²")
            end
        end
    end
    
    # Check 3: Log-likelihood is finite
    if !isfinite(state.log_likelihood)
        push!(errors, "Log-likelihood is not finite: $(state.log_likelihood)")
    end
    
    return (isempty(errors), errors)
end

export validate_state_consistency