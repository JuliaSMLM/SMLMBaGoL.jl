#!/bin/bash

# Simple script to run nmer_demo multiple times with different random seeds

echo "🧪 Running nmer_demo multiple times to test K prior fix"
echo "======================================================"

cd ../../examples

# Backup original demo
cp nmer_demo.jl nmer_demo_backup.jl

# Results arrays
results_file="../dev/debug/multi_trial_results.txt"
echo "# Multi-trial results" > $results_file
echo "# Trial, Seed, True_K, MAPN_K, Error_Pct, Posterior_Mean, Final_mu, N_locs" >> $results_file

for trial in {1..5}; do
    seed=$((3000 + trial))
    echo ""
    echo "🔬 Trial $trial (seed=$seed)"
    echo "========================="
    
    # Modify the demo to use this seed and shorter iterations
    sed -e "s/const N_ITERATIONS = 50000/const N_ITERATIONS = 15000/" \
        -e "s/const BURN_IN = 5000/const BURN_IN = 2000/" \
        -e "s/Random.seed!(1234)/Random.seed!($seed)/" \
        nmer_demo_backup.jl > nmer_demo.jl
    
    # Run the demo and capture key results
    output=$(julia --project=. --threads=auto nmer_demo.jl 2>&1)
    
    # Extract results using grep
    true_k=6
    mapn_k=$(echo "$output" | grep "MAPN estimate:" | sed 's/.*: \([0-9]\+\).*/\1/')
    n_locs=$(echo "$output" | grep "Total localizations:" | sed 's/.*: \([0-9]\+\).*/\1/')
    
    # Try to extract posterior mean from summary
    posterior_mean=$(echo "$output" | grep -A 20 "K Posterior Distribution:" | grep "K=[0-9]" | head -1 | sed 's/.*K=\([0-9]\+\).*/\1/' || echo "7")
    
    # Calculate error  
    if [ ! -z "$mapn_k" ] && [ "$mapn_k" != "" ]; then
        error_pct=$(echo "scale=1; ($mapn_k - $true_k) * 100 / $true_k" | bc -l | sed 's/-//')
        echo "   ✅ MAPN K=$mapn_k (${error_pct}% error), N=$n_locs"
        
        # Save to results file
        echo "$trial, $seed, $true_k, $mapn_k, $error_pct, $posterior_mean, NA, $n_locs" >> $results_file
    else
        echo "   ❌ Failed to extract results"
        echo "$trial, $seed, $true_k, NA, NA, NA, NA, NA" >> $results_file
    fi
done

# Restore original
mv nmer_demo_backup.jl nmer_demo.jl

echo ""
echo "📊 SUMMARY"
echo "=========="

# Calculate statistics from results file
cd ../dev/debug
julia << 'EOF'
using Statistics

# Read results
results = []
open("multi_trial_results.txt", "r") do f
    for line in eachline(f)
        if startswith(line, "#")
            continue
        end
        parts = split(line, ",")
        if length(parts) >= 4 && parts[4] != "NA"
            mapn_k = parse(Int, strip(parts[4]))
            push!(results, mapn_k)
        end
    end
end

if !isempty(results)
    true_k = 6
    n_trials = length(results)
    exact_matches = count(k -> k == true_k, results)
    close_matches = count(k -> abs(k - true_k) <= 1, results)
    mean_error = mean(abs.(results .- true_k))
    
    println("📈 Results from $n_trials successful trials:")
    println("  Exact accuracy (K=6): $(round(exact_matches/n_trials*100, digits=1))%")
    println("  Close accuracy (±1):  $(round(close_matches/n_trials*100, digits=1))%")
    println("  Mean absolute error:  $(round(mean_error, digits=2)) emitters")
    println("  MAPN distribution: $(sort(results))")
    
    if exact_matches/n_trials >= 0.6
        println("✅ EXCELLENT: ≥60% exact accuracy!")
    elseif close_matches/n_trials >= 0.8
        println("✅ GOOD: ≥80% within ±1 emitter")
    elseif mean_error < 1.5
        println("✅ ACCEPTABLE: Mean error < 1.5")
    else
        println("⚠️ NEEDS IMPROVEMENT: High error rate")
    end
else
    println("❌ No successful trials")
end
EOF

echo ""
echo "🎉 Multi-trial validation complete!"
echo "Results saved to: multi_trial_results.txt"