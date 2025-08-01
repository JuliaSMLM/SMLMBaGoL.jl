# Test Organization

This document explains the two-tier testing system used in SMLMBaGoL.

## 1. Formal Tests (`test/`)

**Purpose**: Standard Julia package tests for CI/CD and regression testing.

**Characteristics**:
- Must return pass/fail boolean results
- Fast execution (< 5 minutes total)
- Minimal dependencies
- No plots or visualizations
- Part of `Pkg.test()` workflow

**Files**:
- `runtests.jl` - Test runner
- `test_hierarchical_k_math.jl` - Mathematical correctness tests
- `test_latent_positions.jl` - Position tracking tests
- `test_emitter_utils.jl` - Utility function tests
- `test_hierarchical_fitting.jl` - Fitting algorithm tests

**Run with**:
```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

## 2. Development Tests (`dev/`)

**Purpose**: Exploratory analysis, mathematical validation, and debugging.

**Characteristics**:
- Can produce plots and human-readable output
- May have longer runtimes
- Can use additional dependencies (Plots.jl, etc.)
- Not required to "pass" - used for understanding
- Generate visualizations and detailed analysis

**Structure**:
```
dev/
├── math/              # Mathematical concept validation
├── concepts/          # Algorithm exploration
├── benchmarks/        # Performance analysis
├── Project.toml       # Dev environment with plotting deps
└── README.md          # Dev test documentation
```

**Example Files**:
- `dev/math/validate_hierarchical_k.jl` - Parameter estimation analysis with plots
- `dev/concepts/explore_slice_sampling.jl` - Convergence visualization
- `dev/benchmarks/profile_updates.jl` - Performance profiling

**Run with**:
```bash
cd dev
julia --project=. -e 'include("math/validate_hierarchical_k.jl")'
```

## When to Use Which

### Use Formal Tests (`test/`) for:
- Verifying algorithms produce correct results
- Regression testing
- CI/CD integration
- Quick pass/fail validation
- Mathematical property verification

### Use Development Tests (`dev/`) for:
- Understanding parameter behavior across ranges
- Visualizing convergence properties
- Debugging algorithm issues
- Performance analysis
- Creating publication-quality plots
- Exploring edge cases in detail

## Migration Strategy

1. Keep existing formal tests in `test/`
2. Move exploratory scripts to `dev/`
3. Create new development tests in `dev/` when you need visualization or detailed analysis
4. Only add to formal tests when you need pass/fail validation

This organization allows for both rigorous testing and exploratory analysis without mixing concerns.