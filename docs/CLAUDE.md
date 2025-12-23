# Documentation Guidelines for docs/

This directory contains the package documentation using Documenter.jl. Follow these conventions when creating or updating documentation.

## Directory Structure

```
docs/
├── Project.toml      # Documentation-specific dependencies
├── make.jl          # Build configuration
├── src/             # Markdown source files
│   ├── index.md     # Home page
│   ├── api.md       # API reference
│   └── *.md         # Other topic pages
└── build/           # Generated documentation (gitignored)
```

## Setting Up Documentation

### docs/Project.toml
Create a separate environment for documentation with these dependencies:
```toml
[deps]
Documenter = "e30172f5-a6a5-5a46-863b-614d45cd2de4"
SMLMBaGoL = "..."  # Your package UUID
```

Optional dependencies for enhanced features:
- `DocumenterInterLinks` - Cross-reference other packages
- `DocumenterTools` - Deployment utilities

### docs/make.jl Structure

Basic template:
```julia
using Documenter
using SMLMBaGoL

# Optional: Set up doctests
DocMeta.setdocmeta!(SMLMBaGoL, :DocTestSetup, :(using SMLMBaGoL); recursive=true)

makedocs(
    sitename = "SMLMBaGoL.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://yourusername.github.io/SMLMBaGoL.jl/stable/",
    ),
    modules = [SMLMBaGoL],
    pages = [
        "Home" => "index.md",
        "User Guide" => "guide.md",
        "API Reference" => "api.md",
    ],
)

# Deploy docs (usually from CI)
deploydocs(
    repo = "github.com/yourusername/SMLMBaGoL.jl.git",
    devbranch = "main",
)
```

## Creating Documentation Pages

### Home Page (index.md)
```markdown
# SMLMBaGoL.jl

Brief description of your package.

## Installation

```julia
using Pkg
Pkg.add("SMLMBaGoL")
```

## Quick Start

```@example
using SMLMBaGoL
# Simple example
result = run_bagol(localizations)
```
```

### API Reference (api.md)

For simple packages:
```markdown
# API Reference

```@index
```

```@autodocs
Modules = [SMLMBaGoL]
```
```

For complex packages with public/internal separation:
```markdown
# API Reference

## Public API

```@docs
SMLMBaGoL.run_bagol
SMLMBaGoL.estimate_mapn
```

## Internal API

```@autodocs
Modules = [SMLMBaGoL]
Public = false
```
```

## Writing Documentation

### Code Examples

#### Use @example blocks for demonstrations:
```markdown
```@example
using SMLMBaGoL
data = generate_data()
result = process(data)
println("Result: ", result)
```
```

#### Use jldoctest for verified examples:
```markdown
```jldoctest
julia> using SMLMBaGoL

julia> add(2, 3)
5
```
```

### Best Practices

1. **Doctests**:
   - Place in docstrings for verification
   - Avoid RNG without explicit seeds
   - Use filters for variable output

2. **Examples**:
   - Use @example for complex demonstrations
   - Name blocks to share state: `@example myexample`
   - Hide setup code with `# hide` comments

3. **Cross-references**:
   - Link to functions: `[`SMLMBaGoL.function`](@ref)`
   - Link to sections: `[Installation](@ref)`
   - External links with DocumenterInterLinks

4. **Math and Code**:
   - Use single backticks for inline code: `function_name`
   - Use double backticks for LaTeX: ``α = 1``
   - Prefer Unicode over LaTeX escapes

## Building Documentation

### Local Build
```bash
cd docs
julia --project=. make.jl
```

The documentation will be in `docs/build/`.

### Local Development
For live-reload during development:
```julia
using LiveServer
cd("docs")
servedocs()
```

### CI Integration
Documentation typically builds and deploys via GitHub Actions:
1. Build on pull requests (without deployment)
2. Deploy to gh-pages branch when merging to main

## Common Patterns

### Organizing Complex Documentation
For larger packages, organize pages hierarchically:
```julia
pages = [
    "Home" => "index.md",
    "User Guide" => [
        "Getting Started" => "guide/start.md",
        "Advanced Usage" => "guide/advanced.md",
    ],
    "Examples" => [
        "Basic Examples" => "examples/basic.md",
        "Advanced Examples" => "examples/advanced.md",
    ],
    "API Reference" => "api.md",
]
```

### Including External Files
To include Julia files from examples/:
```markdown
```@example
include("../../examples/demo.jl")
```
```

### Custom Styling
Add custom CSS/JS in make.jl:
```julia
format = Documenter.HTML(
    assets = ["assets/custom.css"],
)
```

## Troubleshooting

- **Missing docstrings**: Set `checkdocs = :none` in makedocs()
- **Broken doctests**: Update examples or use `doctest = false`
- **Build failures**: Check docs/Project.toml dependencies
- **Cross-references not working**: Ensure proper @ref syntax

## SMLMBaGoL-Specific Documentation

### Current Documentation Files
- `movie_generator.md` - Technical documentation for animation features

### Key Documentation Areas for SMLMBaGoL

#### User Guide Topics
- **Installation and Setup** - Package installation and dependencies
- **Quick Start Tutorial** - Basic workflow with simple examples
- **Core Concepts** - RJMCMC, Bayesian inference, emitter detection
- **Advanced Features** - Hierarchical priors, consistency likelihood
- **Performance Optimization** - Threading, partitioning, memory management
- **Integration Guide** - Working with SMLMSim, SMLMData ecosystem

#### API Documentation Sections
- **Main Functions**: `run_bagol()`, `estimate_mapn()`, `diagnose_chains()`
- **Simulation Integration**: `simulate_static_smlm()`, `simulate_n_mer()`
- **Visualization**: `gen_sr_image()`, `sr_circles()`, movie generation
- **Configuration Types**: Likelihood configurations, prior specifications
- **Diagnostics**: Chain assessment, convergence monitoring

#### Mathematical Background
- **Statistical Model** - Bayesian formulation, likelihood functions
- **RJMCMC Algorithm** - Move types, acceptance criteria
- **Consistency Likelihood** - Innovation addressing over-segmentation
- **Hierarchical Priors** - Negative binomial, spatial distributions

### Documentation Best Practices for SMLM

1. **Domain-Specific Terminology**:
   - Define SMLM concepts clearly for new users
   - Explain statistical terms in biological context
   - Use consistent notation throughout

2. **Mathematical Notation**:
   - Use Unicode symbols appropriately (α, κ, τ)
   - Provide both mathematical and intuitive explanations
   - Include references to relevant literature

3. **Example Integration**:
   - Link to examples/ directory for runnable code
   - Show realistic parameter values for SMLM experiments
   - Demonstrate common workflows and pitfalls

4. **Performance Guidance**:
   - Document threading requirements and benefits
   - Provide scaling guidelines for large datasets
   - Include benchmarking information where available