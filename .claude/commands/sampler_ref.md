# Sampler Reference Dashboard

Display the collapsed Gibbs sampler reference document and optionally run optimality tests.

## Instructions

1. **Read and present** `dev/sampler_reference.md` — the authoritative math reference for the collapsed sampler.

2. **Check current test status** by looking for recent output files:
   - `dev/output/` — check for PNG files from optimality tests and their modification dates
   - `dev/output/unified_optimality/dashboard.png` — unified sweep results
   - `test/` — check if the count-model optimality testset is passing

3. **If the user passes `run` as an argument**, execute the tests:
   - Quick (default): `julia --threads=auto --project=dev dev/validate_qpaint.jl`
   - Full: `julia --threads=auto --project=dev dev/optimality_tests.jl`
   - Unified: `julia --threads=auto --project=dev dev/optimality/run.jl`
   - CI only: `julia --project=. -e "using Pkg; Pkg.test()"`

   Ask which test to run if not specified. Run in background and report results when done.

4. **If the user passes a section name** (e.g., `moves`, `area`, `limits`, `tests`, `anti-patterns`), show only that section.

5. **Before any sampler modification**, you MUST:
   - Read `dev/sampler_reference.md`
   - Identify which section describes the code you're changing
   - Verify your change is consistent with the documented math
   - After the change, update the reference doc

6. **Present a summary** showing:
   - Current move types and their proposal/acceptance math (one line each)
   - Key invariants (area cancellation, fixed μ₀, Q-PAINT floor)
   - Most recent test results (from output file dates, or "not run recently")
   - Any anti-patterns relevant to the current work

## Output Format

Use concise tables and math notation. Don't dump the entire doc — highlight the sections relevant to the current conversation context. If the user is about to modify a specific move, focus on that move's math and the tests that would catch a regression.
