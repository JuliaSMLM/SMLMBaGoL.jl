- using statements must only go in the main SMLMBaGoL.jl file. 
- all `using` or `import` commands should be in main module file only, not in other files. 

- You can ask me to find the parent branch using any of these approaches:

  Direct commands:
  - "What branch was this created from?"
  - "Find the parent branch"
  - "What's the source branch for this feature branch?"

  More specific Git requests:
  - "Run git show-branch to find the parent"
  - "Use git merge-base to find where this branched from"
  - "Check git log --graph to see the branch point"

  The most reliable approach:
  git show-branch | head -10
  This shows the branching relationship clearly.

  Or ask me to:
  - "Run the right git commands to find which branch this was created from"
  - "Check the git history to find the actual parent branch"

  The key is I should look at the first commit on the feature branch (9be0e2e) and see what it was based on, rather than just looking at merge-base with main. The git branch --contains <commit> command I used was the
  right approach to see that commit 8d6563f exists on negative-binomial-overdispersion.

- use multiple threads when running examples