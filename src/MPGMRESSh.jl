"""
Main module for `MPGMRESSh.jl` which is a Julia package for solving systems 
of linear equations indexed by a shift `(A + \\sigma_i M)x_i = b` for given matrices
`A`, and `M`, and vectors `x_i`, b with a finite number of shifts `\\sigma_i \\in \\mathbb{C}`, `i = 1,2,3,\\, \\ldots`.
See [Bakhos T. et al., Multipreconditioned GMRES for Shifted Systems](https://arxiv.org/abs/1603.08970)
"""
module MPGMRESSh

# MPGMRESSh core functionalities
include("mpgmressh.jl")

# Iterative methods with a fixed or flexible single preconditioners for later comparison
#include("precongmres.jl")
#include("fgmres.jl")

end