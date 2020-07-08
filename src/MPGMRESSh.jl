"""
Main module for `MPGMRESSh.jl` which is a Julia package for solving systems 
of linear equations indexed by a shift `(A + \\sigma_i M)x_i = b` for given matrices
`A`, and `M`, and vectors `x_i`, b with a finite number of shifts `\\sigma_i \\in \\mathbb{C}`, `i = 1,2,3,\\, \\ldots`.
See [Bakhos T. et al., Multipreconditioned GMRES for Shifted Systems](https://arxiv.org/abs/1603.08970)
"""
module MPGMRESSh

# Shift-and-invert preconditioner functionalities
include("preconditioner.jl")

# MPGMRESSh core functionalities
include("mpgmressh.jl")

# Flexible GMRES for shifted systems as specified in 
# [Saibaba A. et al., A Flexible Krylov Solver for Shifted Systems...](https://arxiv.org/abs/1212.3660)
include("fgmressh.jl")

end