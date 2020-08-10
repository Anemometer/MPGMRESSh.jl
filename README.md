# MPGMRESSh.jl
This `julia` package provides a proof-of-concept implementation of the multi-preconditioned GMRES iterative solution method for shifted systems
of linear problems (see [Bakhos T. et al., Multipreconditioned GMRES for Shifted Systems](https://arxiv.org/abs/1603.08970)).

As of now, it is still in an early experimental stage.

## Remarks
Note that some Krylov-based preconditioning solves are currently not compatible with certain preconditioning techniques provided by [`ExtendableSparse.jl`](https://github.com/j-fu/ExtendableSparse.jl) and [`IncompleteLU.jl`](https://github.com/haampie/IncompleteLU.jl).

Specifically, an in-place `ldiv!` routine is missing in `jacobi.jl` of `ExtendableSparse.jl` which is required by both the GMRES and BiCGStabl solvers as provided by [`IterativeSolvers.jl`](https://github.com/JuliaMath/IterativeSolvers.jl).

Additionally, `IncompleteLU` [lacks support for complex matrices](https://github.com/haampie/IncompleteLU.jl/issues/3) which can be fixed by changing 
```julia 
function append_col!(A::SparseMatrixCSC{Tv}, y::SparseVectorAccumulator{Tv}, j::Int, drop::Tv, scale::Tv = one(Tv)) where {Tv}
```
in `sparse_vector_accumulator.jl` to
```julia
function append_col!(A::SparseMatrixCSC{Tv}, y::SparseVectorAccumulator{Tv}, j::Int, drop::Td, scale::Tv = one(Tv)) where {Tv,Td}
```
