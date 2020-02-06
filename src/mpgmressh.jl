import Base: iterate 
using Printf, BlockDiagonals, LinearAlgebra, IterativeSolvers
export mpgmressh, mpgmressh!

# This code draws heavily from the style already adopted for gmres.jl of IterativeSolvers.jl.

# !TODO: make BlockArnoldi immutable by making BlockArnoldiStep work index-bound in-place
mutable struct BlockArnoldiDecomp{elT, opT, shiftT, precT}
    A::opT # (finite-dimensional) linear operator A (optional)
    M::opT
    V::Array{Matrix{elT}, 1} # orthonormal part of the Block Arnoldi decomposition in the form of matrix blocks
    Z::Matrix{elT} # search directions
    H::Matrix{elT} # shift-independent part of the Hessenberg matrices
    E::Matrix{elT} # E,T: auxiliary matrices to recover the Arnoldi relations for each shift σ
    T::BlockDiagonal{elT, Matrix{elT}}

    precons::Array{precT, 1} # shift-and-invert preconditioners
    allshifts::Array{shiftT, 1} # all shift parameters σ
    #preconshifts::Array{shiftT, 1} # shift parameters to be used for building the shift-and-invert preconditioners (optional if preconditioners are passed by the user)
end
# !TODO: figure out a sensible name for the type parameter T conflicting with the matrix name T
function BlockArnoldiDecomp(A::opT, M::opT, cycleit::Int, allshifts::Array{shiftT, 1}, preconshifts::Array{shiftT, 1}, precons::Array{precT, 1}, elT::Type) where {opT, shiftT, precT}
    nprecons = length(preconshifts)
    H = zeros(elT, nprecons * cycleit + 1, nprecons * cycleit)
    #V = zeros(T, size(M, 1), nprecons * cycleit + 1)
    V = [zeros(elT, size(M, 1), 1)]
    Z = zeros(elT, size(M, 1), nprecons * cycleit)

    # E is a block diagonal of the form 
    # < -  nprecons * cycleit   - >
    # [ 1 ... 1   0    .   .   0  ]
    # |    0      Ek              |
    # |    .           .          |
    # |    .               .      |
    # |    0      .    .   .   Ek |
    # |                           |
    # [    0      .    .   .   0  ] <- nprecons many zero rows
    #
    # 
    # where 
    #      <- nprecons  ->
    #      [ 0 0       0 ]
    #      | . .       . |
    # Ek = | . . . . . . |
    #      | 0 0       0 |
    #      [ 1 1       1 ]

    enp = zeros(elT, nprecons, 1)
    enp[end] = one(elT)
    Ek = enp * ones(elT, 1, nprecons)
    E1 = BlockDiagonal([ones(elT, 1, nprecons),[Ek for k=2:cycleit]...])
    E = zeros(elT, size(H)...)
    E[1:end - nprecons, :] = Matrix(E1)

    # !TODO: find a way to represent the blockdiagonal in terms of 
    # Diagonal Types
    T = BlockDiagonal([Matrix(Diagonal(preconshifts)) for k=1:cycleit])
    #display(typeof(T))

    return BlockArnoldiDecomp(A, M, V, Z, H, E, T, precons, allshifts)
end

mutable struct Residual{T, resT}
    absres::Array{resT, 1}  # Array of absolute residuals for each shift σ
    accumulator::Array{resT, 1} # Placeholder for updating the residual per iteration for each shift σ
    nullvec::Matrix{T} # Vectors for each Hessenberg matrix for computing the residuals
    β::resT # initial residual which is the norm of b since the initial iterate is set to 0
end

Residual(order::Int, nshifts::Int, T::Type) = Residual{T, real(T)}(
    ones(T, nshifts),
    ones(T, nshifts),
    ones(T, order, nshifts),
    one(real(T))
)

"""
    Convergence <: Enum{Int32}
    
Enum type specifying the convergence criteria to be used for the 
MPGMRESSh method.
The choice is between:
* standard: Test whether the current relative residual is smaller than a specified tolerance.
* paige_saunders: Use an extended type of convergence criterion that is better suited for ill-conditioned matrices.
  See [Bakhos T. et al., Multipreconditioned GMRES for Shifted Systems, p.12](https://arxiv.org/abs/1603.08970)
* absolute: Test whether the current absolute residual is smaller than a specified tolerance.
"""
@enum Convergence begin
    standard = 1
    paige_saunders = 2
    absolute = 3
end

# matT for iterate room to work in or AbstractArray{T} and opT for A?
mutable struct MPGMRESShIterable{solT, rhsT, matT, barnoldiT <: BlockArnoldiDecomp, residualT <: Residual, resT <: Real}
    x::Array{solT, 1} # array containing the solutions for every shift σ
    b::rhsT # fixed right-hand side
    Wspace::matT # room to work in for expansions and block orthogonalization

    barnoldi::barnoldiT # all relevant data defining the Block Arnoldi relation and for computing the iterates
    residual::residualT

    k::Int  # iteration number within the current cycle
    maxiter::Int # maximal number of iterations
    cycleit::Int # maximal number of iterations per restart cycle

    btol::resT # relative tolerance for convergence tests
    atol::resT # relative tolerance for the matrix norm part for the Paige-Saunders convergence test 

    convergence::Convergence # convergence criterion for stopping the iteration
    
    mv_products::Int # counter for matrix-vector products performed
    precon_solves::Int # counter for preconditioner solves performed

    explicit_residual::Bool # flag signalling whether to compute the residual explicitly 
end

# initialize the iterable MPGMRESSh object 
function mpgmressh_iterable!(x, b, A, M, shifts, preconshifts, precons; 
    btol = sqrt(eps(real(eltype(b)))),
    atol = btol,
    cycleit::Int = min(20, size(M,2)),
    maxiter::Int = size(M,2),
    convergence::Convergence = standard,
    explicit_residual::Bool = false
)
    # !TODO: make sure types of blockdiag entries T and those of V are eventually consistent!
    T = eltype(eltype(x))

    nprecons = length(preconshifts)
    nshifts = length(shifts)

    # build Block Arnoldi structure  
    barnoldi = BlockArnoldiDecomp(A, M, cycleit, shifts, preconshifts, precons, T)
    mv_products = 0
    precon_solves = 0

    # initiate workspace and residual
    Wspace = zeros(size(M, 2), nprecons)
    residuals = Residual(cycleit * nprecons + 1, nshifts, T)
    residuals.β = init!(barnoldi, b, residuals)

    # the iterable starts with parameter k=1 while the 
    # iteration argument in iterate will be initialized with 0
    return MPGMRESShIterable(x, b, Wspace, barnoldi, 
        residuals, 1, maxiter, cycleit,
        btol, atol, convergence,
        mv_products, precon_solves, explicit_residual
    )
end

# initializes the first column of V as normalized rhs 
function init!(barnoldi::BlockArnoldiDecomp{T}, b, residual::Residual) where T
    #first_dir = view(barnoldi.V, :, 1)
    barnoldi.V = [zeros(T, size(barnoldi.V[1], 1), 1)]
    first_dir = barnoldi.V[1]

    copyto!(first_dir, b)
    β = norm(first_dir)
    first_dir .*= inv(β)

    return β
end

# in case A, M and the preconditioning shifts are given explicitly
function mpgmressh_iterable!(x, b, A, M, shifts::Array{shiftT, 1}, preconshifts::Array{shiftT, 1}; kwargs...) where shiftT
    LU = lu(A + preconshifts[1] .* M)
    precons = Array{typeof(LU)}(undef, length(preconshifts))
    #push!(precons, LU)
    precons[1] = LU

    #@printf("mpgmres_iterable length(preconshifts): %d\n",length(preconshifts))

    for i = 2:length(preconshifts)
        #push!(precons, lu(A + preconshifts[i] .* M))
        precons[i] = lu(A + preconshifts[i] .* M)
    end

    mpgmressh_iterable!(x, b, A, M, shifts, preconshifts, precons; kwargs...)
end

# in case A, M are given explicitly, but not the preconditioning shifts
function mpgmressh_iterable!(x, b, A, M, shifts::Array{shiftT, 1}; nprecons = 3, kwargs...) where shiftT 
    # sample nprecons many shifts evenly spaced on a log scale of the range of shifts
    # !TODO: cast shifts back into general shift type
    preconshifts = (10.) .^ (LinRange(log10(minimum(real(shifts))), log10(maximum(real(shifts))), nprecons))
    #@printf("mpgmressh_iterable preconshifts: %d\n", length(preconshifts))

    mpgmressh_iterable!(x, b, A, M, shifts, preconshifts; kwargs...)
end

start(::MPGMRESShIterable) = 0

function converged(gsh::MPGMRESShIterable) 
    has_converged = false
    
    if gsh.convergence == standard
        has_converged = (maximum(gsh.residual.absres ./ (gsh.residual.β)) < gsh.btol)
    end

    if gsh.convergence == paige_saunders
        @printf("Not yet implemented!")
        has_converged = (maximum(gsh.residual.absres) ./ (gsh.residual.β) < gsh.btol)
    end

    if gsh.convergence == absolute
        has_converged = (maximum(gsh.residual.absres) < gsh.btol)
    end

    return has_converged
end

done(gsh::MPGMRESShIterable, iteration::Int) = iteration >= gsh.maxiter || converged(gsh)

function iterate(gsh::MPGMRESShIterable, iteration::Int=start(gsh))
    if done(gsh, iteration)
        # !TODO: develop a method for a proper residual update
        if gsh.explicit_residual
            #@printf("x[1] size: (%d, %d)\n", size(gsh.x[1],1), size(gsh.x,2))
            #@printf("A, M size: (%d, %d), (%d,%d)\n", size(gsh.barnoldi.A, 1), size(gsh.barnoldi.A, 2), size(gsh.barnoldi.M, 1), size(gsh.barnoldi.M, 2))
            #@printf("b size: (%d,%d)\n", size(gsh.b,1), size(gsh.b,2))
            #@printf("residuals.absres size: (%d, %d)\n", size(gsh.residual.absres,1), size(gsh.residual.absres,2))
            gsh.residual.absres .= [norm(gsh.b - (gsh.barnoldi.A + shift .* gsh.barnoldi.M) * gsh.x[i]) for (i, shift) in enumerate(gsh.barnoldi.allshifts)]            
        end
        return nothing
    end

    nprecons = length(gsh.barnoldi.precons)

    # execute multiple preconditioned directions routine
    MPDirections!(gsh.barnoldi, gsh.k, gsh.Wspace)
    gsh.precon_solves += nprecons
    
    global hsigbefore = gsh.barnoldi.H
    # execute the block orthogonalization routine; iteration = gsh.k-1
    #gsh.mv_products += BlockArnoldiStep!(gsh.Wspace, gsh.barnoldi.V, gsh.barnoldi.H[:, (1 + iteration * nprecons):((iteration + 1) * nprecons)])
    gsh.mv_products += BlockArnoldiStep!(gsh.Wspace, gsh.barnoldi.V, view(gsh.barnoldi.H, :, (1 + (gsh.k - 1) * nprecons):(gsh.k * nprecons)))

    # !TODO: develop a method for a proper residual update
    #if gsh.explicit_residual
    #    @printf("x[1] size: (%d, %d)\n", size(gsh.x[1],1), size(gsh.x,2))
    #    @printf("A, M size: (%d, %d), (%d,%d)\n", size(gsh.barnoldi.A, 1), size(gsh.barnoldi.A, 2), size(gsh.barnoldi.M, 1), size(gsh.barnoldi.M, 2))
    #    @printf("b size: (%d,%d)\n", size(gsh.b,1), size(gsh.b,2))
    #    @printf("residuals.absres size: (%d, %d)\n", size(gsh.residual.absres,1), size(gsh.residual.absres,2))
    #    gsh.residual.absres .= [norm(gsh.b - (gsh.barnoldi.A + shift .* gsh.barnoldi.M) * gsh.x[i]) for (i, shift) in enumerate(gsh.barnoldi.allshifts)]
    #end

    # !TODO: compute iterate only after restart cycle length is concluded once
    # an efficient method for residual computation is found 

    # !TODO: cleanly externalize iterate computation

    # compute solution iterate using fast Hessenberg
    for (i, σ) in enumerate(gsh.barnoldi.allshifts)
        # solve minimal residual problem for σ
        y = solve_shifted_lsq(gsh.barnoldi, σ, gsh.residual.β, gsh.k)
        #@printf("barnoldi.Z[:,1:size(y,1)-1] size: (%d,%d)\n", size(gsh.barnoldi.Z[:, 1:size(y, 1) - 1], 1), size(gsh.barnoldi.Z[:, 1:size(y, 1) - 1], 2))
        #@printf("y[1:end-1] size: (%d, %d)\n", size(y,1) - 1, size(y,2) - 1)
        gsh.x[i] = gsh.barnoldi.Z[:, 1:size(y, 1) - 1] * y[1:end-1]
        #display(gsh.x[i])
        gsh.mv_products += 1
    end

    gsh.k += 1

    # restart when cycle length is reached
    if gsh.k == gsh.cycleit + 1
        #@printf("Ended cycle!\n")
        gsh.k = 1
        gsh.residual.β = init!(gsh.barnoldi, gsh.b, gsh.residual)
    end    

    gsh.residual.absres, iteration + 1
end

global hsigafter = zeros(Float64,31,30)
global hsig = zeros(Float64, 31, 30)

function solve_shifted_lsq(barnoldi::BlockArnoldiDecomp{T}, σ::shiftT, β::resT, k::Int) where {T, shiftT, resT}
    nprecons = length(barnoldi.precons)
    # !TODO: optimize using views
    Hσ = IterativeSolvers.FastHessenberg(barnoldi.E[1:(1 + nprecons * k), 1:(nprecons * k)] + (barnoldi.H[1:(1 + nprecons * k), 1:(nprecons * k)] * (UniformScaling(σ) - barnoldi.T[1:(nprecons * k), 1:(nprecons * k)])))
    e1 = zeros(T, size(Hσ, 1), 1) # rhs has as many rows as Hσ
    e1[1] = β

    global hsigafter = Matrix(barnoldi.H)
    global hsig = deepcopy(Hσ)

    ldiv!(Hσ, e1)

    return e1
end

"""
BlockArnoldi(W, V, H)

Execute one step of the (selective) block Arnoldi algorithm given the iterate W and 
a list of previous pairwise orthonormal matrices V and a section of an upper
Hessenberg matrix H.
"""
# !TODO: rewrite this method using only QR factorization objects for V
function BlockArnoldiStep!(W::Matrix{T}, V::Array{Matrix{T}, 1}, H::StridedVecOrMat{T}) where T
    # assumption: no. of columns in H is fixed by the no. of columns of W 
    # number of rows may vary as the no. of columns in each Block of V varies
    cumulsize = 0 # index of last filled row of Hessenberg matrix 
    mv_products = 0 # number of matrix-vector products

    for (i,v) in enumerate(V)
        bsize = size(v, 2) # blocksize is function of BlockDiagonals
        #@printf("size(V): (%d, %d)\n", size(v,1), size(v,2))
        #@printf("size(W): (%d, %d)\n", size(W,1), size(W,2))
        H[cumulsize+1:cumulsize+bsize, :] = adjoint(v) * W 
        mv_products += bsize * size(W, 2)
        W = W - v * H[cumulsize+1:cumulsize+bsize, :]
        cumulsize += bsize
    end
    fact = qr(W) # thin qr-factorization without pivoting or rank reveal for now 
    Q = Matrix(fact.Q)
    Q = Q[:, 1:size(W, 2)]

    H[cumulsize+1:cumulsize+size(fact.R, 1), :] = Matrix(fact.R)
    push!(V, Q)

    return mv_products
end

function MPDirections!(barnoldi::BlockArnoldiDecomp, k::Int, Wspace)
    #v = view(barnoldi.V, :, k)
    m = size(barnoldi.V[k], 2)
    v = view(barnoldi.V[k], :, m)
    nprecons = length(barnoldi.precons)
    #@printf("nprecons: %d\n", nprecons)

    # fill search directions for iteration k with precon solves
    for (i, precon) in enumerate(barnoldi.precons)
        ldiv!(view(Wspace, :, i), precon, v)
    end

    #display("WSpace size: " * string(size(Wspace)))
    #display("barnoldi.Z indices: " * string((size(barnoldi.Z, 1),(k-1)*nprecons + 1, k*nprecons)))
    
    #@printf("Wspace = \n")
    #display(Wspace)

    # add the search directions to Z
    #copyto!(barnoldi.Z[:,((k-1)*nprecons + 1):(k*nprecons)], Wspace)
    copyto!(view(barnoldi.Z, :, ((k-1)*nprecons + 1):(k*nprecons)), Wspace)
    #@printf("barnoldi.Z[:,((k-1)*nprecons + 1):(k*nprecons)]: \n")
    #display(barnoldi.Z[:, ((k-1)*nprecons + 1):(k*nprecons)])
end

function mpgmressh(b, A, M, shifts; kwargs...) where solT
    # until I can think of something better, use a divtype for M
    # this should probably be replaced by some generic operator type 
    T = typeof(one(eltype(b))/one(eltype(M)))
    
    x = [similar(b, T)]
    fill!(x[1], zero(T))
    for k=2:length(shifts)
        push!(x, similar(b,T))
        fill!(x[k], zero(T))
    end

    display(typeof(x))

    return mpgmressh!(x, b, A, M, shifts; kwargs...)
end

# !TODO: adjust up mpgmres! calls to the different constructors for 
# the iterable object
# !TODO: provide option to supply (A+shift.*M) as operators
function mpgmressh!(x, b, A, M, shifts;
    nprecons = 3,
    btol = sqrt(eps(real(eltype(b)))),
    atol = btol,
    cycleit::Int = min(20, size(M,2)),
    maxiter::Int = size(M,2),
    convergence::Convergence = standard,
    explicit_residual::Bool = false,
    log::Bool = false,
    verbose::Bool = false
)
    # !TODO: create IterativeSolvers.history objects for tracking convergence history

    iterable = mpgmressh_iterable!(x, b, A, M, shifts; nprecons = nprecons,
    btol = btol, atol = atol, cycleit = cycleit, maxiter = maxiter,
    convergence = convergence, explicit_residual = explicit_residual)

    for (it, res) in enumerate(iterable)
        verbose && @printf("%3d\t%3d\t%1.2e\n", 1 + div(it - 1, cycleit), 1 + mod(it - 1, cycleit), maximum(res))
    end

    log ? (x, iterable) : x
end

"""
function mpgmressh_iterable!(x, b, A, M, shifts, preconshifts, precons; 
    btol = sqrt(eps(real(eltype(b)))),
    atol = btol,
    cycleit::Int = min(20, size(M,2)),
    maxiter::Int = size(M,2),
    convergence = Convergence.standard,
    explicit_residual = false
"""