import Base: iterate 
using Printf, BlockDiagonals, LinearAlgebra, IterativeSolvers, SuiteSparse, Statistics
export mpgmressh, mpgmressh!, Convergence

# until a package structure is established
include("preconditioner.jl")

# This code draws heavily from the style already adopted for gmres.jl of IterativeSolvers.jl.

# !TODO: make BlockArnoldi immutable by making BlockArnoldiStep work index-bound in-place
mutable struct BlockArnoldiDecomp{elT, opA, opM, shiftT, precT}
    A::opA # (finite-dimensional) linear operator A
    M::opM
    V::Array{Matrix{elT}, 1} # orthonormal part of the Block Arnoldi decomposition in the form of matrix blocks
    Z::Matrix{elT} # search directions
    H::Matrix{elT} # shift-independent part of the Hessenberg matrices
    E::Matrix{elT} # E,T: auxiliary matrices to recover the Arnoldi relations for each shift σ
    T::BlockDiagonal{shiftT, Matrix{shiftT}}

    precons::Array{precT, 1} # shift-and-invert preconditioners
    currentprecons::Array{Int16, 1} # list of precon indices to be used for the next search direction expansion 
    # (relevant for using the mpgmressh routines to implement the FGMRESSh method)
    allshifts::Array{shiftT, 1} # all shift parameters σ
    preconshifts::Array{shiftT, 1} # preconditioning shifts used to build the precons
    nprecons::Int64 # no. of preconditioners used to expand te search space at every iteration 
    npreconshifts::Int64 # no. of shifts used to construct the precons 
end
# !TODO: figure out a sensible name for the type parameter T conflicting with the matrix name T
function BlockArnoldiDecomp(A::opA, M::opM, maxiter::Int, allshifts::Array{shiftT, 1}, nprecons::Int64, preconshifts::Array{shiftT, 1}, precons::Array{precT, 1}, currentprecons::Array{Int16, 1}, elT::Type) where {opA, opM, shiftT, precT}
    H = zeros(elT, nprecons * maxiter + 1, nprecons * maxiter)
    V = [zeros(elT, size(M, 1), 1)]
    Z = zeros(elT, size(M, 1), nprecons * maxiter)

    # E is a block diagonal of the form 
    # < -  nprecons * maxiter   - >
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
    E1 = BlockDiagonal([ones(elT, 1, nprecons),[Ek for k=2:maxiter]...])
    E = zeros(elT, size(H)...)
    E[1:end - nprecons, :] = Matrix(E1)

    # !TODO: find a way to represent the blockdiagonal in terms of 
    # Diagonal Types
    #T = BlockDiagonal([Matrix(Diagonal(preconshifts)) for k=1:maxiter])
    T = BlockDiagonal([Matrix(Diagonal(preconshifts[1:nprecons])) for k=1:maxiter])
    #display(typeof(T))

    return BlockArnoldiDecomp(A, M, V, Z, H, E, T, precons, currentprecons, allshifts, preconshifts, nprecons, length(preconshifts))
end

mutable struct Residual{elT, resT}
    absres::Array{resT, 1}  # Array of absolute residuals for each shift σ
    current::Array{resT, 1} # Placeholder for updating the residual per iteration for each shift σ
    vec::Matrix{elT} # solution of Hkσ[2:end,:]^H * u = Hkσ[1,:]^H at every iteration (see update_residual)
    accumulator::Array{resT, 1} # accumulator for each shift for computing the residuals via norm2(u)**2
    flag::BitArray{1} # flag indicating which shifted systems are assumed to have converged
    β::resT # initial residual which is the norm of b since the initial iterate is set to 0
end

Residual(order::Int, nshifts::Int, elT::Type) = Residual{elT, real(elT)}(
    #ones(elT, nshifts),
    Inf .* ones(real(elT), nshifts),
    ones(elT, nshifts),
    ones(elT, order, nshifts),
    ones(elT, nshifts),
    falses(nshifts),
    one(real(elT))
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

    btol::resT # relative tolerance for convergence tests
    atol::resT # relative tolerance for the matrix norm part for the Paige-Saunders convergence test 

    convergence::Convergence # convergence criterion for stopping the iteration
    
    mv_products::Int # counter for matrix-vector products performed
    precon_solves::Int # counter for preconditioner solves performed

    explicit_residual::Bool # flag signalling whether to compute the residual explicitly 
end

# initialize the iterable MPGMRESSh object 
# nprecons default=length(precons)=length(preconshifts)
# otherwise set explicitly for FGMRESSh and variants thereof
function mpgmressh_iterable!(x, b, A, M, shifts, preconshifts, precons, nprecons = length(precons);
    btol = sqrt(eps(real(eltype(b)))),
    atol = btol,
    maxiter::Int = size(M,2),
    convergence::Convergence = standard,
    explicit_residual::Bool = false
)
    # !TODO: make sure types of blockdiag entries T and those of V are eventually consistent!
    T = eltype(eltype(x))

    nshifts = length(shifts)

    # set preconditioner indices to be used for BlockArnoldi expansion 
    currentprecons = collect(Int16, 1:nprecons)

    # build Block Arnoldi structure  
    barnoldi = BlockArnoldiDecomp(A, M, maxiter, shifts, nprecons, preconshifts, precons, currentprecons, T)
    mv_products = 0
    precon_solves = 0

    # initiate workspace and residual
    Wspace = zeros(T, size(M, 2), nprecons)
    residuals = Residual(maxiter * nprecons + 1, nshifts, T)
    β = init!(barnoldi, b, residuals)
    init_residual!(residuals, β)

    # the iterable starts with parameter k=1 while the 
    # iteration argument in iterate will be initialized with 0
    return MPGMRESShIterable(x, b, Wspace, barnoldi, 
        residuals, 1, maxiter,
        btol, atol, convergence,
        mv_products, precon_solves, explicit_residual
    )
end

# initializes the first column of V as normalized rhs 
function init!(barnoldi::BlockArnoldiDecomp{T}, b, residual::Residual) where T
    barnoldi.V = [zeros(T, size(barnoldi.V[1], 1), 1)]
    first_dir = barnoldi.V[1]

    copyto!(first_dir, b)
    β = norm(first_dir)
    first_dir .*= inv(β)

    return β
end

# generate the preconditioning shifts equidistantly log-spaced in the set of shifts
function generate_preconshifts(shifts::Array{shiftT, 1}, npreconshifts) where shiftT 
    # sample nprecons many shifts evenly spaced on a log scale of the range of shifts
    # shiftT's can be real or complex
    # in the case of nprecons = 1, we simply take an average value
    if shiftT <: Complex 
        # extract the machine epsilon for the given shift datatype 
        paramType = fieldtype(shiftT, 1)
        ε = eps(paramType)
    else
        paramtype = shiftT
        ε = eps(shiftT)
    end

    # shift real and imaginary parts (if nonempty) both by their minimum
    # values to move the ranges above zero, sample equidistantly log-spaced 
    # and then shift back into the the original range 
    minshiftsIm = minimum(imag.(shifts))
    maxshiftsIm = maximum(imag.(shifts))
    minshiftsRe = minimum(real.(shifts))
    maxshiftsRe = maximum(real.(shifts))

    if npreconshifts > 1
        biasIm = max(ε, abs(minshiftsIm))
        biasRe = max(ε, abs(minshiftsRe))

        if minshiftsIm == maxshiftsRe
            preconshiftsIm = zeros(paramType, npreconshifts)
        else
            if minshiftsIm < 0 
                preconshiftsIm = -2*abs(minshiftsIm) .+ (10.) .^ LinRange(log10(biasIm), log10(2*biasIm + maxshiftsIm), npreconshifts)
            else
                preconshiftsIm = (10.) .^ LinRange(log10(biasIm), log10(maxshiftsIm), npreconshifts)
            end
        end

        if minshiftsRe == maxshiftsRe
            preconshiftsRe = zeros(paramType, npreconshifts)
        else
            if minshiftsRe < 0
                preconshiftsRe = -2*abs(minshiftsRe) .* (10.) .^ LinRange(log10(biasRe), log10(2*biasRe + maxshiftsRe), npreconshifts)
            else
                preconshiftsRe = (10.) .^ LinRange(log10(biasRe), log10(maxshiftsRe), npreconshifts) 
            end
        end

        if shiftT <: Complex
            preconshifts = preconshiftsRe + 1.0im .* preconshiftsIm
        else
            preconshifts = preconshiftsRe
        end
    else
        meanRe = mean(real.(shifts))
        meanIm = mean(imag.(shifts))
        if shiftT <: Complex 
            preconshifts = [meanRe + 1.0im * meanIm]
        else
            preconshifts = [meanRe]
        end
    end

    return preconshifts
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
        return nothing
    end

    # execute multiple preconditioned directions routine
    gsh.precon_solves += MPDirections!(gsh.barnoldi, gsh.k, gsh.Wspace)

    # execute the block orthogonalization routine
    for col in eachcol(gsh.Wspace)
        # iterate through the columns of Wspace to keep the 
        # iterative method matrix free
        col .= gsh.barnoldi.M * col
        gsh.mv_products += 1
    end    
    gsh.mv_products += BlockArnoldiStep!(gsh.Wspace, gsh.barnoldi.V, view(gsh.barnoldi.H, :, (1 + (gsh.k - 1) * gsh.barnoldi.nprecons):(gsh.k * gsh.barnoldi.nprecons)))

    if gsh.explicit_residual
        for (i, σ) in enumerate(gsh.barnoldi.allshifts)
            y = solve_shifted_lsq(gsh.barnoldi, σ, gsh.residual.β, gsh.k)
            gsh.x[i] = gsh.barnoldi.Z[:, 1:size(y, 1) - 1] * y[1:end-1]
            gsh.mv_products += 1
        end
        gsh.residual.absres .= [norm(gsh.b - (gsh.barnoldi.A + shift * gsh.barnoldi.M) * gsh.x[i]) for (i, shift) in enumerate(gsh.barnoldi.allshifts)]
    else
        gsh.mv_products += update_residual!(gsh.residual, gsh.barnoldi, gsh.k, gsh.btol)
        # use the Ayachour method for convergence monitoring
        copyto!(gsh.residual.absres, gsh.residual.current)
    end

    if done(gsh, iteration + 1)
        # compute the iterates after residual monitoring indicates convergence 
        # or the maximal number of iterations is reached 
        # compute solution iterate using fast Hessenberg
        for (i, σ) in enumerate(gsh.barnoldi.allshifts)
            # solve minimal residual problem for σ
            y = solve_shifted_lsq(gsh.barnoldi, σ, gsh.residual.β, gsh.k)
            gsh.x[i] = gsh.barnoldi.Z[:, 1:size(y, 1) - 1] * y[1:end-1]
            gsh.mv_products += 1
        end
        if gsh.explicit_residual
            gsh.residual.absres .= [norm(gsh.b - (gsh.barnoldi.A + shift * gsh.barnoldi.M) * gsh.x[i]) for (i, shift) in enumerate(gsh.barnoldi.allshifts)]
        end
        return nothing
    end

    gsh.k += 1

    gsh.residual.absres, iteration + 1
end

function solve_shifted_lsq(barnoldi::BlockArnoldiDecomp{T}, σ::shiftT, β::resT, k::Int) where {T, shiftT, resT}
    # !TODO: optimize using views
    Hσ = IterativeSolvers.FastHessenberg(barnoldi.E[1:(1 + barnoldi.nprecons * k), 1:(barnoldi.nprecons * k)] + (barnoldi.H[1:(1 + barnoldi.nprecons * k), 1:(barnoldi.nprecons * k)] * (UniformScaling(σ) - barnoldi.T[1:(barnoldi.nprecons * k), 1:(barnoldi.nprecons * k)])))
    e1 = zeros(T, size(Hσ, 1), 1) # rhs has as many rows as Hσ
    e1[1] = β
        
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
        bsize = size(v,2)
        copyto!(view(H, cumulsize + 1:cumulsize + bsize, :), adjoint(v) * W)
        mv_products += bsize * size(W,2)
        W = W - v * H[cumulsize+1:cumulsize+bsize, :]
        mv_products += size(v,1) * size(W,2)
        cumulsize += bsize
    end
    fact = qr(W)
    #fact2 = qr(W,Val(true)) # with pivoting to detect rank deficiencies
    #if rank(fact2.R)< size(fact2.R,2)
    #    @printf("Rank deficiency! Permutation:")
    #    display(fact2.p)
    #end
    Q = Matrix(fact.Q)
    Q = Q[:, 1:size(W, 2)]

    H[cumulsize+1:cumulsize+size(fact.R, 1), :] = Matrix(fact.R)
    push!(V, Q)

    return mv_products
end

function MPDirections!(barnoldi::BlockArnoldiDecomp, k::Int, Wspace)
    m = size(barnoldi.V[k], 2)
    v = view(barnoldi.V[k], :, m)
   
    solves = 0

    # support parallel precon solves for the LU case 
    # iterative solvers are not thread-save yet
    # ! assumption: all preconditioners are of the same type 
    if barnoldi.precons[1].method == LUFac
        Threads.@threads for i=1:length(barnoldi.currentprecons)
            index = barnoldi.currentprecons[i]
            ldiv!(view(Wspace, :, i), barnoldi.precons[index], v)
            solves += 1
        end
    else
        for (i, index) in enumerate(barnoldi.currentprecons)
            ldiv!(view(Wspace, :, i), barnoldi.precons[index], v)
            solves += 1
        end    
    end

    # add the search directions to Z
    copyto!(view(barnoldi.Z, :, ((k-1)*barnoldi.nprecons + 1):(k*barnoldi.nprecons)), Wspace)
   
    return solves
end

function init_residual!(r::Residual{elT, resT}, β) where {elT, resT}
    r.accumulator = zeros(elT, size(r.accumulator))
    view(r.accumulator, 1, :) .= β * one(resT)
    r.β = β
end

# compute the current residual according to the ideas laid out in 
# [Ayachour E.H., A fast implementation for GMRES method](https://ris.utwente.nl/ws/files/26304887/fast.pdf)
# which is free of any Givens rotations and more accurate on the test examples
function update_residual!(r::Residual, barnoldi::BlockArnoldiDecomp, k::Int, btol::Real)
    mv_products = 0
    for (i, σ) in enumerate(barnoldi.allshifts)
        if r.flag[i]
            continue
        end
 
        Hkσ = barnoldi.H[1:(k * barnoldi.nprecons + 1), ((k-1) * barnoldi.nprecons + 1):(k * barnoldi.nprecons)] * (UniformScaling(σ) - BlockDiagonals.getblock(barnoldi.T, BlockDiagonals.nblocks(barnoldi.T)))
        mv_products += size(Hkσ,1) * size(Hkσ,2)
        
        Hkσ[end - barnoldi.nprecons,:] .+= 1.0

        cols = size(Hkσ, 2)
        
        offset = barnoldi.nprecons * (k - 1) + 1
        
        for j=1:cols
            offset = barnoldi.nprecons * (k - 1) + 1

            # initialize first vec entry
            if k==1 && j==1
                if Hkσ[2,1] == 0
                    r.flag[i] = true
                    r.current[i] = 0.
                    break
                end
                r.vec[1, i] = conj(Hkσ[1,1] / Hkσ[2,1])
                r.accumulator[i] += abs2(r.vec[1, i])
                r.current[i] = r.β / √r.accumulator[i]
                continue                
            end

            # heuristic: assume convergence (lucky breakdown) for a zero subdiag element
            # presently, this only happens in tests
            # due to some precon shifts matching input shifts producing zero columns 
            # in the Hessenberg matrix

            if Hkσ[offset + j, j] == 0
                r.flag[i] = true
                r.current[i] = 0.
                break
            end

            r.vec[offset + j - 1, i] = conj((Hkσ[1,j] - dot(r.vec[1:offset + j - 2, i], Hkσ[2:offset + j - 1, j]))/Hkσ[offset + j, j])
            mv_products += 1
            r.accumulator[i] += abs2(r.vec[offset + j - 1, i])
            r.current[i] = r.β / √r.accumulator[i]

            # if the absolute residual is below btol, we consider the 
            # system converged
            if abs(r.current[i]) < btol
                r.flag[i] = true
            end
        end
    end

    return mv_products  
end

function mpgmressh(b, A, M, shifts; kwargs...)
    # use the appropriate divtype for M
    T = typeof(one(eltype(b))/(one(eltype(shifts)) * one(eltype(M))))

    # !TODO implement sanity checks for proper 
    # dimensionality of b, A, M    
    x = [similar(b, T)]
    fill!(x[1], zero(T))
    for k=2:length(shifts)
        push!(x, similar(b,T))
        fill!(x[k], zero(T))
    end

    return mpgmressh!(x, b, A, M, shifts; kwargs...)
end


function mpgmressh!(x, b, A, M, shifts;
    nprecons = 3,
    btol = sqrt(eps(real(eltype(b)))),
    atol = btol,
    maxiter::Int = size(M,2),
    convergence::Convergence = standard,
    explicit_residual::Bool = false,
    log::Bool = false,
    verbose::Bool = false,
    preconmethod = LUFac,
    preconmaxiter = size(A,2),
    preconrestart = min(20, size(A,2)),
    preconAMG = false,
    preconreltol = btol 
)
    # create IterativeSolvers.history objects for tracking convergence history
    if log
        history = IterativeSolvers.ConvergenceHistory(partial = !log)
        history[:btol] = btol
        history[:explicit_residual] = explicit_residual
        # store precon solve info
        history[:preconmethod] = preconmethod
        history[:preconmaxiter] = preconmaxiter
        history[:preconrestart] = preconrestart
        history[:preconAMG] = preconAMG
        history[:preconreltol] = preconreltol
        # reserve residual info 
        IterativeSolvers.reserve!(history, :resmat, maxiter, length(shifts))
        
        setup_time = time_ns()
    end
    # construct preconditioners
    preconshifts = generate_preconshifts(shifts, nprecons)
    precons = generate_preconditioners(A, M, preconshifts, preconmethod, maxiter = preconmaxiter, 
    restart = preconrestart, AMG = preconAMG, reltol = preconreltol)
    
    # instantiate iterable 
    global iterable = mpgmressh_iterable!(x, b, A, M, shifts, preconshifts,
    precons, nprecons; btol = btol, atol = atol, maxiter = maxiter,
    convergence = convergence, explicit_residual = explicit_residual)

    if log 
        setup_time = time_ns() - setup_time
        history[:setup_time] = Int(setup_time)/(1e9)

        iteration_time = time_ns()

        # set up convergence flag array
        IterativeSolvers.reserve!(Int64, history, :conv_flags, 1, length(shifts))

        old_flags = falses(length(shifts))
        new_flags = falses(length(shifts))

        old_flags .= iterable.residual.flag
    end
    
    for (it, res) in enumerate(iterable)
        if log
            new_flags .= iterable.residual.flag
            IterativeSolvers.nextiter!(history)
            history[:resmat][history.iters,:] .= iterable.residual.current
            history[:conv_flags][findall(x->x==true, xor.(old_flags,new_flags))] .= history.iters
            old_flags .= new_flags
        end
        verbose && @printf("%3d\t%1.2e\n", 1 + mod(it - 1, maxiter), maximum(res))
    end
    
    if log 
        # the last step of the for loop calls iterate() one more time,
        # so the last convergence history step needs to be performed here
        new_flags .= iterable.residual.flag
        IterativeSolvers.nextiter!(history)
        history[:resmat][history.iters,:] .= iterable.residual.current
        history[:conv_flags][findall(x->x==true, xor.(old_flags,new_flags))] .= history.iters
        old_flags .= new_flags

        iteration_time = time_ns() - iteration_time
        history[:iteration_time] = Int(iteration_time)/(1e9)
        history.mvps = iterable.mv_products

        history[:resmat] = history[:resmat][1:history.iters,:]
        IterativeSolvers.setconv(history, converged(iterable))
    end

    log ? (x, iterable, history) : x
end
