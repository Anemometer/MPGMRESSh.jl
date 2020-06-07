import Base: iterate 
using Printf, BlockDiagonals, LinearAlgebra, IterativeSolvers, SuiteSparse, Statistics
export mpgmressh, mpgmressh!, Convergence

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
    #preconshifts::Array{shiftT, 1} # shift parameters to be used for building the shift-and-invert preconditioners (optional if preconditioners are passed by the user)
end
# !TODO: figure out a sensible name for the type parameter T conflicting with the matrix name T
function BlockArnoldiDecomp(A::opA, M::opM, maxiter::Int, allshifts::Array{shiftT, 1}, nprecons::Int64, preconshifts::Array{shiftT, 1}, precons::Array{precT, 1}, currentprecons::Array{Int16, 1}, elT::Type) where {opA, opM, shiftT, precT}
    #nprecons = length(preconshifts)
    H = zeros(elT, nprecons * maxiter + 1, nprecons * maxiter)
    #V = zeros(T, size(M, 1), nprecons * maxiter + 1)
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

    #nprecons = length(preconshifts)
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
    #first_dir = view(barnoldi.V, :, 1)
    barnoldi.V = [zeros(T, size(barnoldi.V[1], 1), 1)]
    first_dir = barnoldi.V[1]

    copyto!(first_dir, b)
    β = norm(first_dir)
    first_dir .*= inv(β)

    return β
end

# in case A, M and the preconditioning shifts are given explicitly
# for FGMRESSh support: provide the option to set npreconshifts explicitly 
# which is the number of preconditioners used in the BlockArnoldi expansion
function mpgmressh_iterable!(x, b, A, M, shifts::Array{shiftT, 1}, preconshifts::Array{shiftT, 1}, nprecons::Int64; kwargs...) where shiftT
    LU = lu(A + preconshifts[1] .* M)
    #nprecons = length(preconshifts)
    npreconshifts = length(preconshifts)
    #precons = Array{typeof(LU)}(undef, nprecons)
    precons = Array{typeof(LU)}(undef, npreconshifts)
    precons[1] = LU

    #if nprecons >= 2
    #    for i = 2:nprecons
    #        precons[i] = lu(A + preconshifts[i] .* M)
    #    end
    #end
    if npreconshifts >= 2
        for i = 2:npreconshifts
            precons[i] = lu(A + preconshifts[i] .* M)
        end
    end

    mpgmressh_iterable!(x, b, A, M, shifts, preconshifts, precons, nprecons; kwargs...)
end

# in case A, M are given explicitly, but not the preconditioning shifts
function mpgmressh_iterable!(x, b, A, M, shifts::Array{shiftT, 1}; nprecons = 3, npreconshifts = nprecons, kwargs...) where shiftT 
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


        #preconshiftsRe = (10.) .^ (LinRange(log10(minimum(real.(shifts))), log10(maximum(real.(shifts))), nprecons))
        #preconshiftsIm = (10.) .^ (LinRange(log10(minimum(imag.(shifts))), log10(maximum(imag.(shifts))), nprecons))
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
    #@printf("mpgmressh_iterable preconshifts: %d\n", length(preconshifts))
    preshifts = preconshifts

    #preconshifts = 1im .* collect(LinRange(minimum(imag.(shifts)), maximum(imag.(shifts)), nprecons))
    mpgmressh_iterable!(x, b, A, M, shifts, preconshifts, nprecons; kwargs...)
end

start(::MPGMRESShIterable) = 0

function converged(gsh::MPGMRESShIterable) 
    has_converged = false
    
    if gsh.convergence == standard
        #@printf("absres, β, btol: %1.5e, %1.5e, %1.5e\n", maximum(gsh.residual.absres), gsh.residual.β, gsh.btol)
        #@printf("absres/β: %1.20e\n", maximum(gsh.residual.absres ./ gsh.residual.β))
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
#done(gsh::MPGMRESShIterable, iteration::Int) = iteration >= gsh.maxiter 

function iterate(gsh::MPGMRESShIterable, iteration::Int=start(gsh))
    if done(gsh, iteration)
        return nothing
    end

    #nprecons = length(gsh.barnoldi.precons)

    # execute multiple preconditioned directions routine
    MPDirections!(gsh.barnoldi, gsh.k, gsh.Wspace)
    gsh.precon_solves += gsh.barnoldi.nprecons
    
    global hsigbefore = gsh.barnoldi.H
    # execute the block orthogonalization routine
    gsh.Wspace = gsh.barnoldi.M * gsh.Wspace
    gsh.mv_products += BlockArnoldiStep!(gsh.Wspace, gsh.barnoldi.V, view(gsh.barnoldi.H, :, (1 + (gsh.k - 1) * gsh.barnoldi.nprecons):(gsh.k * gsh.barnoldi.nprecons)))

    #if gsh.explicit_residual
    #    @printf("x[1] size: (%d, %d)\n", size(gsh.x[1],1), size(gsh.x,2))
    #    @printf("A, M size: (%d, %d), (%d,%d)\n", size(gsh.barnoldi.A, 1), size(gsh.barnoldi.A, 2), size(gsh.barnoldi.M, 1), size(gsh.barnoldi.M, 2))
    #    @printf("b size: (%d,%d)\n", size(gsh.b,1), size(gsh.b,2))
    #    @printf("residuals.absres size: (%d, %d)\n", size(gsh.residual.absres,1), size(gsh.residual.absres,2))
    #    gsh.residual.absres .= [norm(gsh.b - (gsh.barnoldi.A + shift .* gsh.barnoldi.M) * gsh.x[i]) for (i, shift) in enumerate(gsh.barnoldi.allshifts)]
    #end

    #hh = update_residual!(gsh.residual, gsh.barnoldi, gsh.k)
    update_residual!(gsh.residual, gsh.barnoldi, gsh.k, gsh.btol)

    # experimental: use the newly implemented Ayachour method for convergence monitoring
    copyto!(gsh.residual.absres, gsh.residual.current)

    if done(gsh, iteration + 1)
        # compute the iterates after residual monitoring indicates convergence 
        # or the maximal number of iterations is reached 
        # compute solution iterate using fast Hessenberg
        for (i, σ) in enumerate(gsh.barnoldi.allshifts)
            # solve minimal residual problem for σ
            y, res = solve_shifted_lsq(gsh.barnoldi, σ, gsh.residual.β, gsh.k)
            #@printf("shifted lsq residual at %d, %f: %f\n", i, imag(σ), res)
            
            # check if hh contains the right entries 
            #@printf("hsig[end-3:end, end-2:end] == hh[:,(i-1)*3 + 1:i*3]? %d \n", hsig.H[end-3:end,end-2:end] == hh[:,(i-1)*3 + 1:i*3])
            #@printf("hsig[end - 1, end] == h? %d\n", hsig.H[end-1,end] == hh[1,i])
            #@printf("hsig[:, end-2:end] == hh[:,(i-1)*3 + 1:i*3]? %d \n", hsig.H[:,end-2:end] == hh[:,(i-1)*3 + 1:i*3])
            
            #@printf("barnoldi.Z[:,1:size(y,1)-1] size: (%d,%d)\n", size(gsh.barnoldi.Z[:, 1:size(y, 1) - 1], 1), size(gsh.barnoldi.Z[:, 1:size(y, 1) - 1], 2))
            #@printf("y[1:end-1] size: (%d, %d)\n", size(y,1) - 1, size(y,2) - 1)
            gsh.x[i] = gsh.barnoldi.Z[:, 1:size(y, 1) - 1] * y[1:end-1]
            #display(gsh.x[i])
            gsh.mv_products += 1
        end
        if gsh.explicit_residual
            #@printf("x[1] size: (%d, %d)\n", size(gsh.x[1],1), size(gsh.x,2))
            #@printf("A, M size: (%d, %d), (%d,%d)\n", size(gsh.barnoldi.A, 1), size(gsh.barnoldi.A, 2), size(gsh.barnoldi.M, 1), size(gsh.barnoldi.M, 2))
            #@printf("b size: (%d,%d)\n", size(gsh.b,1), size(gsh.b,2))
            #@printf("residuals.absres size: (%d, %d)\n", size(gsh.residual.absres,1), size(gsh.residual.absres,2))
            gsh.residual.absres .= [norm(gsh.b - (gsh.barnoldi.A + shift .* gsh.barnoldi.M) * gsh.x[i]) for (i, shift) in enumerate(gsh.barnoldi.allshifts)]            
        end
        return nothing
    end

    gsh.k += 1

    gsh.residual.absres, iteration + 1
end

function solve_shifted_lsq(barnoldi::BlockArnoldiDecomp{T}, σ::shiftT, β::resT, k::Int) where {T, shiftT, resT}
    #@printf("lsq_solve: k = %d \n", k)
    #nprecons = length(barnoldi.precons)
    # !TODO: optimize using views
    Hσ = IterativeSolvers.FastHessenberg(barnoldi.E[1:(1 + barnoldi.nprecons * k), 1:(barnoldi.nprecons * k)] + (barnoldi.H[1:(1 + barnoldi.nprecons * k), 1:(barnoldi.nprecons * k)] * (UniformScaling(σ) - barnoldi.T[1:(barnoldi.nprecons * k), 1:(barnoldi.nprecons * k)])))
    #Hσ = qr(barnoldi.E[1:(1 + nprecons * k), 1:(nprecons * k)] + (barnoldi.H[1:(1 + nprecons * k), 1:(nprecons * k)] * (UniformScaling(σ) - barnoldi.T[1:(nprecons * k), 1:(nprecons * k)])))
    Hbak = barnoldi.E[1:(1 + barnoldi.nprecons * k), 1:(barnoldi.nprecons * k)] + (barnoldi.H[1:(1 + barnoldi.nprecons * k), 1:(barnoldi.nprecons * k)] * (UniformScaling(σ) - barnoldi.T[1:(barnoldi.nprecons * k), 1:(barnoldi.nprecons * k)]))

    e1 = zeros(T, size(Hσ, 1), 1) # rhs has as many rows as Hσ
    e1[1] = β
    e2 = deepcopy(e1)

    global hsigafter = Matrix(barnoldi.H)
    global hsig = deepcopy(Hσ)
    
    #ldiv!(Hσ, e1)
    ldiv!(Hσ, e1)

    #return e1, norm(b - hsig.H * e1[1:end-1])
    return e1, norm(e2 - Hbak * e1[1:end-1])
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

    #@printf("BStep: types of W, V, H: \n")
    #display(typeof(W))
    #display(typeof(V))
    #display(typeof(H))

    for (i,v) in enumerate(V)
        bsize = size(v, 2) # blocksize is function of BlockDiagonals
        #@printf("size(V): (%d, %d)\n", size(v,1), size(v,2))
        #@printf("size(W): (%d, %d)\n", size(W,1), size(W,2))
        #H[cumulsize+1:cumulsize+bsize, :] = adjoint(v) * W 
        copyto!(view(H, cumulsize + 1:cumulsize + bsize, :), adjoint(v) * W)
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
    #nprecons = length(barnoldi.precons)
    #@printf("nprecons: %d\n", nprecons)

    # fill search directions for iteration k with precon solves
    #for (i, precon) in enumerate(barnoldi.precons)
    #    ldiv!(view(Wspace, :, i), precon, v)
    #end

    #display(barnoldi.currentprecons)
    #display(barnoldi.precons[barnoldi.currentprecons[1]])

    for (i, index) in enumerate(barnoldi.currentprecons)
        ldiv!(view(Wspace, :, i), barnoldi.precons[index], v)
    end

    #display("WSpace size: " * string(size(Wspace)))
    #display("barnoldi.Z indices: " * string((size(barnoldi.Z, 1),(k-1)*nprecons + 1, k*nprecons)))
    
    #@printf("Wspace after MPDirections = \n")
    #display(Wspace)

    # add the search directions to Z
    copyto!(view(barnoldi.Z, :, ((k-1)*barnoldi.nprecons + 1):(k*barnoldi.nprecons)), Wspace)
    #@printf("barnoldi.Z[:,((k-1)*nprecons + 1):(k*nprecons)]: \n")
    #display(barnoldi.Z[:, ((k-1)*nprecons + 1):(k*nprecons)])
end

function init_residual!(r::Residual{elT, resT}, β) where {elT, resT}
    r.accumulator = zeros(elT, size(r.accumulator))
    view(r.accumulator, 1, :) .= β * one(resT)
    r.β = β
    #@printf("accumulator after init: \n")
    #display(r.accumulator)
end

# compute the current residual according to the ideas laid out in 
# [Ayachour E.H., A fast implementation for GMRES method](https://ris.utwente.nl/ws/files/26304887/fast.pdf)
# which is free of any Givens rotations and more accurate on the test examples
function update_residual!(r::Residual, barnoldi::BlockArnoldiDecomp, k::Int, btol::Real)
    # hh saves the isolated Hessenberg parts for debugging purposes
    #global hh = zeros(ComplexF64, k * barnoldi.nprecons + 1, 3*size(barnoldi.allshifts, 1))
    for (i, σ) in enumerate(barnoldi.allshifts)
        if r.flag[i]
            continue
        end
 
        Hkσ = barnoldi.H[1:(k * barnoldi.nprecons + 1), ((k-1) * barnoldi.nprecons + 1):(k * barnoldi.nprecons)] * (UniformScaling(σ) - BlockDiagonals.getblock(barnoldi.T, BlockDiagonals.nblocks(barnoldi.T)))
        
        #Hkσ[end - 3,:] .+= 1.0
        Hkσ[end - barnoldi.nprecons,:] .+= 1.0

        cols = size(Hkσ, 2)
        
        #copyto!(view(hh, :,(i-1)*cols+1:i*cols), Hkσ)

        offset = barnoldi.nprecons * (k - 1) + 1

        #if i==28
        #    @printf("Hk[offset: offset+3, :] for σ = %f\n", imag(σ))
        #    display(Hkσ[offset:offset+3, :])                
        #end

        #if i==55
        #    @printf("Hk[offset: offset+3, :] for σ = %f\n", imag(σ))
        #    display(Hkσ[offset:offset+3, :])
        #end
        
        for j=1:cols
            offset = barnoldi.nprecons * (k - 1) + 1

            # initialize first vec entry
            if k==1 && j==1
                if Hkσ[2,1] == 0
                    r.flag[i] = true
                    r.current[i] = 0.
                    #@printf("Fired Hkk = 0 at i=%d, offset + j - 1 = %d!\n", i, offset + j - 1)
                    break
                end
                r.vec[1, i] = conj(Hkσ[1,1] / Hkσ[2,1])
                r.accumulator[i] += abs2(r.vec[1, i])
                r.current[i] = r.β / √r.accumulator[i]
                continue                
            end

            # for now: assume convergence for a zero subdiag element until 
            # I have a theorem worked out specifically - in tests atm this only happens 
            # due to some precon shifts matching input shifts producing zero columns 
            # in the Hessenberg matrix

            if Hkσ[offset + j, j] == 0
                r.flag[i] = true
                r.current[i] = 0.
                #@printf("Fired Hkk = 0 at i=%d, offset + j - 1 = %d!\n", i, offset + j - 1)
                break
            end

            r.vec[offset + j - 1, i] = conj((Hkσ[1,j] - dot(r.vec[1:offset + j - 2, i], Hkσ[2:offset + j - 1, j]))/Hkσ[offset + j, j])
            r.accumulator[i] += abs2(r.vec[offset + j - 1, i])
            r.current[i] = r.β / √r.accumulator[i]

            # if the absolute residual is below btol, we consider the 
            # system converged
            if abs(r.current[i]) < btol
                #@printf("System i=%d converged with absres=%f\n",i,r.current[i])
                r.flag[i] = true
            end

            #if i==size(barnoldi.allshifts,1)
            #    @printf("setting r.vec[offset+j-1,i]: (%d,%d)\n", offset+j-1, i)
            #end
        end
    end

    #@printf("max acc: %1.2e \n", maximum(r.current))
    #return hh    
end

function mpgmressh(b, A, M, shifts; kwargs...) where solT
    # until I can think of something better, use a divtype for M
    # this should probably be replaced by some generic operator type 
    T = typeof(one(eltype(b))/(one(eltype(shifts)) * one(eltype(M))))

    # !TODO implement sanity checks for proper 
    # dimensionality of b, A, M    
    x = [similar(b, T)]
    fill!(x[1], zero(T))
    for k=2:length(shifts)
        push!(x, similar(b,T))
        fill!(x[k], zero(T))
    end

    #@printf("typeof(x): ")
    #display(typeof(x))
    #@printf("typeof(b): ")
    #display(typeof(b))

    #@printf("typeof(shifts): ")
    #display(typeof(shifts))

    #@printf("typeof(M, A): ")
    #display(typeof(M))
    #display(typeof(A))

    return mpgmressh!(x, b, A, M, shifts; kwargs...)
end

# !TODO: adjust up mpgmres! calls to the different constructors for 
# the iterable object
# !TODO: provide option to supply (A+shift.*M) as operators
function mpgmressh!(x, b, A, M, shifts;
    nprecons = 3,
    btol = sqrt(eps(real(eltype(b)))),
    atol = btol,
    maxiter::Int = size(M,2),
    convergence::Convergence = standard,
    explicit_residual::Bool = false,
    log::Bool = false,
    verbose::Bool = false
)
    # !TODO: create IterativeSolvers.history objects for tracking convergence history

    global iterable = mpgmressh_iterable!(x, b, A, M, shifts; nprecons = nprecons,
    npreconshifts = nprecons, btol = btol, atol = atol, maxiter = maxiter,
    convergence = convergence, explicit_residual = explicit_residual)

    @time for (it, res) in enumerate(iterable)
        verbose && @printf("%3d\t%1.2e\n", 1 + mod(it - 1, maxiter), maximum(res))
    end

    log ? (x, iterable) : x
end
