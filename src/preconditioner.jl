import LinearAlgebra.ldiv!
using IterativeSolvers, LinearAlgebra, SparseArrays
# name clash with GaussSeidel in PreconMethod: use needed methods directly
using AlgebraicMultigrid: ruge_stuben, aspreconditioner
using ExtendableSparse: JacobiPreconditioner, ExtendableSparseMatrix, flush!
using IncompleteLU: ilu

export PreconMethod, generate_preconditioners

@enum PreconMethod begin
    LUFac = 1
    GaussSeidel = 2
    CG = 3
    GMRES = 4
    BiCGStab = 5
end

# generic shift and invert preconditioner data structure 
# to provide standard preconditioner solution methods
mutable struct SaIPreconditioner{shiftT, methoddataT, resT}
    # shift associated with preconditioner
    shift::shiftT
    
    # preconditioner solution method to be applied
    method::PreconMethod
    # solution method metadata prepared in advance such as
    # LU factorizations or Krylov iterables
    methoddata::methoddataT
    tol::resT
end

function SaIPreconditioner(shift::shiftT, method::PreconMethod, methoddata::methoddataT; 
    tol = sqrt(eps(real(eltype(shift))))) where {shiftT, methoddataT}
    return SaIPreconditioner(shift, method, methoddata, tol)
end

# AlgebraicMultigrid.jl currently does not support 
# complex matrices! But in case it shall, the 
# AMG machinery here already supports complex AMG construction.

function constr_selfadjoint(K::AbstractArray{Tv} where {Tv<:Complex})
    return Hermitian(K)
end
constr_selfadjoint(K::AbstractArray{Tv} where {Tv<:Real})=(return Symmetric(K))

function amg_ruge_stuben(K::AbstractArray; kwargs...)
    K = constr_selfadjoint(K)
    ml = ruge_stuben(K; kwargs...)
    return aspreconditioner(ml)
end

function amg_ruge_stuben(K::ExtendableSparseMatrix; kwargs...)
    flush!(K)
    mat = constr_selfadjoint(K.cscmatrix)
    ml = ruge_stuben(mat; kwargs...)
    return aspreconditioner(ml)
end

function jacobi_precon(K::AbstractSparseMatrix; kwargs...)
    return JacobiPreconditioner(K)
end

function incomplete_lu(K::SparseMatrixCSC; kwargs...)
    return ilu(K; kwargs...)
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

# ExtendableSparse does not provide a "+" for ExtendableSparseMatrices 
# or "*" for ExtendableSparseMatrices and scalars returning 
# a sparse structure (see https://github.com/j-fu/ExtendableSparse.jl/issues/7)
function add_shift(A::ExtendableSparseMatrix{Tv,Ti},M::ExtendableSparseMatrix{Tv,Ti},σ::Number) where {Tv,Ti<:Integer}
    @inbounds flush!(A)
    @inbounds flush!(M)
    return A.cscmatrix + σ * M.cscmatrix
end

function add_shift(A,M,σ)
    return A + σ * M
end

# generate preconditioners given A,M and preconditioning shifts to provide 
# preconitioning solves of the desired type
function generate_preconditioners(A, M, preconshifts::Array{shiftT, 1},
    preconmethod::PreconMethod, 
    preconpreconmethod; 
    preconreltol = sqrt(eps(real(eltype(A)))),
    preconmaxiter = size(A,2),
    preconrestart = min(20, size(A,2)),
    kwargs... # arguments provided to preconpreconmethod
    ) where shiftT
    
    npreconshifts = length(preconshifts)
    precons = Array{SaIPreconditioner}(undef, npreconshifts)

    preconpreconsupplied = false
    if !isnothing(preconpreconmethod)
        preconpreconsupplied = true
    end

    # assume preconditioners map from the eltype of (A+shift.*M) to the same type
    T = eltype(preconshifts[1] * M)
    m = size(M, 2)
    # pre-allocate local data to make 
    # pc.methoddata thread-safe
    x = Array{Any,1}(undef,npreconshifts)
    b = Array{Any,1}(undef,npreconshifts)
    pl = Array{Any,1}(undef,npreconshifts)
    K = Array{Any,1}(undef,npreconshifts)

    if preconmethod == LUFac
        Threads.@threads for i=1:length(preconshifts)
            K[i] = add_shift(A,M,preconshifts[i])
            precons[i] = SaIPreconditioner(preconshifts[i], LUFac, lu(K[i]))
        end
        return precons
    end
    
    if preconmethod == GMRES
        Threads.@threads for i=1:length(preconshifts)
            K[i] = add_shift(A,M,preconshifts[i])

            pl[i] = Identity()
            if preconpreconsupplied 
                pl[i] = preconpreconmethod(K[i]; kwargs...)
            end
            x[i] = zeros(T,m)
            b[i] = ones(T,m)
            it = IterativeSolvers.gmres_iterable!(x[i], K[i], b[i], Pl = pl[i], maxiter=preconmaxiter, restart=preconrestart)
            it.reltol = preconreltol
            precons[i] = SaIPreconditioner(preconshifts[i], GMRES, it, tol = preconreltol)
        end

        return precons
    end

    if preconmethod == CG
        Threads.@threads for i=1:length(preconshifts)
            K[i] = add_shift(A,M,preconshifts[i])
            pl[i] = Identity()            
            if preconpreconsupplied 
                pl[i] = preconpreconmethod(K[i]; kwargs...)
            end
            x[i] = zeros(T,m)
            b[i] = ones(T,m)
            it = IterativeSolvers.cg_iterator!(x[i], K[i], b[i], pl[i], maxiter=preconmaxiter)
            it.reltol = preconreltol
            precons[i] = SaIPreconditioner(preconshifts[i], CG, it, tol = preconreltol)
        end

        return precons
    end

    if preconmethod == GaussSeidel
        Threads.@threads for i=1:length(preconshifts)
            K[i] = add_shift(A,M,preconshifts[i])
            x[i] = zeros(T,m)
            b[i] = ones(T,m)
            it = IterativeSolvers.DenseGaussSeidelIterable(K[i], x[i], b[i], preconmaxiter)
            precons[i] = SaIPreconditioner(preconshifts[i], GaussSeidel, it, tol = preconreltol)
        end

        return precons
    end

    if preconmethod == BiCGStab
        Threads.@threads for i=1:length(preconshifts)
            # note: maxiter here takes the role of maximum matrix-vector products performed 
            # in the bicgstab algorithm as specified in IterativeSolvers
            # default: l=1 for standard BiCGStab
            K[i] = add_shift(A,M,preconshifts[i])
            pl[i] = Identity()
            if preconpreconsupplied 
                pl[i] = preconpreconmethod(K[i]; kwargs...)
            end
            x[i] = zeros(T,m)
            b[i] = ones(T,m)
            it = IterativeSolvers.bicgstabl_iterator!(x[i], K[i], b[i], 1, Pl = pl[i], max_mv_products=preconmaxiter, tol=preconreltol)
            precons[i] = SaIPreconditioner(preconshifts[i], BiCGStab, it, tol = preconreltol)
        end

        return precons
    end
end

function ldiv!(y, pc::SaIPreconditioner, v; l::ReentrantLock=ReentrantLock())
    if pc.method == LUFac 
        # simply call the ldiv method for LU factorizations
        ldiv!(y, pc.methoddata, v)
    end

    if pc.method == GMRES 
        # initialize with y
        copyto!(pc.methoddata.x, y)
        # copy the right-hand side into the iterator
        copyto!(pc.methoddata.b, v)
        # initiate first search direction and residual data
        pc.methoddata.residual.current = IterativeSolvers.init!(pc.methoddata.arnoldi, pc.methoddata.x, pc.methoddata.b, pc.methoddata.Pl, pc.methoddata.Ax)
        # the least squares solve is called with iterable.β instead of residual.β
        # so we need to explicitly reset both β's
        pc.methoddata.β = pc.methoddata.residual.current 
        IterativeSolvers.init_residual!(pc.methoddata.residual, pc.methoddata.residual.current)
        pc.methoddata.reltol = pc.tol * pc.methoddata.residual.current
        # perform the iteration
        for (it,res) in enumerate(pc.methoddata)

        end
        # copy the solution to y
        copyto!(y, pc.methoddata.x)
    end

    if pc.method == CG
        T = eltype(pc.methoddata.x)
        copyto!(pc.methoddata.x, y)

        pc.methoddata.mv_products = 0

        pc.methoddata.u .= zero(T)
        copyto!(pc.methoddata.r, v)

        # compute residual
        pc.methoddata.mv_products += 1
        mul!(pc.methoddata.c, pc.methoddata.A, pc.methoddata.x)
        pc.methoddata.r .-= pc.methoddata.c
        pc.methoddata.residual = norm(pc.methoddata.r)
        pc.methoddata.reltol = pc.tol * pc.methoddata.residual

        for (it,res) in enumerate(pc.methoddata)
            
        end
        copyto!(y, pc.methoddata.x)
    end

    if pc.method == GaussSeidel
        copyto!(pc.methoddata.x, y)
        copyto!(pc.methoddata.b, v)
        
        for (i, nothing) in enumerate(pc.methoddata)
            if norm(pc.methoddata.b - pc.methoddata.A * pc.methoddata.x) < pc.tol * norm(pc.methoddata.b)
                break
            end
        end

        copyto!(y, pc.methoddata.x)
    end

    if pc.method == BiCGStab        
        T = eltype(pc.methoddata.x)
        n = size(pc.methoddata.A, 1)
        l = pc.methoddata.l

        copyto!(pc.methoddata.x, y)

        pc.methoddata.mv_products = 0
        pc.methoddata.r_shadow = rand(T, n)
        pc.methoddata.rs = Matrix{T}(undef, n, l + 1)
        pc.methoddata.us = zeros(T, n, l+1)

        #pc.methoddata.rs[:,1] .= v .- (pc.methoddata.A * pc.methoddata.x)
        temp = view(pc.methoddata.rs, : , 1)
        mul!(temp, pc.methoddata.A, pc.methoddata.x)
        temp .= v .- temp
        pc.methoddata.mv_products += 1
        
        ldiv!(pc.methoddata.Pl, temp)

        pc.methoddata.residual = norm(temp)

        pc.methoddata.γ = zeros(T, l)
        pc.methoddata.ω = one(T)
        pc.methoddata.σ = one(T)
        pc.methoddata.M = zeros(T, l+1, l+1)

        pc.methoddata.reltol = pc.tol * pc.methoddata.residual
        
        for (it, nothing) in enumerate(pc.methoddata)

        end
        
        copyto!(y, pc.methoddata.x)
    end
end
