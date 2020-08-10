import LinearAlgebra.ldiv!
using IterativeSolvers, LinearAlgebra, SparseArrays
# name clash with GaussSeidel in PreconMethod: use needed methods directly
using AlgebraicMultigrid: ruge_stuben, aspreconditioner
using ExtendableSparse: JacobiPreconditioner
using IncompleteLU: ilu

export PreconMethod, generate_preconditioners

@enum PreconMethod begin
    LUFac = 1
    CG = 2
    GMRES = 3
    GaussSeidel = 4
    BiCGStab = 5
    custom = 6
end

# generic shift and invert preconditioner data structure 
# to provide standard preconditioner solution methods
mutable struct SaIPreconditioner{shiftT, methoddataT, resT}
    # shift associated with preconditioner
    shift::shiftT
    
    # preconditioner solution method to be applied
    method::PreconMethod
    # solution method metadata prepared in advance such as
    # LU factorizations or GMRES iterables
    methoddata::methoddataT
    tol::resT
end

function SaIPreconditioner(shift::shiftT, method::PreconMethod, methoddata::methoddataT; 
    tol = sqrt(eps(real(eltype(shift))))) where {shiftT, methoddataT}
    return SaIPreconditioner(shift, method, methoddata, tol)
end

# generate preconditioners given A,M and preconditioning shifts to provide 
# preconitioning solves of the desired type
function generate_preconditioners(A, M, preconshifts::Array{shiftT, 1}, method::PreconMethod; 
    maxiter = size(A,2),
    restart = min(20, size(A,2)),
    AMG = false, # flag to provide AMG precon for Krylov methods
    jacobi=false, # flag to provide jacobi precon for Krylov methods
    iluprec=false, # flag to provide ILU precon for Krylov methods 
    τ=0.1, # drop threshold parameter for IncompleteLU.ilu call
    reltol = sqrt(eps(real(eltype(A))))) where shiftT
    
    npreconshifts = length(preconshifts)
    precons = Array{SaIPreconditioner}(undef, npreconshifts)

    if method == LUFac
        Threads.@threads for i=1:length(preconshifts)
            precons[i] = SaIPreconditioner(preconshifts[i], LUFac, lu(A + preconshifts[i] * M))
            #println("\t set up precon $i on thread $(Threads.threadid())")
        end
        return precons
    end
    
    if method == GMRES
        # assume preconditioners map from the eltype of (A+shift.*M) to the same type
        #T = typeof(one(eltype(A)) + one(eltype(preconshifts)) * one(eltype(M)))
        T = eltype(preconshifts[1] * M)
        m = size(M, 2)
        x = zeros(T, m)
        b = ones(T, m)
        pl = Identity()
        Threads.@threads for i=1:length(preconshifts)
            K = A + preconshifts[i]*M
            if AMG
                K = Symmetric(sparse(K))
                ml = ruge_stuben(K)
                pl = aspreconditioner(ml)
            end
            if jacobi
                pl = JacobiPreconditioner(K)
            end
            if iluprec
                pl = ilu(K,τ=τ)
            end
            it = IterativeSolvers.gmres_iterable!(x, K, b, Pl = pl, maxiter=maxiter, restart=restart)
            it.reltol = reltol
            precons[i] = SaIPreconditioner(preconshifts[i], GMRES, it, tol = reltol)
        end

        return precons
    end

    if method == CG
        T = eltype(preconshifts[1] * M)
        m = size(M, 2)
        x = zeros(T, m)
        b = ones(T, m)
        pl = Identity()
        Threads.@threads for i=1:length(preconshifts)
            K = A + preconshifts[i] * M
            if AMG
                K = Symmetric(sparse(K))
                ml = ruge_stuben(K)
                pl = aspreconditioner(ml)
            end
            if jacobi
                pl = JacobiPreconditioner(K)
            end
            if iluprec
                pl = ilu(K,τ=τ)
            end
            it = IterativeSolvers.cg_iterator!(x, K, b, pl, maxiter=maxiter)
            it.reltol = reltol
            precons[i] = SaIPreconditioner(preconshifts[i], CG, it, tol = reltol)
        end

        return precons
    end

    if method == GaussSeidel
        T = eltype(preconshifts[1] * M)
        m = size(M, 2)
        x = zeros(T, m)
        b = ones(T, m)
        Threads.@threads for i=1:length(preconshifts)
            it = IterativeSolvers.DenseGaussSeidelIterable(A + preconshifts[i] * M, x, b, maxiter)
            precons[i] = SaIPreconditioner(preconshifts[i], GaussSeidel, it, tol = reltol)
        end

        return precons
    end

    if method == BiCGStab
        T = eltype(preconshifts[1] * M)
        m = size(M, 2)
        x = zeros(T, m)
        b = ones(T, m)
        pl = Identity()
        Threads.@threads for i=1:length(preconshifts)
            # note: maxiter here takes the role of maximum matrix-vector products performed 
            # in the bicgstab algorithm as specified in IterativeSolvers
            # default: l=1 for standard BiCGStab
            K = A + preconshifts[i] * M
            if AMG 
                K = Symmetric(sparse(K))
                ml = ruge_stuben(K)
                pl = aspreconditioner(ml)
            end
            if jacobi
                pl = JacobiPreconditioner(K)
            end
            if iluprec
                pl = ilu(K,τ=τ)
            end
            it = IterativeSolvers.bicgstabl_iterator!(x, K, b, 1, Pl = pl, max_mv_products=maxiter, tol=reltol)
            precons[i] = SaIPreconditioner(preconshifts[i], BiCGStab, it, tol = reltol)
        end

        return precons
    end

    if method == custom 
        for (i,v) in enumerate(preconshifts)
            precons[i] = SaIPreconditioner(preconshifts[i], custom, nothing)
        end
        return precons
    end
end

# !TODO: make ldiv! thread safe
function ldiv!(y, pc::SaIPreconditioner, v)
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
        #println("\t precon solve: ")
        #j = 1
        for (it,res) in enumerate(pc.methoddata)
        #    println("\t it, res: ", it, ", ", res)
        #    j = j+1
        end
        #println("\t took ",j," iterations")
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

        #j = 1
        #println("\t\t thread: $(Threads.threadid())")
        #println("\t precon solve:")
        for (it,res) in enumerate(pc.methoddata)
            #println("\t\t thread: $(Threads.threadid())")
            #println("\t it, res: ", it, ", ", res)
            #j = j+1
        end
        #println("\t\t thread: $(Threads.threadid())")
        #println("\t took ", j, " iterations")
        copyto!(y, pc.methoddata.x)

        return 0
    end

    if pc.method == GaussSeidel
        copyto!(pc.methoddata.x, y)
        copyto!(pc.methoddata.b, v)
        
        #println("\t precon solve:")
        #j = 1
        for (i, nothing) in enumerate(pc.methoddata)
            if norm(pc.methoddata.b - pc.methoddata.A * pc.methoddata.x) < pc.tol * norm(pc.methoddata.b)
                break
            end
            #println("\t res: ", norm(pc.methoddata.b - pc.methoddata.A * pc.methoddata.x))
            #j = j+1
        end
        #println("\t took ", j, " iterations")

        copyto!(y, pc.methoddata.x)
    end

    if pc.method == BiCGStab
        # the setup of all internal structures is done in 
        # the iterator constructor
        #pc.methoddata = IterativeSolvers.bicgstabl_iterator!(y, pc.methoddata.A, v, 2,
        #max_mv_products = pc.methoddata.max_mv_products, tol = pc.methoddata.reltol)
        
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
        
        #println(" \t precon solve:")
        #j = 1
        for (it, nothing) in enumerate(pc.methoddata)
            #j = j+1
            #println("\t res: ", pc.methoddata.residual)
        end

        #println(" \t took ", j, " iterations")
        
        copyto!(y, pc.methoddata.x)
    end

    if pc.method == custom
        error("custom preconditioning solve requires custom ldiv! method!")
    end
end