using BlockDiagonals

function fgmressh(b, A, M, shifts; kwargs...)
    #T = typeof(one(eltype(b))/one(eltype(M)))
    T = typeof(one(eltype(b))/(one(eltype(shifts)) * one(eltype(M))))
  
    x = [similar(b, T)]
    fill!(x[1], zero(T))
    for k=2:length(shifts)
        push!(x, similar(b,T))
        fill!(x[k], zero(T))
    end

    return fgmressh!(x, b, A, M, shifts; kwargs...)
end

function fgmressh!(x, b, A, M, shifts;
    preconvals = 3, # set of preconvalues 
    cycle_length = 8*preconvals, # preconditioners are swapped every cycle_length/preconvals iterations
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
    preconAMG = false, # flags for iterative preconditioner
    preconjacobi = false, # solve preconditioners
    preconilu = false,
    τ = 0.1,
    preconreltol = btol
)
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
        IterativeSolvers.reserve!(history, :it_times, maxiter)
        
        setup_time = time_ns()
    end
    # construct preconditioners
    preconshifts = generate_preconshifts(shifts, preconvals)
    precons = generate_preconditioners(A, M, preconshifts, preconmethod, maxiter = preconmaxiter, 
    restart = preconrestart, AMG = preconAMG, jacobi = preconjacobi, iluprec = preconilu, τ = τ, reltol = preconreltol)

    # build an mpgmressh iterable with nprecons=1 search directions added 
    # per Arnoldi expansion and a set of preconvals shift-and-invert preconditioners
    global iterable = mpgmressh_iterable!(x, b, A, M, shifts,
    preconshifts, precons, 1;
    btol = btol, atol = atol, maxiter = maxiter,
    convergence = convergence, explicit_residual = explicit_residual)

    mk = floor(Int64, cycle_length/preconvals)
    precon_index = 1

    iterable.barnoldi.currentprecons[1] = precon_index
    BlockDiagonals.getblock(iterable.barnoldi.T, iterable.k)[1] = iterable.barnoldi.preconshifts[precon_index]

    if log 
        setup_time = time_ns() - setup_time
        history[:setup_time] = Int(setup_time)/(1e9)

        # set up convergence flag array
        IterativeSolvers.reserve!(Int64, history, :conv_flags, 1, length(shifts))

        old_flags = falses(length(shifts))
        new_flags = falses(length(shifts))

        old_flags .= iterable.residual.flag 
        
        iteration_time = time_ns()
    end

    lastittime = 0
    ittime = time_ns()

    for (it, res) in enumerate(iterable)
        # if one preconditioning cycle is completed, swap the precon index
        #@printf("precon index: %d\n",precon_index)
        lastittime = 0
        if mod(iterable.k, mk) == 0
            if mod(iterable.k, cycle_length) == 0 
                precon_index = 0
            end
            precon_index +=1
            iterable.barnoldi.currentprecons[1] = precon_index
            #println("\t Swapped precon...")
        end
        # set the diagonal element of the auxiliary matrix T to the currently 
        # applied preconditioning shift value
        BlockDiagonals.getblock(iterable.barnoldi.T, iterable.k)[1] = iterable.barnoldi.preconshifts[precon_index]

        if log
            new_flags .= iterable.residual.flag
            IterativeSolvers.nextiter!(history)
            ittime = time_ns() - ittime
            history[:it_times][history.iters] = Int(ittime)/(1e9)
            history[:resmat][history.iters,:] .= iterable.residual.current
            history[:conv_flags][findall(x->x==true, xor.(old_flags,new_flags))] .= history.iters
            old_flags .= new_flags
        end

        verbose && @printf("%3d\t%1.2e\n", 1 + mod(it - 1, maxiter), maximum(res))
        lastittime = time_ns()

        ittime = time_ns()
    end
    lastittime = time_ns() - lastittime

    if log 
        # the last step of the for loop calls iterate() one more time,
        # so the last convergence history step needs to be performed here
        new_flags .= iterable.residual.flag
        IterativeSolvers.nextiter!(history)
        history[:resmat][history.iters,:] .= iterable.residual.current
        history[:conv_flags][findall(x->x==true, xor.(old_flags,new_flags))] .= history.iters
        old_flags .= new_flags

        history[:it_times][history.iters] = Int(lastittime)/(1e9)

        iteration_time = time_ns() - iteration_time
        history[:iteration_time] = Int(iteration_time)/(1e9)
        history.mvps = iterable.mv_products

        history[:resmat] = history[:resmat][1:history.iters,:]
        history[:it_times] = history[:it_times][1:history.iters]
        IterativeSolvers.setconv(history, converged(iterable))
    end

    log ? (x, iterable, history) : x
end
