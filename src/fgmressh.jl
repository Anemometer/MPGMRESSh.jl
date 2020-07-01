using BlockDiagonals

function fgmressh(b, A, M, shifts; kwargs...)
    T = typeof(one(eltype(b))/one(eltype(M)))
  
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
    preconreltol = btol
)
    # construct preconditioners
    preconshifts = generate_preconshifts(shifts, preconvals)
    precons = generate_preconditioners(A, M, preconshifts, preconmethod, maxiter = preconmaxiter, 
    restart = preconrestart, reltol = preconreltol)

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

    @time for (it, res) in enumerate(iterable)
        # if one preconditioning cycle is completed, swap the precon index
        #if mod(it, mk) == 0
        @printf("precon index: %d\n",precon_index)
        if mod(iterable.k, mk) == 0
            #if mod(it, cycle_length) == 0 
            if mod(iterable.k, cycle_length) == 0 
                precon_index = 0
            end
            precon_index +=1
            iterable.barnoldi.currentprecons[1] = precon_index            
        end        
        # set the diagonal element of the auxiliary matrix T to the currently 
        # applied preconditioning shift value
        #iterable.barnoldi.T[it][1] = iterable.barnoldi.preconshifts[precon_index]
        #BlockDiagonals.getblock(iterable.barnoldi.T, it)[1] = iterable.barnoldi.preconshifts[precon_index]
        BlockDiagonals.getblock(iterable.barnoldi.T, iterable.k)[1] = iterable.barnoldi.preconshifts[precon_index]
        verbose && @printf("%3d\t%1.2e\n", 1 + mod(it - 1, maxiter), maximum(res))
    end

    log ? (x, iterable) : x
end
