using LinearAlgebra

function test_Arnoldi_relationship(it_mpgmressh)
    nprecons = length(it_mpgmressh.barnoldi.precons)
    A = it_mpgmressh.barnoldi.A 
    M = it_mpgmressh.barnoldi.M 
    res = []
    for (i, σ) in enumerate(it_mpgmressh.barnoldi.allshifts)
        V = hcat(it_mpgmressh.barnoldi.V...)
        m, n = size(V)
        H = it_mpgmressh.barnoldi.H[1:n, 1:n-1]
        E = it_mpgmressh.barnoldi.E[1:n, 1:n-1]
        T = it_mpgmressh.barnoldi.T[1:n-1, 1:n-1]
        Hm = H * (UniformScaling(σ) - T)
        Z = it_mpgmressh.barnoldi.Z[:, 1:n-1]
        rhs = V * (E + Hm)
        #lhs = (A + σ * M) * Z
        lhs = hcat([(A + σ * M) * col for col in eachcol(Z)]...)
        #@printf("σ = %.2e + 1im * %2.e: (lhs ≈ rhs = %d) at res = %.10e \n", real(σ), imag(σ), lhs ≈ rhs, maximum(abs.(rhs-lhs)))
        push!(res, maximum(abs.(rhs-lhs)))
    end
    return res
end

function test_residual(it_mpgmressh)
    return [norm(it_mpgmressh.b - (it_mpgmressh.barnoldi.A + shift * it_mpgmressh.barnoldi.M) * it_mpgmressh.x[i]) for (i, shift) in enumerate(it_mpgmressh.barnoldi.allshifts)]
end