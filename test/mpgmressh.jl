using SparseArrays
using LinearAlgebra
using Printf
using LinearMaps
using Random

using Test
using AlgebraicMultigrid: poisson

#include("../src/MPGMRESSh.jl")
#include("./test_utilities.jl")

#*******************************************************
#********************* Unit Tests **********************
#*******************************************************

@testset "MPGMRESSh" begin

Random.seed!(1234321)
n = 100

# test MPGMRESSh with 1 and 3 preconditioners
# on dense matrices, sparse matrices and LinearMaps
# for all preconditioning solve methods

println("Running MPGMRESSh test for dense matrices...")
@testset "Matrix{$T}" for T in (Float32, Float64)
    @testset "nprecons = $nprecons" for nprecons in 1:2:3
        @testset "preconmethod = $preconmethod" for preconmethod in (MPGMRESSh.LUFac,MPGMRESSh.CG,MPGMRESSh.GMRES,MPGMRESSh.GaussSeidel,MPGMRESSh.BiCGStab)
            # use Poisson test matrix for tests since it is SPD and diagonally dominant
            A = Matrix(poisson(n))
            M = diagm(ones(n))
            shifts = vcat(1e-3 .* collect(1:40), 1.4 .+ 1e-3 .* collect(1:40))
            b = ones(n)

            maxiter = n
            preconreltol = 1.0e-17

            if preconmethod == MPGMRESSh.GaussSeidel
                # make the diagonal dominance of the systems 
                # more pronounced for Gauß Seidel to decrease
                # the number of iterations needed for the solves
                @inbounds for i in 1:size(A,2)
                    A[i,i] += 1.0
                end
            end

            x,it,his = MPGMRESSh.mpgmressh(b, A, M, shifts, nprecons=nprecons, 
            maxiter=maxiter, log = true, verbose = false,
            btol=1.0e-10, convergence=MPGMRESSh.standard, preconmethod=preconmethod,
            preconreltol=preconreltol,
            preconmaxiter=3*size(A,2),
            preconrestart=size(A,2));
            
            @test maximum(test_residual(it)) / norm(b) < it.btol;
        end;
    end;
end;

println("Running MPGMRESSh test for sparse matrices...")
@testset "Sparse{$T}" for T in (Float32, Float64)
    @testset "preconmethod = $preconmethod" for preconmethod in (MPGMRESSh.LUFac,MPGMRESSh.CG,MPGMRESSh.GMRES,MPGMRESSh.GaussSeidel,MPGMRESSh.BiCGStab)
        nprecons = 3
        # use Poisson test matrix for tests since it is SPD and diagonally dominant
        A = poisson(n)
        M = sparse(diagm(ones(n)))
        shifts = vcat(1e-3 .* collect(1:40), 1.4 .+ 1e-3 .* collect(1:40))
        b = ones(n)

        maxiter = n
        preconreltol = 1.0e-17

        if preconmethod == MPGMRESSh.GaussSeidel
            # make the diagonal dominance of the systems 
            # more pronounced for Gauß Seidel to decrease
            # the number of iterations needed for the solves
            @inbounds for i in 1:size(A,2)
                A[i,i] += 1.0
            end
        end
        
        x,it,his = MPGMRESSh.mpgmressh(b, A, M, shifts, nprecons=nprecons, 
        maxiter=maxiter, log = true, verbose = false,
        btol=1.0e-10, convergence=MPGMRESSh.standard, preconmethod=preconmethod,
        preconreltol=preconreltol,
        preconmaxiter=3*size(A,2),
        preconrestart=size(A,2));
        
        @test maximum(test_residual(it)) / norm(b) < it.btol;
    end;
end;

println("Running MPGMRESSh test for LinearMaps...")
@testset "LinearMap{$T}" for T in (Float32, Float64)
    @testset "preconmethod = $preconmethod" for preconmethod in (MPGMRESSh.CG,MPGMRESSh.GMRES,MPGMRESSh.BiCGStab)
        nprecons = 3
        # use the same LinearMap example as IterativeSolvers.gmres
        # and the poisson example for the CG case
        if preconmethod == MPGMRESSh.CG
            A = LinearMap(poisson(n))
        else
            A = LinearMap(cumsum!, n; ismutating = true)
        end
        M = LinearMap(diagm(ones(n)))
        
        shifts = vcat(1e-3 .* collect(1:40), 1.4 .+ 1e-3 .* collect(1:40))
        b = ones(n)
        
        maxiter = n
        preconreltol = 1.0e-17
        
        x,it,his = MPGMRESSh.mpgmressh(b, A, M, shifts, nprecons=nprecons, 
        maxiter=maxiter, log = true, verbose = false,
        btol=1.0e-10, convergence=MPGMRESSh.standard, preconmethod=preconmethod,
        preconreltol=preconreltol,
        preconmaxiter=3*size(A,2),
        preconrestart=size(A,2));
        
        @test maximum(test_residual(it)) / norm(b) < it.btol;
    end;    
end;

# test MPGMRESSh with AMG preconditioner for the 
# CG and BiCGStab preconditioner solves
println("Running MPGMRESSh AMG test for sparse matrices...")
@testset "AMG Test for Sparse{$T}" for T in (Float32, Float64)
    @testset "preconmethod = $preconmethod" for preconmethod in (MPGMRESSh.CG,MPGMRESSh.BiCGStab)
        nprecons = 3
        # use Poisson test matrix for tests since it is SPD and diagonally dominant
        A = poisson(n)
        M = sparse(diagm(ones(n)))
        shifts = vcat(1e-3 .* collect(1:40), 1.4 .+ 1e-3 .* collect(1:40))
        b = ones(n)
        
        maxiter = n
        preconreltol = 1.0e-17
        
        x,it,his = MPGMRESSh.mpgmressh(b, A, M, shifts, nprecons=nprecons, 
        maxiter=maxiter, log = true, verbose = false,
        btol=1.0e-10, convergence=MPGMRESSh.standard, preconmethod=preconmethod,
        preconreltol=preconreltol,
        preconAMG = true,
        preconmaxiter=3*size(A,2),
        preconrestart=size(A,2));
        
        @test maximum(test_residual(it)) / norm(b) < it.btol;
    end;
end;

# test if the Arnoldi decomposition holds
println("Running MPGMRESSh Arnoldi decomposition test for sparse matrices...")
@testset "Arnoldi Test for Sparse{$T}" for T in (Float32, Float64)
    nprecons = 3
    preconmethod = MPGMRESSh.LUFac

    # use Poisson test matrix for tests since it is SPD and diagonally dominant
    A = poisson(n)
    M = sparse(diagm(ones(n)))
    shifts = vcat(1e-3 .* collect(1:40), 1.4 .+ 1e-3 .* collect(1:40))
    b = ones(n)
    
    maxiter = n
    preconreltol = 1.0e-17
    
    x,it,his = MPGMRESSh.mpgmressh(b, A, M, shifts, nprecons=nprecons, 
    maxiter=maxiter, log = true, verbose = false,
    btol=1.0e-10, convergence=MPGMRESSh.standard, preconmethod=preconmethod,
    preconreltol=preconreltol,
    preconAMG = true,
    preconmaxiter=3*size(A,2),
    preconrestart=size(A,2));

    # expected max error: ~5e-14
    @test maximum(test_Arnoldi_relationship(it)) < 1.0e-13
end;
end;