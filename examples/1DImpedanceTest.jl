# # 150: Impedance calculation
#  taken from https://github.com/j-fu/VoronoiFVM.jl/blob/master/examples/Example150_Impedance1D.jl
#  and modified to showcase MPGMRESSh.jl application
#
#  Impedance calculation for 
#
#    C u_t - (D u_x)_x + Ru = 0   in (0,1)
#      u(0,t)=1 + exp(iωt)
#      u(1,t)=0
#
#    Measurement: I(t)= D u_x(1,t)
#     
#    Steady state: 
#    - (D u0_x)_x + Ru0 = 0
#    u0(0,t)=1 
#    u0(1,t)=0 
#
#    Small signal ansatz for ω
#
#    u(x,t)= u0(x)+ ua(x) exp(iωt)
#
#    iωC ua - (D ua_x)_x + R u_a =0
#      ua(0)=1
#      ua(1)=0
# 
#        

module Example150_Impedance1D

using Printf
using VoronoiFVM
using LaTeXStrings
using SparseArrays
using LinearAlgebra

include("../src/MPGMRESSh.jl")

# Structure containing  userdata information
mutable struct Data  <: VoronoiFVM.AbstractData
    D::Float64           
    C::Float64
    R::Float64
    Data()=new()
end


function main(;nref=0, preconmethod = MPGMRESSh.LUFac, nprecons = 3, maxiter = 20, preconreltol = 1.0e-10, regular = false, Plotter=nothing, verbose=false, dense=false, animate=false,
    R=1,
    D=1,
    C=2)

    L=1.0

    if !regular
        # Create array which is refined close to 0
        h0=0.1/2.0^nref
        h1=0.5/2.0^nref
        X=VoronoiFVM.geomspace(0.0,L,h0,h1)
    else
        X=collect(range(0.,stop=L,length=nref))
    end

    # Create and plot discretitzation grid
    grid=VoronoiFVM.Grid(X)
    p = VoronoiFVM.ExtendableGrids.plot(grid, Plotter=Plotter)
    display(p)

    # Create and fill data 
    data=Data()
    data.R=R
    data.D=D
    data.C=C

    # Declare constitutive functions
    flux=function(f,u,edge,data)
        f[1]=data.D*(u[1]-u[2])
    end

    storage=function(f,u,node,data)
        f[1]=data.C*u[1]
    end

    reaction=function(f,u,node,data)
        f[1]=data.R*u[1]
    end

    # Create physics struct
    physics=VoronoiFVM.Physics(data=data,
                               flux=flux,
                               storage=storage,
                               reaction=reaction
                               )
    # Create discrete system and enabe species
    if dense
        sys=VoronoiFVM.DenseSystem(grid,physics)
    else
        sys=VoronoiFVM.SparseSystem(grid,physics)
    end
    enable_species!(sys,1,[1])

    # Create test functions for current measurement

    excited_bc=1
    excited_bcval=1.0
    excited_spec=1


    factory=VoronoiFVM.TestFunctionFactory(sys)
    measurement_testfunction=testfunction(factory,[1],[2])

    # define a set voltage of 1 on bc region 1 and 
    # 0 on region 2 (penalty-method-enforced Dirichlet)
    boundary_dirichlet!(sys,excited_spec,excited_bc,excited_bcval)
    boundary_dirichlet!(sys,1,2,0.0)
    
    # solve the steady-state system 
    inival=unknowns(sys)
    steadystate=unknowns(sys)
    inival.=0.0
    solve!(steadystate,inival,sys)

    if Plotter!=nothing
        p = VoronoiFVM.ExtendableGrids.plot(grid,steadystate[1,:], Plotter=Plotter)
        display(p)
    end

    # define the measure functionals for measuring the terminal contact current
    function meas_stdy(meas,U)
        u=reshape(U,sys)
        meas[1]=VoronoiFVM.integrate_stdy(sys,measurement_testfunction,u)[1]
        nothing
    end

    function meas_tran(meas,U)
        u=reshape(U,sys)
        meas[1]=VoronoiFVM.integrate_tran(sys,measurement_testfunction,u)[1]
        nothing
    end

    # define the derivatives of the measure functionals
    dmeas_stdy=measurement_derivative(sys,meas_stdy,steadystate)
    dmeas_tran=measurement_derivative(sys,meas_tran,steadystate)

    # Create Impeadance system from steady state
    excited_spec=1
    excited_bc=1
    isys=VoronoiFVM.ImpedanceSystem(sys,steadystate,excited_spec, excited_bc)

    # array to contain the excitation frequencies
    allomega=zeros(0)

    # for calculated data
    allI0=zeros(Complex{Float64},0)
    allIL=zeros(Complex{Float64},0)

    # for exact data
    allIx0=zeros(Complex{Float64},0)
    allIxL=zeros(Complex{Float64},0)

    ω0=0.5
    ω1=1.0e4
    ω=ω0

    testval=0.0
    UZ=unknowns(isys)
    noωs = Int64(ceil((log(2) + 4* log(10))/(log(6) - log(5))))
    allUZ = zeros(ComplexF64, (size(UZ)[2], noωs))

    if isplots(Plotter) && animate
        anim = Plotter.Animation()
    end

    # ----------------------------------------------------------------

    # solve the impedance problem directly for each shift by direct 
    # LU factorization of the system matrix (in freqdomain_impedance)
    i = 0
    while ω<ω1        
        i = i + 1
        iω=1im*ω

        # solve impedance system for excited solution u_a
        # to put it onto allUZ
        solve!(UZ,isys,ω)
        allUZ[:, i] = UZ[1,:]

        # add real part of computed solution to animation frame 
        if isplots(Plotter) && animate
            p1 = VoronoiFVM.ExtendableGrids.plot(Plotter, grid, real(UZ[1,:]), show=false, color=(1,0,0), label="Re(u_a)")
            p2 = VoronoiFVM.ExtendableGrids.plot(Plotter, grid, imag(UZ[1,:]), show=false, color=(0,0,1), label="Im(u_a)")
            Plotter.plot(p1,p2,layout = (2,1))
            Plotter.frame(anim)
        end

        # calculate aproximate solution
        # obtain measurement in frequency domain
        IL=freqdomain_impedance(isys,ω,steadystate,excited_spec,excited_bc,excited_bcval,dmeas_stdy, dmeas_tran)

        # record approximate solution
        push!(allomega, ω)
        push!(allIL,IL)

        # record exact solution
        z=sqrt(iω*data.C/data.D+data.R/data.D);
        eplus=exp(z*L);
        eminus=exp(-z*L);
        IxL=2.0*data.D*z/(eminus-eplus);
        push!(allIxL,IxL)

        # increase omega
        ω=ω*1.2
    end

    # ----------------------------------------------------------------

    # solve all impedance systems for all shifts simultaneously 
    # using the MPGMRESSh method and calculate the respective currents    
    # the shifted systems are defined by (sys.matrix + iω * isys.storderiv) * UZ_ω = isys.F

    # call the mpgmressh routine to obtain an array of solution vectors UZω 
    # each index corresponding to the respective shift iω
    iωs = 1.0im .* allomega

    @printf("MPGMRESSh solution without preconditioning: \n")
    # -rhs F needs reshaping into a column vector
    # -we need to make convergence decisions based on the absolute residual 
    #  due to the penalty factor on the rhs which fools the method into 
    #  instantaneous convergence after one iteration if relative residuals are used
    UZω, it_mpgmressh = MPGMRESSh.mpgmressh(reshape(isys.F, (size(isys.F)[2],)), sys.matrix, isys.storderiv, iωs, nprecons = nprecons, maxiter = maxiter,
                                log = true, verbose = true, btol=1.0e-10,
                                convergence = MPGMRESSh.absolute, preconmethod = preconmethod,
                                preconreltol = preconreltol)

    # now calculate the associated measurements for the unpreconditioned system
    allILMPGMRES = zeros(Complex{Float64}, length(allIL))
    for (i, iω) in enumerate(iωs)
        allILMPGMRES[i] = (dmeas_stdy*values(UZω[i]))[1] + iω * (dmeas_tran*values(UZω[i]))[1]
    end

    # the large penalty factor, however, spoils the absolute residual due to numerical artifacts
    absres = [norm(it_mpgmressh.b - (it_mpgmressh.barnoldi.A + shift .* it_mpgmressh.barnoldi.M) * it_mpgmressh.x[i]) for (i, shift) in enumerate(it_mpgmressh.barnoldi.allshifts)]
    @printf("Maximal absolute residual across shifts: %1.5e\n", maximum(absres))
    @printf("Maximal relative residual across shifts: %1.5e\n\n", maximum(absres ./ it_mpgmressh.residual.β))

    # checking the errors in the actual solution vectors and 
    # associated measurements
    maxnormSols = maximum([maximum(abs.(allUZ[:,i] - it_mpgmressh.x[i])) for i=1:length(it_mpgmressh.barnoldi.allshifts)])
    @printf("Maximum linf norm of MPGMRESSh and direct solutions: %1.20e\n", maxnormSols)
    maxnormILs = maximum(abs.(allIL - allILMPGMRES))
    @printf("Maximum linf norm of resulting measurements: %1.20e\n\n", maxnormILs)

    # ----------------------------------------------------------------
    
    if isplots(Plotter)
        if animate
            Plotter.gif(anim, "/tmp/impedancetest.gif", fps=15)
        end

        p=Plotter.plot(grid=true)
        
        Plotter.plot!(p,real(allIxL),imag(allIxL),label=LaTeXString("exact (\$I_{\\alpha}^{a}\$)"),lw=2)
        Plotter.plot!(p,real(allILMPGMRES),imag(allILMPGMRES),label=LaTeXString("approximated (\$\\widetilde{I}_{\\alpha}^{a}\$)"),lw=0, marker=:rect)
        Plotter.xlabel!(LaTeXString("\$\\mathrm{Re}(\\widetilde{I}_{\\alpha}^{a})\$"))
        Plotter.ylabel!(LaTeXString("\$\\mathrm{Im}(\\widetilde{I}_{\\alpha}^{a})\$"))

        display(p)
    end

    return sys, isys, allIL, allIxL, allILMPGMRES, allUZ, UZω, it_mpgmressh;
end

function test()
    main(dense=true) ≈ 0.23106605162049176 &&  main(dense=false) ≈ 0.23106605162049176
end


end

