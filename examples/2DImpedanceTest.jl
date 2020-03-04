module Example161_Impedance2D

using Printf
using VoronoiFVM
using LaTeXStrings
using SparseArrays
using LinearAlgebra
#using Triangulate

include("../src/MPGMRESSh.jl")

# Structure containing  userdata information
mutable struct Data  <: VoronoiFVM.AbstractData
    D::Float64           
    C::Float64
    R::Float64
    Data()=new()
end

# ### Triangulation of a rectangular domain for use in impedance analysis 
# L: length of the rectangle 
# h: height of the rectangle 
# left contact: boundary region = 1 (Dirichlet value = 1.0)
# right contact: boundary region = 2 (Dirichlet value = 0.0)
# no-flow conditions at boundary (hom. Neumann)
function generate_refined_triangle_input(;L=1.0, h = 1.0, minangle = 20, maxarea = 0.01)
    #triin = Triangulate.TriangulateIO()
    triin = VoronoiFVM.Triangulate.TriangulateIO()
    triin.pointlist = Matrix{Cdouble}([0.0 0.0 ; L 0.0 ; L h ; 0.0 h]')
    triin.segmentlist = Matrix{Cint}([1 2; 2 3; 3 4; 4 1]')
    triin.segmentmarkerlist=Vector{Int32}([3, 2, 3, 1])
    # use one bulk region ( rg = 1 ) with a region point square in the middle of the cell
    reglist = zeros(Cdouble, 4, 1)
    reglist[:,] = [L/2; h/2; 1; maxarea]
    triin.regionlist=reglist

    function refine_unsuitable(x1, y1, x2, y2, x3, y3, area)
        # squared distance of barycentric centroid to leftmost segment 
        dist = ((x1 + x2 + x3)/3.0)^2
        # require refinement if the area does not scale
        # by the factor of 0.1 in relation to the distance 
        # to the leftmost segment
        (dist>1.0e-5 && area > 0.1*dist) || (area > maxarea)
    end

    VoronoiFVM.Triangulate.triunsuitable(refine_unsuitable)
    
    return "paAuq$(angle)Q", triin;
end


function main(;L=1.0, h=1.0, maxarea=0.01, Plotter=nothing, trisurf=false, dense=false)

    # Create 2D rectangular mesh refined close to the left boundary
    switches, triin = generate_refined_triangle_input(L=L, h=h, maxarea=maxarea)

    grid=VoronoiFVM.Grid(switches, triin)

    VoronoiFVM.plot(Plotter, grid)
    @printf("Press ENTER to continue...")
    readline();

    # Create and fill data 
    data=Data()
    data.R=0
    data.D=1
    data.C=2

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
    excited_bc=1 # excited contact
    excited_bcval=1.0 # applied voltage
    excited_spec=1

    #factory=VoronoiFVM.TestFunctionFactory(sys)
    #measurement_testfunction=testfunction(factory,[1],[2])

    # define a set voltage of 1 on bc region 1 and 
    # 0 on region 2 (penalty-method-enforced Dirichlet)
    boundary_dirichlet!(sys,excited_spec,excited_bc,excited_bcval)
    boundary_dirichlet!(sys,1,2,0.0)
    boundary_neumann!(sys,1,3,0.0) # homogeneous Neumann on top and bottom 
    
    # solve the steady-state system 
    inival=unknowns(sys)
    steadystate=unknowns(sys)
    inival.=0.0
    solve!(steadystate,inival,sys)

    if trisurf
        Plotter.plot_trisurf(tridata(grid)..., steadystate[1,:], cmap = "cool")
        # Spectral or cool
    else
        VoronoiFVM.plot(Plotter, grid, steadystate[1,:])
    end

    @printf("Press Enter to continue...")
    readline()

    """ 
    # Create Impedance system from steady state
    excited_spec=1
    excited_bc=1
    isys=VoronoiFVM.ImpedanceSystem(sys,steadystate,excited_spec, excited_bc)
    UZ=unknowns(isys)

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
            #p1 = VoronoiFVM.plot(Plotter, grid, real(UZ[1,:]), show=false, color=(1,0,0), label=LaTeXString("\$\\Re(u_a), \\; \\omega = "*string(ω))*"\$")
            #p2 = VoronoiFVM.plot(Plotter, grid, imag(UZ[1,:]), show=false, color=(0,0,1), label=LaTeXString("\$\\Im(u_a), \\; \\omega = "*string(ω))*"\$")
            p1 = VoronoiFVM.plot(Plotter, grid, real(UZ[1,:]), show=false, color=(1,0,0), label="Re(u_a)")
            p2 = VoronoiFVM.plot(Plotter, grid, imag(UZ[1,:]), show=false, color=(0,0,1), label="Im(u_a)")
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

    # jacobi preconditioning combats numerical inflation of errors
    # in the absolute residual
    K = sparse(sys.matrix)
    M = deepcopy(isys.storderiv)
    b = deepcopy(isys.F.node_dof)
    jac = diag(K)
    
    for (i, el) in enumerate(jac)
        K[i, :] .*= 1/el
        M[i, :] .*= 1/el
        b[i] *= 1/el
    end

    @printf("MPGMRESSh solution without preconditioning: \n")
    # -rhs F needs reshaping into a column vector
    # -we need to make convergence decisions based on the absolute residual 
    #  due to the penalty factor on the rhs which fools the method into 
    #  instantaneous convergence after one iteration if relative residuals are used
    #
    # however, the implemented Ayachour residual update helps cancel out this problem
    # since it scales with the penalty parameter in its numerator and denominator equally
    UZω, it_mpgmressh = MPGMRESSh.mpgmressh(reshape(isys.F, (size(isys.F)[2],)), sys.matrix, isys.storderiv, iωs, nprecons = 3, maxiter = 20,
                                log = true, verbose = true, btol=1.0e-10,
                                convergence = MPGMRESSh.absolute)

    # now calculate the associated measurements for the unpreconditioned system
    allILMPGMRES = zeros(Complex{Float64}, length(allIL))
    for (i, iω) in enumerate(iωs)
        allILMPGMRES[i] = (dmeas_stdy*values(UZω[i]))[1] + iω * (dmeas_tran*values(UZω[i]))[1]
    end

    # the large penalty factor, however, spoils the absolute residual due to numerical artifacts
    absres = [norm(it_mpgmressh.b - (it_mpgmressh.barnoldi.A + shift .* it_mpgmressh.barnoldi.M) * it_mpgmressh.x[i]) for (i, shift) in enumerate(it_mpgmressh.barnoldi.allshifts)]
    @printf("Maximal absolute residual across shifts: %1.5e\n", maximum(absres))
    @printf("Maximal relative residual across shifts: %1.5e\n\n", maximum(absres ./ it_mpgmressh.residual.β))

    maxnormSols = maximum([maximum(abs.(allUZ[:,i] - it_mpgmressh.x[i])) for i=1:length(it_mpgmressh.barnoldi.allshifts)])
    @printf("Maximum linf norm of MPGMRESSh and direct solutions: %1.20e\n", maxnormSols)
    maxnormILs = maximum(abs.(allIL - allILMPGMRES))
    @printf("Maximum linf norm of resulting measurements: %1.20e\n\n", maxnormILs)

    @printf("----------------------------------------------------------------\n\n")

    @printf("MPGMRESSh solution with preconditioning: \n")
    # the jacobi preconditioning allows us to iterate normally with respect to the 
    # relative residual (Convergence.standard)
    UZω, it_mpgmressh = MPGMRESSh.mpgmressh(reshape(b, (size(b)[2],)), K, M, iωs, nprecons = 3, maxiter = 20,
                                log = true, verbose = true, btol=1.0e-10,
                                convergence = MPGMRESSh.standard)

    # now calculate the associated measurements for the jacobi-preconditioned system
    allILMPGMRES = zeros(Complex{Float64}, length(allIL))
    for (i, iω) in enumerate(iωs)
        allILMPGMRES[i] = (dmeas_stdy*values(UZω[i]))[1] + iω * (dmeas_tran*values(UZω[i]))[1]
    end

    # the rescaling by the jacobi preconditioner has taken care of the excessive absolute residuals
    # in our case, they are of course now identical to the relative residuals
    absres = [norm(it_mpgmressh.b - (it_mpgmressh.barnoldi.A + shift .* it_mpgmressh.barnoldi.M) * it_mpgmressh.x[i]) for (i, shift) in enumerate(it_mpgmressh.barnoldi.allshifts)]
    @printf("Maximal absolute residual across shifts: %1.5e\n", maximum(absres))
    @printf("Maximal relative residual across shifts: %1.5e\n\n", maximum(absres ./ it_mpgmressh.residual.β))

    maxnormSols = maximum([maximum(abs.(allUZ[:,i] - it_mpgmressh.x[i])) for i=1:length(it_mpgmressh.barnoldi.allshifts)])
    @printf("Maximum linf norm of MPGMRESSh and direct solutions: %1.20e\n", maxnormSols)
    maxnormILs = maximum(abs.(allIL - allILMPGMRES))
    @printf("Maximum linf norm of resulting measurements: %1.20e\n", maxnormILs)

    # ----------------------------------------------------------------
    
    if isplots(Plotter)
        if animate
            Plotter.gif(anim, "/tmp/impedancetest.gif", fps=15)
        end

        p=Plotter.plot(grid=true)
        Plotter.plot!(p,real(allIL),imag(allIL),label="calc")
        Plotter.plot!(p,real(allIxL),imag(allIxL),label="exact")

        #Plotter.gui(p)
        display(p)
    end
    """

    #return sys, isys, allIL, allIxL, allILMPGMRES, allUZ, UZω, it_mpgmressh;
    return steadystate, sys;
end

#function test()
#    main(dense=true) ≈ 0.23106605162049176 &&  main(dense=false) ≈ 0.23106605162049176
#end


end
