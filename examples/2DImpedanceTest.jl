module Example161_Impedance2D

using Printf
using VoronoiFVM
using LaTeXStrings
using SparseArrays
using LinearAlgebra
using Triangulate

include("../src/MPGMRESSh.jl")
include("../src/preconditioner.jl")

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
    #triin = VoronoiFVM.Triangulate.TriangulateIO()
    triin = Triangulate.TriangulateIO()
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

    #VoronoiFVM.Triangulate.triunsuitable(refine_unsuitable)
    Triangulate.triunsuitable(refine_unsuitable)
    
    return "paAuq$(angle)Q", triin;
end


function main(;regular = false, L=1.0, h=1.0, maxarea=0.01, dense=false, nprecons=3, preconmethod = MPGMRESSh.LUFac, maxiter=20, preconreltol = 1.0e-10, Plotter=nothing, trisurf=false, animate = false)

    if regular 
        # create regular grid with squares as cells
        X = collect(0.0:sqrt(maxarea):L)
        Y = collect(0.0:sqrt(maxarea):h)
        grid = VoronoiFVM.Grid(X,Y)
        # set correct region numbers: 1 for the west contact, 2 for the east contact 
        # and 3 for the north and south boundaries

        # south
        VoronoiFVM.ExtendableGrids.bfacemask!(grid, [0.0,0.0],[1.0,0.0],3)

        # north
        VoronoiFVM.ExtendableGrids.bfacemask!(grid, [0.0,1.0],[1.0,1.0],3)

        # east
        VoronoiFVM.ExtendableGrids.bfacemask!(grid, [1.0,0.0],[1.0,1.0],2)

        # west
        VoronoiFVM.ExtendableGrids.bfacemask!(grid, [0.0,0.0],[0.0,1.0],1)
    else
        # Create 2D rectangular mesh refined close to the left boundary
        switches, triin = generate_refined_triangle_input(L=L, h=h, maxarea=maxarea)

        grid=VoronoiFVM.Grid(switches, triin)
    end
    #VoronoiFVM.plot(Plotter, grid)
    """
    if(Plotter != nothing)
        p = VoronoiFVM.ExtendableGrids.plot(grid, Plotter=Plotter)
        #display(p)
        if VoronoiFVM.ispyplot(Plotter)
            Plotter.savefig("grid.svg")
        else
            Plotter.svg(p,"grid.svg")
        end
    end
    """
    @printf("Press ENTER to continue...")
    readline();

    # Create and fill data 
    data=Data()
    data.R=10
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

    if trisurf && VoronoiFVM.ispyplot(Plotter)
        #Plotter.plot_trisurf(VoronoiFVM.ExtendableGrids.tridata(grid)..., steadystate[1,:], cmap = "cool")
        tri = VoronoiFVM.ExtendableGrids.tridata(grid)
        p = Plotter.plot_trisurf(tri..., steadystate[1,:], cmap="cool")
        Plotter.savefig("steadystate_surf.svg")
        # Spectral or cool
    else
        #VoronoiFVM.plot(Plotter, grid, steadystate[1,:])
        # Plots currently does not support this branch
        if(Plotter != nothing)
            p = VoronoiFVM.ExtendableGrids.plot(grid, steadystate[1,:], Plotter = Plotter, cmap="cool")
            if VoronoiFVM.ispyplot(Plotter)
                Plotter.savefig("steadystate_contour.svg")
            else
                Plotter.svg(p,"steadystate_contour.svg")
            end
        end
    end

    @printf("Press ENTER to continue...")
    readline()

    # Create Impedance system from steady state
    excited_spec=1
    excited_bc=1
    isys=VoronoiFVM.ImpedanceSystem(sys,steadystate,excited_spec,excited_bc)
    UZ=unknowns(isys)

    # define a test function for terminal current measurement 
    # [see e.g. j-fu/ysz/txt/impedance-derivation]
    factory=VoronoiFVM.TestFunctionFactory(sys)
    measurement_testfunction=testfunction(factory,[1],[2])

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

    # arrays containing the measured terminal current
    allIL=zeros(Complex{Float64},0)

    # frequency range [ω0, ω1] in geometric steps
    ω0=0.5
    ω1=1.0e4
    ω=ω0

    # time-steady impedance solution for every frequency ω
    UZ=unknowns(isys)
    noωs = Int64(ceil((log(2) + 4* log(10))/(log(6) - log(5))))
    allUZ = zeros(ComplexF64, (size(UZ)[2], noωs))
    allDirRes = zeros(noωs)

    # control whether an animation of the time evolution
    # of IL should be created and saved
    if VoronoiFVM.isplots(Plotter) && animate
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

        # add real and imaginary parts of the computed solution to animation frame 
        if VoronoiFVM.isplots(Plotter) && animate
            #p1 = VoronoiFVM.plot(Plotter, grid, real(UZ[1,:]), show=false, color=(1,0,0), label=LaTeXString("\$\\Re(u_a), \\; \\omega = "*string(ω))*"\$")
            #p2 = VoronoiFVM.plot(Plotter, grid, imag(UZ[1,:]), show=false, color=(0,0,1), label=LaTeXString("\$\\Im(u_a), \\; \\omega = "*string(ω))*"\$")
            #p1 = VoronoiFVM.plot(Plotter, grid, real(UZ[1,:]), show=false, color=(1,0,0), label="Re(u_a)")
            #p2 = VoronoiFVM.plot(Plotter, grid, imag(UZ[1,:]), show=false, color=(0,0,1), label="Im(u_a)")
            p1 = VoronoiFVM.ExtendableGrids.plot(grid, real(UZ[1,:]), Plotter = Plotter, show=false, color=(1,0,0), label="Re(u_a)")
            p2 = VoronoiFVM.ExtendableGrids.plot(grid, imag(UZ[1,:]), Plotter = Plotter, show=false, color=(0,0,1), label="Im(u_a)")
            Plotter.plot(p1,p2,layout = (2,1))
            Plotter.frame(anim)
        end

        # calculate aproximate solution
        # obtain measurement in frequency domain
        IL=freqdomain_impedance(isys,ω,steadystate,excited_spec,excited_bc,excited_bcval,dmeas_stdy, dmeas_tran)

        # record approximate solution
        push!(allomega, ω)
        push!(allIL,IL)

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
    UZω, it_mpgmressh = MPGMRESSh.mpgmressh(reshape(isys.F, (size(isys.F)[2],)), sys.matrix, isys.storderiv, iωs, nprecons = nprecons, maxiter = maxiter,
                                log = true, verbose = true, btol=1.0e-10,
                                convergence = MPGMRESSh.absolute, preconmethod = preconmethod,
                                preconreltol = preconreltol)
    # preconditioner solution method: LU factorization

    # now calculate the associated measurements for the unpreconditioned system
    # and the absolute residuals of the direct solutions
    allILMPGMRES = zeros(Complex{Float64}, length(allIL))
    for (i, iω) in enumerate(iωs)
        allILMPGMRES[i] = (dmeas_stdy*values(UZω[i]))[1] + iω * (dmeas_tran*values(UZω[i]))[1]
        allDirRes[i] = norm(isys.F[1,:] - (sys.matrix + iω * isys.storderiv)*allUZ[:,i])
    end

    @printf("Maximal absolute residual of the direct solutions across shifts: %1.5e\n", maximum(allDirRes))

    # the large penalty factor, however, spoils the absolute residual due to numerical artifacts
    absres = [norm(it_mpgmressh.b - (it_mpgmressh.barnoldi.A + shift .* it_mpgmressh.barnoldi.M) * it_mpgmressh.x[i]) for (i, shift) in enumerate(it_mpgmressh.barnoldi.allshifts)]
    @printf("Maximal absolute residual of MPGMRESSh across shifts: %1.5e\n", maximum(absres))
    @printf("Maximal relative residual of MPGMRESSh across shifts: %1.5e\n\n", maximum(absres ./ it_mpgmressh.residual.β))

    maxnormSols = maximum([maximum(abs.(allUZ[:,i] - it_mpgmressh.x[i])) for i=1:length(it_mpgmressh.barnoldi.allshifts)])
    @printf("Maximum linf norm of MPGMRESSh and direct solutions: %1.20e\n", maxnormSols)
    maxnormILs = maximum(abs.(allIL - allILMPGMRES))
    @printf("Maximum linf norm of resulting measurements: %1.20e\n\n", maxnormILs)

    @printf("----------------------------------------------------------------\n\n")

    @printf("MPGMRESSh solution with preconditioning: \n")
    # the jacobi preconditioning allows us to iterate normally with respect to the 
    # relative residual (Convergence.standard)
    UZω, it_mpgmresshprecon = MPGMRESSh.mpgmressh(reshape(b, (size(b)[2],)), K, M, iωs, nprecons = nprecons, maxiter = maxiter,
                                log = true, verbose = true, btol=1.0e-10,
                                convergence = MPGMRESSh.standard, preconmethod = preconmethod,
                                preconreltol = preconreltol)
    # preconditioner solution method: LU factorization

    # now calculate the associated measurements for the jacobi-preconditioned system
    allILMPGMRES = zeros(Complex{Float64}, length(allIL))
    for (i, iω) in enumerate(iωs)
        allILMPGMRES[i] = (dmeas_stdy*values(UZω[i]))[1] + iω * (dmeas_tran*values(UZω[i]))[1]
    end

    # the rescaling by the jacobi preconditioner has taken care of the excessive absolute residuals
    # in our case, they are of course now identical to the relative residuals
    absres = [norm(it_mpgmresshprecon.b - (it_mpgmresshprecon.barnoldi.A + shift .* it_mpgmresshprecon.barnoldi.M) * it_mpgmresshprecon.x[i]) for (i, shift) in enumerate(it_mpgmresshprecon.barnoldi.allshifts)]
    @printf("Maximal absolute residual across shifts: %1.5e\n", maximum(absres))
    @printf("Maximal relative residual across shifts: %1.5e\n\n", maximum(absres ./ it_mpgmresshprecon.residual.β))

    maxnormSols = maximum([maximum(abs.(allUZ[:,i] - it_mpgmresshprecon.x[i])) for i=1:length(it_mpgmresshprecon.barnoldi.allshifts)])
    @printf("Maximum linf norm of MPGMRESSh and direct solutions: %1.20e\n", maxnormSols)
    maxnormILs = maximum(abs.(allIL - allILMPGMRES))
    @printf("Maximum linf norm of resulting measurements: %1.20e\n", maxnormILs)

    # ----------------------------------------------------------------
    
    if VoronoiFVM.isplots(Plotter)
        if animate
            Plotter.gif(anim, "/tmp/impedancetest.gif", fps=15)
        end

        p=Plotter.plot(grid=true)
        #Plotter.plot!(p,real(allIL),imag(allIL),label="calc")
        #Plotter.plot!(p,real(allIL),imag(allIL),label=LaTeXString("approximated (\$\\widetilde{I}_{\\alpha}^{a}\$)"))
        Plotter.plot!(p,real(allILMPGMRES),imag(allILMPGMRES),label=LaTeXString("approximated (\$\\widetilde{I}_{\\alpha}^{a}\$)"))

        #Plotter.gui(p)
        #display(p)
        Plotter.svg(p,"impedance.svg")
    end

    return sys, isys, allIL, allILMPGMRES, allUZ, UZω, it_mpgmressh, it_mpgmresshprecon;
    #return steadystate, sys;
end

#function test()
#    main(dense=true) ≈ 0.23106605162049176 &&  main(dense=false) ≈ 0.23106605162049176
#end


end

