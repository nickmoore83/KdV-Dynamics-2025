using FFTW
using Parameters
using Plots
plotly()

# Test the routines for time-stepping KdV
# KdV: u_t + C2 u_xxx + C_3 u*u_x - D u_xx = 0

# Data structure for the PDE parameters.
# TO DO: structure your KdV solver to accept this format.
@with_kw struct PDE_params
	D::Float64 = 0.		# Diffusion
	C2::Float64 = 1.	# Dispersion
	C3::Float64 = 1.	# Nonlinearity
end

# TO DO: List the included files here.
function energy(uhat::AbstractVector)
    return sum(abs2, uhat)
end

function reconstruct_uhat(uhat_pos)
    K = length(uhat_pos) - 1
    uhat_neg = conj(reverse(uhat_pos[2:end-1]))
    return vcat(uhat_pos, uhat_neg)
end

include("trap_periodic.jl")
include("kdv_solve_final.jl") # Forward Euler
include("KdV_solver_final.jl") # RK2

# The main testing routine.
function test_kdv()

	# TO DO: Specify which routine to use as the KdV solver here.
	# if use RK2, uncomment this.
	kdv_solve = (params, uhat0_pos, h, tfin; dealias=false) -> KdV_solver(params, 0.0, uhat0_pos, h, tfin; dealias=dealias)

	# Specify the cutoff wavenumber.
	BigK = 16

	#I. Qualitative test with Burgers equation (i.e. C2=0)
		#1. Set the parameters.
		pde_params = PDE_params(D=0.05, C2=0., C3=1.)
		dt = 0.02; tfin = 1.

		#2. Set the initial condition.
		# Format: [uh_0, uh_1, ..., uh_K]
		uhat_0 = zeros(ComplexF64, BigK + 1)
		uhat_0[2] = 1.	# Can modify this.
		
		#3. Time-step the KdV equation.
		tvals, uhat, En, Mom, Ham, H2, H3 = kdv_solve(pde_params, uhat_0, dt, tfin; dealias = false)

		# NOTE: the format of uhat should be uhat[kvals,tvals]

		#4. TO DO: Make plots of solution.
		K = BigK
		N = 2 * K
		x = -π .+ 2π * (0:N-1) / N 

		uhat0_full = reconstruct_uhat(uhat[:, 1])
		u0 = real(ifft(uhat0_full) * N)

		uhatf_full = reconstruct_uhat(uhat[:, end])
		uf = real(ifft(uhatf_full) * N)

    	p = plot(x, u0, label = "t = $(round(tvals[1], digits=2))", lw=2)
		plot!(p, x, uf, label = "t = $(round(tvals[end], digits=2))", lw=2)
		xlabel!("x")
		ylabel!("u(x,t)")
		title!("Burgers equation: initial and final condition")
		display(p)

	#II. Quantitative test using the exact solution of the linear Airy PDE (i.e. C3=0)
		#1. Set the parameters.
		pde_params = PDE_params(D=rand(), C2=rand(), C3=0.)

		#2. Set the initial conditions.
		uhat_0 = randn(Complex{Float64}, BigK+1)
		# Note: I think we should normalize to make energy equal to one for this test.
		uhat_0 = uhat_0./energy(uhat_0)

		#3. Time-step the KdV equation.
		tvals, uhat, En, Mom, Ham, H2, H3 = kdv_solve(pde_params, uhat_0, dt, tfin; dealias = false)

		#4. TO DO: Calculate the relative L2 error between the exact and numerical solution at tfin.
		uhat_exact = uhat_0 .* exp.(-(pde_params.D .* (0:BigK).^2 .- pde_params.C2 .* im .* (0:BigK).^3) * tfin)
		err1 = energy(uhat_exact - uhat[:,end])/energy(uhat_exact)
		# println("L2 error between exact and numerical solution is $(err1)")
		println("D=$(pde_params.D), C2=$(pde_params.C2), C3=$(pde_params.C3) | L2 error = $(err1)")

		#5. TO DO: Also calculate the error with a dt/2 and estimate the order of convergence.
		tvals, uhat, En, Mom, Ham, H2, H3 = kdv_solve(pde_params, uhat_0, dt/2, tfin; dealias = true)
		err2 = energy(uhat_exact - uhat[:,end])/energy(uhat_exact)
		order = log2(err1 / err2)
		#println("The estimated order is $(order)")
		println("D=$(pde_params.D), C2=$(pde_params.C2), C3=$(pde_params.C3)| L2 error at t = 0.01 = $(err2) | Estimated order = $(order)")

	#III. Test conserved quantities of the KdV equation (i.e. D=0)
		#1. Set the parameters.
		pde_params = PDE_params(D=0., C2=rand(), C3=rand())

		#2. Set the initial conditions.
		uhat_0 = randn(Complex{Float64}, BigK+1)
		uhat_0 = uhat_0./energy(reconstruct_uhat(uhat_0))

		#3. Time-step the KdV equation.
		tvals, uhat, En, Mom, Ham, H2, H3 = kdv_solve(pde_params, uhat_0, dt, tfin; dealias = true)

        p_ham = plot(tvals, H2, label="H2 (Dispersion)", lw=2)
        plot!(p_ham, tvals, H3, label="H3 (Nonlinearity)", lw=2)
        xlabel!("Time")
        ylabel!("Hamiltonian Components")
        title!("Hamiltonian Components vs Time")
        gui()

		#4. Calculate relative error between the initial and final values of E, M, and H.
		# Here is an example of the code for Energy; Please FILL IN the code for M and H.
		E_err = abs(En[end] - En[1]) / abs(En[1])
		M_err = abs(Mom[end] - Mom[1]) / abs(Mom[1])
		H_err = abs(Ham[end] - Ham[1]) / abs(Ham[1])
		#println("Relative errors of E, M, H are E: $E_err, M: $M_err, H: $H_err")
		println("D=$(pde_params.D), C2=$(pde_params.C2), C3=$(pde_params.C3) | Relative errors E: $E_err, M: $M_err, H: $H_err")


		#5. TO DO: Also estimate the order of convergnece using dt/2.
		tvals2, uhat2, En2, Mom2, Ham2, H2, H3 = kdv_solve(pde_params, uhat_0, dt/2, tfin; dealias = true)

		E_err2 = abs(En2[end] - En2[1]) / abs(En2[1])
		M_err2 = abs(Mom2[end] - Mom2[1]) / abs(Mom2[1])
		H_err2 = abs(Ham2[end] - Ham2[1]) / abs(Ham2[1])

		# Order estimate (use energy as example; can compute for M or H similarly)
		order_E = log2(E_err / E_err2)
		order_M = log2(M_err / M_err2)
		order_H = log2(H_err / H_err2)

		#println("Estimated order of convergence (E) = $order_E")
		#println("Estimated order of convergence (M) = $order_M")
		#println("Estimated order of convergence (H) = $order_H")
		println("D=$(pde_params.D), C2=$(pde_params.C2), C3=$(pde_params.C3) | Relative errors at t = 0.01: $E_err2, M: $M_err2, H: $H_err2")
		println("D=$(pde_params.D), C2=$(pde_params.C2), C3=$(pde_params.C3) | Estimated order of convergence E: $order_E, M: $order_M, H: $order_H")

end

test_kdv()