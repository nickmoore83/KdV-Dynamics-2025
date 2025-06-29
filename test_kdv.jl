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

# The main testing routine.
function test_kdv()

	# TO DO: Specify which routine to use as the KdV solver here.
	kdv_solve = # You can fill in whatever you call your KdV solver here

	# Specify the cutoff wavenumber.
	BigK = 16

	#I. Qualitative test with Burgers equation (i.e. C2=0)
		#1. Set the parameters.
		pde_params = PDE_params(D=0.05, C2=0., C3=1.)
		dt = 0.02; tfin = 1.

		#2. Set the initial condition.
		# Format: [uh_0, uh_1, ..., uh_K]
		uhat_0 = zeros(BigK+1)
		uhat_0[2] = 1.	# Can modify this.
		
		#3. Time-step the KdV equation.
		tvals,uhat,En,Mom,Ham = kdv_solve(pde_params, uhat_0, dt, tfin)
		# NOTE: the format of uhat should be uhat[kvals,tvals]

		#4. TO DO: Make plots of solution.


	#II. Quantitative test using the exact solution of the linear Airy PDE (i.e. C3=0)
		#1. Set the parameters.
		pde_params = PDE_params(D=rand(), C2=rand(), C3=0.)

		#2. Set the initial conditions.
		uhat_0 = randn(Complex{Float64}, BigK)
		# Note: I think we should normalize to make energy equal to one for this test.
		uhat_0 = uhat_0./energy(uhat_0)

		#3. Time-step the KdV equation.
		tvals,uhat,En,Mom,Ham = kdv_solve(pde_params, uhat_0, dt, tfin)

		#4. TO DO: Calculate the relative L2 error between the exact and numerical solution at tfin.
		uhat_exact = #FILL IN the formula for uhat_exact at tfin.
		err1 = energy(uhat_exact - uhat[:,end])/energy(uhat_exact)
		println("L2 error between exact and numerical solution is $(err1)")

		#5. TO DO: Also calculate the error with a dt/2 and estimate the order of convergence.
		tvals,uhat,En,Mom,Ham = kdv_solve(pde_params, uhat_0, dt/2, tfin)
		err2 = energy(uhat_exact - uhat[:,end])/energy(uhat_exact)
		order = # FILL IN the appropriate formula using err1 and err2
		println("The estimated order is $(order)")


	#III. Test conserved quantities of the KdV equation (i.e. D=0)
		#1. Set the parameters.
		pde_params = PDE_params(D=0., C2=rand(), C3=rand())

		#2. Set the initial conditions.
		uhat_0 = randn(Complex{Float64}, BigK)
		uhat_0 = uhat_0./energy(uhat_0)

		#3. Time-step the KdV equation.
		tvals,uhat,En,Mom,Ham = kdv_solve(pde_params, uhat_0, dt, tfin)

		#4. Calculate relative error between the initial and final values of E, M, and H.
		# Here is an example of the code for Energy; Please FILL IN the code for M and H.
		E_err1 = abs(En[end]-En[1])/abs(En[1])
		println("Relative errors of E, M, H are $(E_err1)")

		#4. TO DO: Also estimate the order of convergnece using dt/2.
end

test_kdv()
