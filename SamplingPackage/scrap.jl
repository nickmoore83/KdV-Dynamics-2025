using SharedArrays
using Parameters
using BenchmarkTools
using Distributed

function test_fft()
	N = 16
	dx = 2*pi/N
	xgrid = -pi .+ (0:N-1)*dx
	eikx(k) = exp.(im*k*xgrid)
	kvals = [0:N÷2-1; -N÷2:-1]

	# Set uhat
	ncoeffs = N÷2+1
	uhat = zeros(ComplexF64,N)
	uhat[1:ncoeffs] = randn(ComplexF64,ncoeffs)

	# Enforce conjugate relation.
	uhat[1] = real(uhat[1])
	uhat[N:-1:N÷2+1] = conj( uhat[2:N÷2+1] )
	uhat[N÷2+1] = real(uhat[N÷2+1])
	println("\n uhat: "); display(uhat .|> rnd)

	# Compute the true u.
	utrue = zeros(ComplexF64,N)
	for idx in 1:N
		utrue .+= uhat[idx] * eikx(kvals[idx])
	end
	# Compute numerical u.
	uvals = length(uhat)*ifft(uhat .* (-1).^(kvals) )
	# Display both.
	#println("\n utrue: "); display(utrue .|> rnd)
	#println("\n uvals: "); display(uvals .|> rnd)
	
	# Print the error.
	err = maximum(abs.(utrue-uvals))
	println("\n Error = ", round(err, sigdigits=3),"\n")

	# Use Parseval to compute the energy.
	K = N÷2
	summ = 0.5*( abs(uhat[1])^2 + abs(uhat[K+1])^2 )
	for k in 1:K-1
		summ += abs(uhat[k+1])^2
	end
	En_Par = 2*pi*summ
	# Compute the energy via integration.
	En = 0.5*dx*sum(uvals.^2)
	println("Energies: $(En_Par |> rnd); $(En |> rnd)")
	println("Difference = $(abs(En_Par-En) |> rnd)")
end
#test_fft()

function collect_data()
	K = 3
	samps_per_thread = 5
	nthreads = Threads.nthreads()

	# Initialize the dictionary to store data accross different threads
	xdict = Dict()
	for thread in 1:nthreads
		xdict[thread] = 0
	end

	# Draw random samples and accept/reject in parallel.
	Threads.@threads for thread in 1:nthreads
		xacc = zeros(K,0)
		for samps in 1:samps_per_thread
			xnew = randn(K)
			if rand() > 0.5
				xacc = [xacc xnew]
			end
		end
		xdict[thread] = xacc
	end
	
	# Combine all accepted samples in serial.
	xall = zeros(K,0)
	for thread in 1:nthreads
		xall = [xall xdict[thread] ]
	end
	display(xall)
end
#collect_data()



include("algos_multi_thread.jl")
function test_algos_multi_thread()
	param_set = ParamSet()
	xsamps = draw_samps(param_set)
end

test_algos_multi_thread()
