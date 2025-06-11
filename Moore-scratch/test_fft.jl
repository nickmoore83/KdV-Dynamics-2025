# Test out FFT Stuff in Julia
using FFTW, LinearAlgebra

## Modify this function.
function nice_fft(uvals)
	return fft(uvals)
end

## Test nice_fft for x in interval [0,2*pi).
function test(N::Int)
	println("\nRunning test")
	dx = 2*pi/N
	x = dx*(0:N-1)
	kvec = fftfreq(N)*N
	println("kvec = $kvec")
	
	# Construct a complex-valued function with random Fourier coefficients.
	uvals = 0*x
	uhat_ex = randn(Complex{Float64},N)
	for idx = 1:N
		uh = uhat_ex[idx]
		k = kvec[idx]
		uvals += uh*exp.(im*k*x)
	end

	# Call nice_fft
	uhat = nice_fft(uvals)

	# Print error
	uhat_err = norm(uhat_ex - uhat)
	println("Error in uhat = $(uhat_err)")
end

test(8)