#include("algos_multi_thread.jl")

#---------------------------------------------------#
# Find the uhats from a set of samples.
function get_uhat(xsamps)
	nmodes = size(xsamps,1) ÷ 2
	nsamps = size(xsamps,2)
	uhat = zeros(ComplexF64, nmodes, nsamps)
	for n in 1:nsamps
		uhat[:,n] = xsamps[1:nmodes,n] - im*xsamps[nmodes+1:end,n]
	end
	return uhat
end

# Find the physical u values a set of samples.
function get_uvals(xsamps)
	# Calculate the uhat values with appropriate tweaks for FFT.
	nmodes = size(xsamps,1) ÷ 2
	nsamps = size(xsamps,2)
	uhat = zeros(ComplexF64, 2*nmodes, nsamps)
	for n in 1:nsamps
		for k in 1:nmodes
			mult = (-1)^(k)
			uhat[k+1, n] = (xsamps[k,n] - im*xsamps[k+nmodes,n]) * mult
			uhat[2*nmodes-k+1, n] = (xsamps[k,n] + im*xsamps[k+nmodes,n]) * mult
		end
	end
	# Inverse FFT to physical space.
	umatrix = zeros(size(uhat))
	for i in 1:nsamps
		uvals = 2*nmodes*ifft(uhat[:,i]) |> real 	# physical space
		# Check that the imaginary part is small.
		max_imag = imag.(uvals) |> maximum
		@test max_imag < 1e-6
		umatrix[:,i] = uvals	
	end
	return umatrix
end

# Plot the Dirichlet kernel
function plot_Dirichlet(K::Integer; E0::Real=1.0)
	xsamp = sqrt(E0/(2*pi*K)) * [ones(K); zeros(K)]
	uvals = get_uvals(xsamp)
	write_data("Dirichlet.txt", "# xi_vals, uvals \n", 
		[xi_grid(K) wrap_vec(uvals)] )
end

#---------------------------------------------------#
# IO functions.
rnd(x) = round(x,sigdigits=3)
jld_file(params) = string("Data/Data_K=$(params.nmodes)_bprime=$(params.bprime)_cratio=$(params.cratio)_nsamps=$(params.min_samps_accept).jld2")
wrap_vec(vec) = [vec; vec[1]]
xi_grid(K::Integer) = -pi .+ pi/K*(0:2*K)

# Write output to a text file; used for histogram data.
function write_data(file, header, data)
	open(file; write=true) do io
		write(io, header)
		writedlm(io, data)
	end
end
