include("algos_multi_thread.jl")
include("io_routines.jl")

#---------------------------------------------------#

#= Run the main program to sample from the Gibbs distribution 
and save the output in a Julia file. =#
function run_main(param_set)
	cpu_time = @elapsed xdata, accept_rate = main(param_set)
	cpu_time /= 60
	println("Completed sampling, CPU time = $(cpu_time |> rnd) minutes.")
	jldsave(jld_file(param_set); xdata, param_set, accept_rate, cpu_time)
end

# Make text files for Veusz plots.
function make_veusz_data(params)
	# Unload the parameters and the JLD data.
	@unpack nmodes = params
	xdata, accept_rate, cpu_time = load(jld_file(params), "xdata", "accept_rate", "cpu_time")
	println("\n-----------------------------")
	println("Data loaded from $(jld_file(params))")
	println("CPU time of run was $((cpu_time/(60*24)) |> rnd) days.")
	println("Acceptance rate = $(accept_rate |> rnd).")
	println("Number of accepted samples = $(size(xdata,2)).")

	# Make the histogram data for Veusz.
	uvals = get_uvals(xdata)
	skew = skewness(uvals[:])
	write_data("hist_data.txt", "# All u values. \n", uvals[:])
	println("Skewness = $(skew |> rnd).")

	# Make the power-spectrum data for Veusz.
	uhat = get_uhat(xdata)
	pow_spec = zeros(nmodes)
	for k in 1:nmodes
		pow_spec[k] = mean( abs.(uhat[k,:]).^2 )
	end
	pow_spec = pow_spec./mean(pow_spec)
	kvals = collect(1:nmodes)
	ksq_spec = kvals.^(-2)
	ksq_spec = ksq_spec./mean(ksq_spec)
	header = "# kvals, u power spectrum, 1/k^2 spectrum. \n"
	write_data("spec_data.txt", header, [kvals pow_spec ksq_spec])

	# Extract the largest wave and a random wave to plot in Veusz.
	peak_idx = 0
	peak_val = 0.
	nwaves = min(1000, size(uvals,2))
	for i in 1:nwaves 
		umax = maximum(uvals[:,i])
		if umax > peak_val
			peak_idx = i
			peak_val = umax
		end
	end
	big_wave = wrap_vec( uvals[:, peak_idx] )
	ran_idx = 1 #rand(1:size(uvals,2))
	ran_wave = wrap_vec( uvals[:, ran_idx] )
	header = "# nmodes=$(nmodes), bprime=$(params.bprime), cratio=$(params.cratio), nsamps=$(params.min_samps_accept), nwaves=$(nwaves). \n"
	header = string(header,"# xi vals, uvals for big wave, uvals for random wave. \n")
	write_data("wave_data.txt", header, [xi_grid(nmodes) big_wave ran_wave])
end

#---------------------------------------------------#

# Warmup run call.
run_main(ParamSet(bprime=10, cratio=10, min_samps_accept=10))
make_veusz_data(ParamSet(bprime=10, cratio=10, min_samps_accept=10))


#= Make Veusa data calls.
make_veusz_data(ParamSet(bprime=40, cratio=30, min_samps_accept=2000))

make_veusz_data(ParamSet(bprime=20, cratio=120, min_samps_accept=2000))


make_veusz_data(ParamSet(nmodes=32, bprime=40, cratio=400, min_samps_accept=2000))
=#


