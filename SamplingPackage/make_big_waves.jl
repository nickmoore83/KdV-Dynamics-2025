include("algos_multi_thread.jl")
include("io_routines.jl")

#---------------------------------------------------#


# Make text files for Veusz plots.
function make_veusz_waves(params, label, nwaves, seed)
	# Unload the parameters and the JLD data.
	@unpack nmodes = params
	xdata, param_set, accept_rate, cpu_time = load(jld_file(params), "xdata", "param_set", "accept_rate", "cpu_time")

	# Make the histogram data for Veusz.
	uvals = get_uvals(xdata)
	skew = skewness(uvals[:])
	println("\n------------------------------")
	println("Making Big Waves Data.")
	println("K=$(param_set.nmodes), bprime=$(param_set.bprime), cratio=$(param_set.cratio).")
	println("\nSkewness = $(skew |> rnd)")
	println("Acceptance rate = $(accept_rate |> rnd)")
	println("CPU time = $(cpu_time |> rnd)")

	# Extract the largest wave and a random wave to plot in Veusz.
	peak_idx = 0
	peak_val = 0.
	nwaves_total = size(uvals,2)
	nwaves = min(nwaves, nwaves_total)
	for i in shuffle(Xoshiro(seed), 1:nwaves_total)[1:nwaves]
		maxU = maximum(uvals[:,i])
		if maxU > peak_val
			peak_idx = i
			peak_val = maxU
		end
	end
	println("Largest wave height = $(peak_val |> rnd)")
	big_wave = wrap_vec( uvals[:, peak_idx] )
	ran_idx = 1 #rand(1:size(uvals,2))
	ran_wave = wrap_vec( uvals[:, ran_idx] )
	header = "# nmodes=$(nmodes), bprime=$(params.bprime), cratio=$(params.cratio), nsamps=$(params.min_samps_accept), nwaves=$(nwaves). \n"
	header = string(header,"# xi vals, uvals for big wave, uvals for random wave. \n")
	write_data("waves$(label).txt", header, [xi_grid(nmodes) big_wave ran_wave])
end


# Set parameters and call whichever routine is desired.
function big_waves(nmodes, bprime, cratio; nwaves=1000, seed=0, min_samps_accept=[2000,2000,2000])
	paramsA = ParamSet(nmodes=nmodes, bprime=bprime, cratio=cratio[1], min_samps_accept=min_samps_accept[1])
	paramsB = ParamSet(nmodes=nmodes, bprime=bprime, cratio=cratio[2], min_samps_accept=min_samps_accept[2])
	paramsC = ParamSet(nmodes=nmodes, bprime=bprime, cratio=cratio[3], min_samps_accept=min_samps_accept[3])
	make_veusz_waves(paramsA, "A", nwaves, seed)
	make_veusz_waves(paramsB, "B", nwaves, seed)
	make_veusz_waves(paramsC, "C", nwaves, seed)	
end

# Call
#big_waves(16, 20, [0,150,300], nwaves=500, seed=0)
#big_waves(16, 40, [0,75,150], nwaves=500, seed=0)
#big_waves(16, 60, [0,50,100], nwaves=500, seed=0)

# Plots on overleaf are already final (even though they were done with older data)
#big_waves(32, 20, [0,450,900], nwaves=500, seed=4)
big_waves(32, 40, [0,225,450], nwaves=500, seed=1)
#big_waves(32, 60, [0,150,300], nwaves=500, seed=0)




# Before Oct 16, 2024
#big_waves(16, 40, [0,80,160], nwaves=500, seed=1)
#big_waves(16, 60, [0,50,100], nwaves=500, seed=0)
#big_waves(32, 40, [0,200,400], nwaves=500, seed=0)
#big_waves(32, 40, [0,225,450], nwaves=500, seed=0, min_samps_accept=[5000,1000,1000])
#big_waves(32, 60, [0,150,300], nwaves=500, seed=23) #Good seeds: 6, 7, 10, 12, 16, 17, 18, 20, 23
