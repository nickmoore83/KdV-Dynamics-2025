using BenchmarkTools
include("algos_multi_thread.jl")
include("io_routines.jl")

# Compute the improvement ratio using the good g versus naive g.
function improve_ratio(; nmodes=16, bprime=10, cratio=10, 
		min_samps_good=2, min_samps_naive=1, min_time=0.1)

	# Set the parameters.
	params_good  = ParamSet(nmodes=nmodes, bprime=bprime, cratio=cratio, min_samps_accept=min_samps_good, min_time=min_time)
	params_naive = ParamSet(nmodes=nmodes, bprime=bprime, cratio=cratio, min_samps_accept=min_samps_naive, min_time=min_time, good_g=false)

	# Compute the acceptance rate using the good g.
	println("\n\n\n\n------------------------------")
	println("Running algorithm with good g.")
	time_good = @elapsed xdata, accept_rate_good = main(params_good)
	time_good /= 60
	println("Completed sampling with good g, CPU time = $(time_good |> rnd) minutes.")
	jldsave(jld_file(params_good); xdata, param_set=params_good, accept_rate=accept_rate_good, cpu_time=time_good)
	nsamps_good = size(xdata,2)
	skew_good = skewness(get_uvals(xdata))

	# Compute the acceptance rate using the naive g.
	println("\n------------------------------")
	println("Running algorithm with naive g."); 	t0 = time()
	xdata, accept_rate_naive = main(params_naive)
	time_naive = (time()-t0)/60
	nsamps_naive = size(xdata,2)
	skew_naive = skewness(get_uvals(xdata))
	improve_ratio = accept_rate_good/accept_rate_naive	

	println("\n------------------------------")
	println("Results")
	println("K=$(nmodes), bprime=$(bprime), cratio=$(cratio), min_time=$(min_time) minutes, nthreads=$(Threads.nthreads()).")
	println("Good g: $(time_good |> rnd) minutes for $(nsamps_good) samples.")
	println("Naive g: $(time_naive |> rnd) minutes for $(nsamps_naive) samples.")

	println("\ngood skewness = $(skew_good |> rnd)")
	println("good acceptance rate = $(accept_rate_good |> rnd)")
	println("Improvement ratio = $(improve_ratio |> rnd)")

	write_data("improve_ratio.txt", "# See code for explanation. \n", 
		[time_good; time_naive; nmodes; bprime; cratio; nsamps_good; nsamps_naive; skew_good; skew_naive; accept_rate_good; accept_rate_naive; improve_ratio] )
end

# Warmup
improve_ratio(min_samps_good=200, min_time=0.01)


#= 
Calls with K = 16
bprime = 40, cratio step = 30, up to max of 150
improve_ratio(nmodes=16, bprime=40, cratio=90, min_samps_good=2000, min_samps_naive=100, min_time=8*60)

bprime = 20, cratio step = 60, up to max of 300
improve_ratio(nmodes=16, bprime=20, cratio=240, min_samps_good=2000, min_samps_naive=100, min_time=8*60)


Calls with K = 32
bprime = 40, cratio step = 75, up to max of 375 or 450
improve_ratio(nmodes=32, bprime=40, cratio=300, min_samps_good=2000, min_samps_naive=100, min_time=8*60)

bprime = 20, cratio step = 150, up to max of 750 or 900
improve_ratio(nmodes=32, bprime=20, cratio=450, min_samps_good=2000, min_samps_naive=100, min_time=8*60)

=#
