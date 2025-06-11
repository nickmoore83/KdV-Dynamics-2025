# Goal: Test the speed of the algorithm in parallel versus serial.

using BenchmarkTools

include("algos_multi_thread.jl")
include("io_routines.jl")

# Set parameters.
#nmodes = 16; nsamps_per_thread = 2*10^5
nmodes = 128; nsamps_per_thread = 1*10^4

params_par = ParamSet(nmodes=nmodes, nsamps_per_thread=nsamps_per_thread)
params_ser = ParamSet(nmodes=nmodes, nsamps_per_thread=nsamps_per_thread, parallel=false)

# These constants don't matter so much since we are just timing the code.
alpha = 1.2
rej_const = max_fg(alpha, params_par)
sigmas = sqrt.(sig_squares(alpha, params_par.nmodes, params_par.bprime) )

# Measure the time required to draw a batch of samples from the Gibbs measure.
par_time = @belapsed draw_gibbs_batch(alpha, rej_const, sigmas, params_par)
ser_time = @belapsed draw_gibbs_batch(alpha, rej_const, sigmas, params_ser)
speedup = ser_time/par_time

# Print results.
println("Parallel time $(par_time |> rnd)")
println("Serial time $(ser_time |> rnd)")
println("Speedup $(speedup |> rnd)")


