# Goal: Profile the code to see which steps take the longest.
# Candidates are draw_xhats() and compute_fg_ratio()
# Within draw_xhats() is randn(), mult_sigs!(), normalize_samps!()
# Within compute_fg_ratio() is compute_h2() and compute_h3()

#using Profile
#using Juno
#using ProfileView
#using ProfileVega
using StatProfilerHTML

include("algos_multi_thread.jl")
include("io_routines.jl")

# Set parameters.
nmodes = 16; nsamps_per_thread = 2*10^5
#nmodes = 128; nsamps_per_thread = 1*10^4

param_set = ParamSet(nmodes=nmodes, nsamps_per_thread=nsamps_per_thread)

# These constants don't matter so much since we are just timing the code.
alpha = 1.2
rej_const = max_fg(alpha, param_set)
sigmas = sqrt.(sig_squares(alpha, param_set.nmodes, param_set.bprime) )

# Measure the time required to draw a batch of samples from the Gibbs measure.
draw_gibbs_batch(alpha, rej_const, sigmas, param_set)
@profilehtml draw_gibbs_batch(alpha, rej_const, sigmas, param_set)

