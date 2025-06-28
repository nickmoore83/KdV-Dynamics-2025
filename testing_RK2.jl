using FFTW, Plots, PrettyTables, Statistics
include("linprop_KdV.jl")

N = 256                    # Number of grid points
x = -π .+ 2π * (0:N-1)/N   # Grid on [-π, π)
dx = x[2] - x[1]

# FFT wave number vector for spectral derivative
kvec = vcat(0:N÷2-1, -N÷2:-1)  # Assumes even N


###----------------------------------
u_init = cos.(x)                   # Example: cos(x) initial condition
u0 = fft(u_init) / N              # Normalized Fourier coefficients


###-----------------------------------------
# RUN strictly nonlinearity --> u_t = -C3u*u_x
C2 = 0       # dispersion coefficient
C3 = 1       # nonlinearity coefficient
D = 0        # diffusion coefficient

a = 0.0        # initial time
h = 0.1      # time step
tfin = 1.0     # final time

t4, uk4, E4, M4, H4 = linprop_KdV(C2, C3, D, kvec, a, u0, h, tfin);

# Prepare data for table (align lengths)
data = hcat(t4[1:end-1], E4, M4, H4)

# Print table
println()
println()
println()
println("u_t + uu_x  = 0 \n")
pretty_table(data, header=["Time", "E", "M", "H"], formatters=ft_printf("%.6f"))
println("Energy error: ", (E4[1] - E4[end])/E4[1])
println("Momentum error: ", (M4[1] - M4[end])/M4[1])
println("Hamiltonian error: ", (H4[1] - H4[end])/H4[1])