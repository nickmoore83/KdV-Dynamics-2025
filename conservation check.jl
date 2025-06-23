using FFTW, Plots, Printf

function trap(y, x)
    area = 0.0
    for i in 1:length(x)-1
        dx = x[i+1] - x[i]
        area += dx/2 * (y[i] + y[i+1])
    end
    return area
end

function solve_kdv_pseudospectral(N=256, D=0.01, C2=1.0, C3=0.0, dt=0.1, tfinal=1.0)
    L = 2π
    x = L * (0:N-1) / N
    k = [0:N÷2; -N÷2+1:-1]
    ik = im .* k
    alpha_k = D .* k.^2 .- C2 .* ik.^3         # Linear part
    propagator = exp.(-alpha_k * dt)           # Exponential linear propagator

    # Initial condition
    u0 = sin.(x)
    u = copy(u0)
    u_hat_k = fft(u) / N

    nsteps = Int(round(tfinal / dt))

    E = zeros(nsteps)
    M = zeros(nsteps)
    H = zeros(nsteps)
    times = (1:nsteps) .* dt

    for n in 1:nsteps
        u = real(ifft(u_hat_k) * N)            # Back to physical space
        u_x = real(ifft(ik .* u_hat_k) * N)    # Derivative in physical space
        nonlinear_term = u .* u_x
        v_hat_k = fft(nonlinear_term) / N      # Nonlinear term in Fourier space

        # Time stepping
        u_hat_k = (u_hat_k .- C3 * dt .* v_hat_k) .* propagator

        E[n] = 0.5 * trap(u.^2, x)
        M[n] = trap(u, x)
        H[n] = (C2/2) * trap(u_x.^2, x) - (C3/6) * trap(u.^3, x)
    end

    u_final = real(ifft(u_hat_k) * N)
    return x, u_final, E, M, H, times
end

# Solve and get final result
x, u_final, E, M, H, times = solve_kdv_pseudospectral()

# Initial condition for plotting
u_initial = sin.(x)
# Plotting
p = plot(x, u_final, lw=2, label="Final solution", xlabel="x", ylabel="u(x, t)",
         title="KdV Solution (Pseudo-Spectral)", legend=:topright)
plot!(p, x, u_initial, lw=2, ls=:dash, label="Initial condition")
display(p)

# Initial values (at time t = 0)
E0, M0, H0 = E[1], M[1], H[1]

# Relative errors
E_err = abs.((E .- E0) ./ E0)
M_err = abs.((M .- M0) ./ M0)
H_err = abs.((H .- H0) ./ H0)

println("    t     |    E_error    |    M_error    |    H_error")
for i in 1:length(times)
    @printf("%8.4f | %12.4e | %12.4e | %12.4e\n", times[i], E_err[i], M_err[i], H_err[i])
end
