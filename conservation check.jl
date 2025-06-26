using FFTW, Plots, Printf

function trap_periodic(fvals, x)
    h = x[2] - x[1]
    return h * sum(fvals)
end

function solve_kdv_pseudospectral(N=256, D=0.01, C2=0.0, C3=1.0, dt=0.1, tfinal=1.0)
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

        E[n] = 0.5 * trap_periodic(u.^2, x)
        M[n] = trap_periodic(u, x)
        H[n] = (C2/2) * trap_periodic(u_x.^2, x) - (C3/6) * trap_periodic(u.^3, x)
    end

    u_final = real(ifft(u_hat_k) * N)
    return x, u0, u_final, E, M, H, times
end

# Solve and get final result
x, u0, u_final, E, M, H, times = solve_kdv_pseudospectral()

# Plotting
p = plot(x, u_final, lw=2, label="Final solution", xlabel="x", ylabel="u(x, t)",
         title="KdV Solution (Pseudo-Spectral)", legend=:topright)
plot!(p, x, u0, lw=2, ls=:dash, label="Initial condition")
display(p)

println("\nEnergy error: ", maximum(abs.(E .- E[1])))
println("Momentum error: ", maximum(abs.(M .- M[1])))
println("Hamiltonian error: ", maximum(abs.(H .- H[1])))


println(@sprintf("%6s | %12s | %12s | %12s", "t", "E", "M", "H"))
println("-"^52)
for i in eachindex(times)
    println(@sprintf("%6.2f | %12.5e | %12.5e | %12.5e", times[i], E[i], M[i], H[i]))
end

# L2 error over time for E, M, H
function l2_time_error(Q, dt)
    return sqrt(dt * sum((Q .- Q[1]).^2))
end

l2_E = l2_time_error(E, dt)
l2_M = l2_time_error(M, dt)
l2_H = l2_time_error(H, dt)

println("\nL2 error over time:")
println("L2 error (E): ", @sprintf("%.5e", l2_E))
println("L2 error (M): ", @sprintf("%.5e", l2_M))
println("L2 error (H): ", @sprintf("%.5e", l2_H))
