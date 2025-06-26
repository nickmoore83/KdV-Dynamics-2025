using FFTW, Plots, Printf

function trap_periodic(fvals, x)
    h = x[2] - x[1]
    return h * sum(fvals)
end

# Exponential spectral filter
function exp_filter(k::Vector{Int}, N::Int; alpha=36, m=36)
    return exp.(-alpha * (abs.(k) ./ N).^m)
end

function solve_kdv_pseudospectral_filtered(N=256, D=0.01, C2=0.0, C3=1.0, dt=0.1, tfinal=1.0)
    N2 = 2 * N
    L = 2π
    x = L * (0:N2-1) / N2
    k = [0:N2÷2; -N2÷2+1:-1]
    ik = im .* k
    alpha_k = D .* k.^2 .- C2 .* (ik).^3
    propagator = exp.(-alpha_k * dt)

    filter_rho = exp_filter(k, N2)

    u0 = sin.(x)

    u_hat_k = fft(u0) / N2

    nsteps = Int(round(tfinal / dt))
    times = (1:nsteps) .* dt
    E = zeros(nsteps)
    M = zeros(nsteps)
    H = zeros(nsteps)

    for n in 1:nsteps
        u = real(ifft(u_hat_k) * N2)
        u_x = real(ifft(ik .* u_hat_k) * N2)

        nonlin = u .* u_x
        v_hat_k = fft(nonlin) / N2
        v_hat_k_filtered = v_hat_k .* filter_rho

        u_hat_k = (u_hat_k .- C3 * dt .* v_hat_k_filtered) .* propagator

        # Conserved quantities integration on x grid (length N2)
        E[n] = 0.5 * trap_periodic(u.^2, x)
        M[n] = trap_periodic(u, x)
        H[n] = (C2/2) * trap_periodic(u_x.^2, x) - (C3/6) * trap_periodic(u.^3, x)
    end

    u_final = real(ifft(u_hat_k) * N2)
    return x, u0, u_final, E, M, H, times
end

# Run simulation
x, u_initial, u_final, E, M, H, times = solve_kdv_pseudospectral_filtered()

# Plot initial and final profiles
plot(x, u_final, lw=2, label="Final", xlabel="x", ylabel="u(x)", title="KdV Solution (Filtered)")
plot!(x, u_initial, lw=2, ls=:dash, label="Initial")

# Print conservation table
println(@sprintf("%6s | %12s | %12s | %12s", "t", "E", "M", "H"))
println("-"^52)
for i in eachindex(times)
    println(@sprintf("%6.2f | %12.5e | %12.5e | %12.5e", times[i], E[i], M[i], H[i]))
end

# Print max deviation from initial value
println("\nMax E error: ", @sprintf("%.5e", maximum(abs.(E .- E[1]))))
println("Max M error: ", @sprintf("%.5e", maximum(abs.(M .- M[1]))))
println("Max H error: ", @sprintf("%.5e", maximum(abs.(H .- H[1]))))

dt = 0.1
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
