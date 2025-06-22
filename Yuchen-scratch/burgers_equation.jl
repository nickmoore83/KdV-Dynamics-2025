using FFTW, Plots

function solve_kdv_pseudospectral(N=256, D=0.01, C2=0.0, C3=1.0, dt=0.0001, tfinal=1.0)
    L = 2π
    x = L * (0:N-1) / N
    k = fftfreq(N) * N
    ik = im .* k
    alpha_k = D .* k.^2 .- C2 .* ik.^3         # Linear part
    propagator = exp.(-alpha_k * dt)           # Exponential linear propagator

    # Initial condition
    u = sin.(x) .+ 0.5sin.(2x)
    u_hat_k = fft(u) / N

    nsteps = Int(round(tfinal / dt))

    for n in 1:nsteps
        u = real(ifft(u_hat_k) * N)            # Back to physical space
        u_x = real(ifft(ik .* u_hat_k) * N)    # Derivative in physical space
        nonlinear_term = u .* u_x
        v_hat_k = fft(nonlinear_term) / N      # Nonlinear term in Fourier space

        # Time stepping
        u_hat_k = (u_hat_k .- C3 * dt .* v_hat_k) .* propagator
    end

    u_final = real(ifft(u_hat_k) * N)
    return x, u_final
end

x, u_final = solve_kdv_pseudospectral()

u0 = sin.(x) .+ 0.5sin.(2x)

# Plot comparison
plot(x, u0, label="Initial Condition", lw=2, linestyle=:dash, color=:blue)
plot!(x, u_final, label="Final Solution", lw=2, color=:red)
xlabel!("x")
ylabel!("u(x)")
title!("Burgers Equation with Pseudo-Spectral Method")
gui();
