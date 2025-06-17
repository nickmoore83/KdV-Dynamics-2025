using FFTW, Plots

function solve_burgers_pseudospectral(N=256, D=0.01, dt=0.001, tfinal=1.0)
    L = 2π
    x = L * (0:N-1)/N
    k = [0:N÷2; -N÷2+1:-1]  # Wave numbers
    
    u0 = sin.(x) .+ 0.5sin.(2x)
    u = copy(u0)
    
    propagator = exp.(-D * k.^2 * dt)
    
    nsteps = Int(round(tfinal/dt))
    
    for step in 1:nsteps
        # Compute nonlinear term in physical space
        u_hat = fft(u)
        ux_hat = im * k .* u_hat
        ux = real(ifft(ux_hat))
        nonlinear_term = -u .* ux  # -u*u_x for Burgers equation
        
        # Transform nonlinear term to Spectral space
        v_hat = fft(nonlinear_term)
        
        # Apply the propagator
        u_hat = propagator .* (u_hat + dt * v_hat)
        
        # Transform back to physical space
        u = real(ifft(u_hat))
    end
    
    return x, u0, u
end

# Solve and plot
x, u0, u_final = solve_burgers_pseudospectral()

# Plot comparison
plot(x, u0, label="Initial Condition", lw=2, linestyle=:dash, color=:blue)
plot!(x, u_final, label="Final Solution", lw=2, color=:red)
xlabel!("x")
ylabel!("u(x)")
title!("Burgers Equation with Pseudo-Spectral Method")
gui();