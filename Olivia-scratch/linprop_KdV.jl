using FFTW

function linprop_KdV(C2, C3, D, kvec, a, u0, h, tfin)
    N = length(kvec)
    nsteps = Int(round((tfin - a) / h))
    
    # Precompute linear operator
    alpha = D .* kvec.^2 .- C2 .* im .* kvec.^3
    E = exp.(-alpha .* h)  # exponential term
    
    # Initialize Fourier modes over time
    uk = zeros(ComplexF64, N, nsteps+1)
    uk[:, 1] = u0  # initial condition in Fourier space
    
    t = range(a, length=nsteps+1, step=h)
    
    for n in 1:nsteps
        # U from uk at time n
        u_phys = real(ifft(uk[:, n]) * N)
        
        # Compute u_x:
        # Derivative: i k * uk, then ifft
        ux_phys = real(ifft(im .* kvec .* uk[:, n]) * N)
        
        # Get vk
        vk = fft(u_phys .* ux_phys) / N
        
        # Update Fourier coefficients uk for next step
        uk[:, n+1] = E .* (uk[:, n] - C3 * h .* vk)
    end
    
    return t, uk
end