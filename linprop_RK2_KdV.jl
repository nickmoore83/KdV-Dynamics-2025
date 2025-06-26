using FFTW
include("trap_periodic.jl")


function linprop_RK2_KdV(C2, C3, D, kvec, a, u0, h, tfin)
    N = length(kvec)
    nsteps = Int(round((tfin - a) / h))
    
    # Precompute linear operator
    alpha = D .* kvec.^2 .- C2 .* im .* kvec.^3
    E = exp.(-alpha .* h)  # exponential term
    E2 = exp.(alpha.*(h/2))
    
    # Initialize Fourier modes over time
    uk = zeros(ComplexF64, N, nsteps+1)
    uk[:, 1] = u0  # initial condition in Fourier space
    
    t = range(a, length=nsteps+1, step=h)

    Energy = zeros(nsteps)
    M = zeros(nsteps)
    H = zeros(nsteps)

    x = -π .+ 2π * (0:N-1)/N   # grid with N points, periodic

    
    for n in 1:nsteps
        # U from uk at time n
        u_phys = real(ifft(uk[:, n]) * N)
        
        # Compute u_x:
        # Derivative: i k * uk, then ifft
        ux_phys = real(ifft(im .* kvec .* uk[:, n]) * N)
        
        # Get vk at current time
        vk = fft(u_phys .* ux_phys) / N




        # Midpoint estimate for u_k
        uk_mid = uk[:,n] - (C3*h/2).*vk

        # Compute v_k at Midpoint
        u_phys_mid = real(ifft(uk_mid)*N)
        ux_phys_mid = real(ifft(im.*kvec.*uk_mid)*N)
        vk_mid = fft(u_phys_mid .* ux_phys_mid)/N




        
        # Update Fourier coefficients uk for next step
        uk[:, n+1] = E .* (uk[:, n] - C3 * h .* E2 .* vk_mid)

        Energy[n] = 0.5 * trap_periodic(u_phys.^2, x)
        M[n] = trap_periodic(u_phys, x)
        H[n] = (C2/2) * trap_periodic(ux_phys.^2, x) - (C3/6) * trap_periodic(u_phys.^3, x)   
    end
    
    return t, uk,Energy,M,H
end
