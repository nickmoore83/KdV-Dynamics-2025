using FFTW

function kdv_solve(
    pde::PDE_params,
    uhat_0::Vector{ComplexF64},
    dt::Float64,
    tfinal::Float64;
    dealias::Bool = false
)
    K = length(uhat_0) - 1
    N = 2 * K

    # Define base wave numbers
    k = [0:K; -K+1:-1]
    ik = im .* k

    # Define dealias grid size and k if needed
    if dealias
        Kd = 2 * K      # doubled cutoff for dealiasing
        Nd = 2 * N      # doubled FFT size for dealiasing
        kd = [0:Kd; -Kd+1:-1]
        filter_rho = exp_filter(kd, Nd)
    else
        Kd = K
        Nd = N
    end

    # Linear propagator on original k (K, N)
    alpha_k = pde.D .* k.^2 .- pde.C2 .* ik.^3
    propagator = exp.(-alpha_k * dt)

    nsteps = Int(round(tfinal / dt))
    tvals = range(0, stop=tfinal, length=nsteps+1)

    uhat_pos = copy(uhat_0)
    uhat_hist = zeros(ComplexF64, K+1, nsteps+1)
    En = zeros(nsteps+1)
    Mom = zeros(nsteps+1)
    Ham = zeros(nsteps+1)

    # Initial quantities (original grid)
    uhat_hist[:, 1] = uhat_pos
    uhat_full = reconstruct_uhat(uhat_pos)
    u = real(ifft(uhat_full) * N)
    ux = real(ifft(ik .* uhat_full) * N)

    En[1] = 0.5 * trap_periodic(u.^2, x)
    Mom[1] = trap_periodic(u, x)
    Ham[1] = (pde.C2/2) * trap_periodic(ux.^2, x) - (pde.C3/6) * trap_periodic(u.^3, x)

    # Time stepping loop
    for n = 1:nsteps
        # Reconstruct full spectrum on original grid
        uhat_full = reconstruct_uhat(uhat_pos)
        u = real(ifft(uhat_full) * N)
        ux = real(ifft(ik .* uhat_full) * N)

        nonlinear = u .* ux

        if dealias
            nonlinear_pad = zeros(Float64, Nd)
            nonlinear_pad[1:Int(N/2)] = nonlinear[1:Int(N/2)]
            nonlinear_pad[end-Int(N/2)+1:end] = nonlinear[end-Int(N/2)+1:end]

            vhat_full = fft(nonlinear_pad) / Nd

            # Apply filter
            vhat_full .= vhat_full .* filter_rho

            # Truncate back to original size
            vhat_full_trunc = vcat(vhat_full[1:K+1], vhat_full[end-K+1:end])
            vhat_pos = vhat_full_trunc[1:K+1]

        else
            vhat_full = fft(nonlinear) / N
            vhat_pos = vhat_full[1:K+1]
        end

        uhat_pos = (uhat_pos .- pde.C3 * dt .* vhat_pos) .* propagator[1:K+1]
        uhat_hist[:, n+1] = uhat_pos

        # Update conserved quantities
        uhat_full = reconstruct_uhat(uhat_pos)
        u = real(ifft(uhat_full) * N)
        ux = real(ifft(ik .* uhat_full) * N)

        En[n+1] = 0.5 * trap_periodic(u.^2, x)
        Mom[n+1] = trap_periodic(u, x)
        Ham[n+1] = (pde.C2/2) * trap_periodic(ux.^2, x) - (pde.C3/6) * trap_periodic(u.^3, x)
    end

    return tvals, uhat_hist, En, Mom, Ham
end
