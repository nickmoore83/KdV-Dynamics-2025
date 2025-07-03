using FFTW
include("trap_periodic.jl")

# Struct to hold PDE parameters
struct PDE_params
    D::Float64
    C2::Float64
    C3::Float64
end

# Helper: extend positive-k û to full symmetric spectrum
function extend_uhat(uhat_pos)
    K = length(uhat_pos) - 1
    uhat_neg = conj(reverse(uhat_pos[2:end-1]))  # omit k=0 and k=K from conjugate symmetry
    return vcat(uhat_pos, uhat_neg)
end

function KdV_solver(params::PDE_params, a, uhat0_pos, h, tfin)
    D, C2, C3 = params.D, params.C2, params.C3

    K = length(uhat0_pos) - 1        # highest non-negative mode
    N = 2 * K                        # full FFT size
    kvec = [0:K; -K+1:-1]            # full k-vector
    x = -π .+ 2π * (0:N-1) / N       # physical grid

    nsteps = Int(round((tfin - a) / h))
    t = range(a, length=nsteps+1, step=h)

    # Precompute exponential factors
    alpha = D .* kvec.^2 .- C2 .* im .* kvec.^3
    E = exp.(-alpha .* h)
    E2 = exp.(alpha .* (h/2))

    # Initialize uhat array (non-negative modes only)
    uk_pos = zeros(ComplexF64, K+1, nsteps+1)
    uk_pos[:, 1] = uhat0_pos

    Energy = zeros(nsteps+1)
    M = zeros(nsteps+1)
    H = zeros(nsteps+1)

    # Compute initial conserved quantities
    uhat_full = extend_uhat(uk_pos[:, 1])
    u_phys = real(ifft(uhat_full) * N)
    ux_phys = real(ifft(im .* kvec .* uhat_full) * N)
    Energy[1] = 0.5 * trap_periodic(u_phys.^2, x)
    M[1] = trap_periodic(u_phys, x)
    H[1] = (C2 / 2) * trap_periodic(ux_phys.^2, x) - (C3 / 6) * trap_periodic(u_phys.^3, x)

    for n in 1:nsteps
        uhat_full = extend_uhat(uk_pos[:, n])
        u_phys = real(ifft(uhat_full) * N)
        ux_phys = real(ifft(im .* kvec .* uhat_full) * N)
        vk_full = fft(u_phys .* ux_phys) / N

        uk_mid = uhat_full .- (C3 * h / 2) .* vk_full
        u_phys_mid = real(ifft(uk_mid) * N)
        ux_phys_mid = real(ifft(im .* kvec .* uk_mid) * N)
        vk_mid = fft(u_phys_mid .* ux_phys_mid) / N

        uk_next_full = E .* (uhat_full .- C3 * h .* E2 .* vk_mid)
        uk_pos[:, n+1] = uk_next_full[1:K+1]

        u_phys = real(ifft(uk_next_full) * N)
        ux_phys = real(ifft(im .* kvec .* uk_next_full) * N)
        Energy[n+1] = 0.5 * trap_periodic(u_phys.^2, x)
        M[n+1] = trap_periodic(u_phys, x)
        H[n+1] = (C2 / 2) * trap_periodic(ux_phys.^2, x) - (C3 / 6) * trap_periodic(u_phys.^3, x)
    end

    return t, uk_pos, Energy, M, H
end