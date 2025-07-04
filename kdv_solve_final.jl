using FFTW
include("trap_periodic.jl")

function reconstruct_uhat(uhat_pos)
    K = length(uhat_pos) - 1
    uhat_neg = conj(reverse(uhat_pos[2:end-1]))
    return vcat(uhat_pos, uhat_neg)
end

function kdv_solve(
    pde::PDE_params,
    uhat_0::Vector{ComplexF64},
    dt::Float64,
    tfinal::Float64;
    dealias::Bool = false
)
    K = length(uhat_0) - 1
    N = 2 * K

    x = -π .+ 2π * (0:N-1) / N
    k = [0:K; -K+1:-1]
    ik = im .* k

    if dealias
        Nd = 2 * N
        Kd = Nd ÷ 2
        kd = [0:Kd; -Kd+1:-1]
        filter_rho = exp_filter(kd, Nd)
    else
        Nd = N
        kd = k
    end

    alpha_k = pde.D .* k.^2 .- pde.C2 .* ik.^3
    propagator = exp.(-alpha_k * dt)

    nsteps = Int(round(tfinal / dt))
    tvals = range(0, stop=tfinal, length=nsteps+1)

    uhat_pos = copy(uhat_0)
    uhat_hist = zeros(ComplexF64, K+1, nsteps+1)
    En = zeros(nsteps+1)
    Mom = zeros(nsteps+1)
    Ham = zeros(nsteps+1)
    H2 = zeros(nsteps+1)
    H3 = zeros(nsteps+1)

    # Initial conserved quantities
    uhat_hist[:, 1] = uhat_pos
    uhat_full = reconstruct_uhat(uhat_pos)
    u = real(ifft(uhat_full) * N)
    ux = real(ifft(ik .* uhat_full) * N)

    En[1] = 0.5 * trap_periodic(u.^2, x)
    Mom[1] = trap_periodic(u, x)
    Ham[1] = (pde.C2/2) * trap_periodic(ux.^2, x) - (pde.C3/6) * trap_periodic(u.^3, x)
    H2[1] = (pde.C2/2) * trap_periodic(ux.^2, x)
    H3[1] = -(pde.C3/6) * trap_periodic(u.^3, x)

    for n = 1:nsteps
        uhat_full = reconstruct_uhat(uhat_pos)
        u = real(ifft(uhat_full) * N)
        ux = real(ifft(ik .* uhat_full) * N)

        nonlinear = u .* ux

        if dealias
            nonlinear_pad = zeros(Float64, Nd)
            nonlinear_pad[1:N÷2] .= nonlinear[1:N÷2]
            nonlinear_pad[end-N÷2+1:end] .= nonlinear[end-N÷2+1:end]

            vhat_full = fft(nonlinear_pad) / Nd
            vhat_full .= vhat_full .* filter_rho

            vhat_pos = vhat_full[1:K+1]
        else
            vhat_full = fft(nonlinear) / N
            vhat_pos = vhat_full[1:K+1]
        end

        uhat_pos = propagator[1:K+1] .* (uhat_pos .- pde.C3 * dt .* vhat_pos)
        uhat_hist[:, n+1] = uhat_pos

        uhat_full = reconstruct_uhat(uhat_pos)
        u = real(ifft(uhat_full) * N)
        ux = real(ifft(ik .* uhat_full) * N)

        En[n+1] = 0.5 * trap_periodic(u.^2, x)
        Mom[n+1] = trap_periodic(u, x)
        Ham[n+1] = (pde.C2/2) * trap_periodic(ux.^2, x) - (pde.C3/6) * trap_periodic(u.^3, x)
        H2[n+1] = (pde.C2/2) * trap_periodic(ux.^2, x)
        H3[n+1] = -(pde.C3/6) * trap_periodic(u.^3, x)
    end

    return tvals, uhat_hist, En, Mom, Ham, H2, H3
end
