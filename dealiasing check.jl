using FFTW, Plots, Printf

# Exponential spectral filter for dealiasing
function exp_filter(k::Vector{Int}, N::Int; alpha=36, m=36)
    return exp.(-alpha * (abs.(k) ./ N).^m)
end

# Trapezoidal integration for L² norm
function trap_periodic(fvals, x)
    h = x[2] - x[1]
    return h * sum(fvals)
end

function compute_L2_error(u_num, u_exact, x)
    sqrt(trap_periodic((u_num .- u_exact).^2, x))
end

# Parameters
N = 256
N2 = 2N
L = 2π

# Grids and wavenumbers
x = L * (0:N-1) / N
x2 = L * (0:N2-1) / N2

k = [0:N÷2; -N÷2+1:-1]

k2 = zeros(Int, N2)
k2[1:(N÷2+1)] = k[1:(N÷2+1)]
k2[(N2 - N÷2 + 2):N2] = k[(N÷2 + 2):N]

#k2 = [0:N2÷2; -N2÷2+1:-1]

ik = im .* k
ik2 = im .* k2

# Initial condition
u = sin.(x)
u2 = sin.(x2)
u_hat = fft(u) / N

# Filter for N2 grid
filter_rho_N2 = exp_filter(k2, N2)

# Filter for N grid
filter_rho_N = exp_filter(k, N)

### --- Method 1: 2N padding + dealiasing ---
u_hat_1 = fft(u2) / N2
u1 = real(ifft(u_hat_1) * N2)
u_x1 = real(ifft(ik2 .* u_hat_1) * N2)

nonlin1 = u1 .* u_x1
v_hat1 = fft(nonlin1) / N2

v_hat1_filtered = v_hat1 .* filter_rho_N2
uu_x1_filtered = real(ifft(v_hat1_filtered) * N2)

### --- Method 2: N grid direct ---
u2 = real(ifft(u_hat)) * N
u_x2 = real(ifft(ik .* u_hat)) * N
nonlin2 = u2 .* u_x2
uu_x_direct = nonlin2

### --- Method 3: N grid + dealiasing ---
nonlin3 = u2 .* u_x2
v_hat3 = fft(nonlin3) / N
v_hat3_filtered = v_hat3 .* filter_rho_N
uu_x3_filtered = real(ifft(v_hat3_filtered) * N)

### --- Exact solution ---
u2_exact = sin.(x2)
u2_x_exact = cos.(x2)
uu_x2_exact = u2_exact .* u2_x_exact

u_exact = sin.(x)
u_x_exact = cos.(x)
uu_x_exact = u_exact .* u_x_exact

### --- L2 error ---
L2_error_dealias1 = compute_L2_error(uu_x1_filtered, uu_x2_exact, x2)
L2_error_direct = compute_L2_error(uu_x_direct, uu_x_exact, x)
L2_error_dealias2 = compute_L2_error(uu_x3_filtered, uu_x_exact, x)

@printf("L2 error with 2N padding + dealiasing = %.5e\n", L2_error_dealias1)
@printf("L2 error without padding (direct N grid) = %.5e\n", L2_error_direct)
@printf("L2 error with N + dealiasing = %.5e\n", L2_error_dealias2)

### --- Plot ---
p1 = plot(x2, uu_x2_exact, label="Exact uu_x (fine grid)", lw=2, color=:black)
plot!(p1, x2, uu_x1_filtered, label="2N padded + filtered", lw=2, ls=:dash)
xlabel!(p1, "x")
ylabel!(p1, "uu_x")
title!(p1, "Dealiasing with 2N padding")

p2 = plot(x, uu_x_exact, label="Exact uu_x (N grid)", lw=2, color=:black)
plot!(p2, x, uu_x_direct, label="Direct N grid", lw=2, ls=:dot)
xlabel!(p2, "x")
ylabel!(p2, "uu_x")
title!(p2, "Without dealiasing")

p3 = plot(x, uu_x_exact, label="Exact uu_x (N grid)", lw=2, color=:black)
plot!(p3, x, uu_x3_filtered, label="N grid + filtered", lw=2, ls=:dash)
xlabel!(p3, "x")
ylabel!(p3, "uu_x")
title!(p3, "Dealiasing on N grid")

plot(p1, p2, p3, layout=(3,1), size=(800,900))
