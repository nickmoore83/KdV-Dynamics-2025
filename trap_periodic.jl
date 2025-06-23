function trap_periodic(fvals, x)
    h = x[2] - x[1]  # assume uniform spacing
    return h * sum(fvals)
end