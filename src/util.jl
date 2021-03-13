
# Norms and distances -------------------------------------------------------------------

# Norms
average_norm(x::AbstractArray, p::Real=1) = norm(x, p) / length(x)
average_norm(x::Number, p::Real=1) = abs(x)

# Distances
distance(x, y, p::Real=1) = norm(x - y, p)
relative_distance(x, y, p::Real=1) = distance(x, y, p) / norm(x, p)
average_distance(x, y, p::Real=1) = average_norm(x - y, p) 
std_distance(x, y) = std(x - y)

# Aliases
norm_L1(x) = norm(x, 1)
norm_L2(x) = norm(x, 2)
norm_L∞(x) = norm(x, Inf)
dist_L1(x, y) = distance(x, y, 1)
dist_L2(x, y) = distance(x, y, 2)
dist_L∞(x, y) = distance(x, y, Inf)
rel_dist_L1(x, y) = relative_distance(x, y, 1)
rel_dist_L2(x, y) = relative_distance(x, y, 2)
rel_dist_L∞(x, y) = relative_distance(x, y, Inf)
avg_dist_L1(x, y) = average_distance(x, y, 1)
avg_dist_L2(x, y) = average_distance(x, y, 2)
avg_dist_L∞(x, y) = average_distance(x, y, Inf)

# Post-processing -----------------------------------------------------------------------

"""
    filter_nonnormal!(s::Signal; value::Real=0)

Replaces non-normal values of `s` with `value`.
"""
function filter_nonnormal!(s::Signal; value::Real=0)
    s[abs.(s) .> 1] .= value
    return s
end

"""
    normalize(s::Signal, ref::AbstractVector; norm::Function=norm_L1, filter_nonnormal::Bool=true)

Normalizes a signal with respect to a reference signal (vector).
"""
function normalize(s::Signal, ref::AbstractVector; norm::Function=norm_L1, filter_nonnormal::Bool=true)
    if length(s) != length(ref); @warn "Signal and ref lengths do not match" end
    s *= norm(ref) / norm(s)
    if filter_nonnormal; filter_nonnormal!(s) end
    return s
end

"""
    resync(s::Signal, ref::AbstractArray; Δt_max::Real, Δt_ignore::Real, distance::Function, verbose::Bool)

Resynchronizes a delayed signal with respect to a reference signal (vector).
"""
function resync(s::Signal, ref::AbstractVector;
                Δt_max::Real=min(0.05, 0.01*duration(s)), Δt_ignore::Real=0.4*Δt_max,
                distance::Function=dist_L∞, verbose::Bool=false)
    if length(s) != length(ref); @warn "Signal and ref lengths do not match" end
    Δmax = round(Int, Δt_max * fs(s))
    ignore = round(Int, Δt_ignore * fs(s))
    Δ = [distance(ref[ignore:length(ref)-ignore-i], s[i+ignore:length(s)-ignore]) for i in 1:Δmax]
    Δmin = findmin(Δ)
    x = zeros(length(s))
    x[1:length(s)-Δmin[2]+1] = s[Δmin[2]:length(s)]
    return verbose ? (Signal(x, fs(s)), Δ) : Signal(x, fs(s))
end