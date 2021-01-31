
import Pkg
Pkg.activate("..")
import WCA1

using Distributions, StatsBase
using Glob, WAV, Plots, StatsPlots
pyplot()

cauchy_median(data) = Cauchy(median(data), median(abs.(data)))
cauchy_interquartile(data) = Cauchy(median(data), quantile(data, 0.75) - quantile(data, 0.25)/2)

# Calculates L∞ distance between two vectors
max_dist(x, y) = maximum(abs.(x - y))

# Calculates point estimate of data with respect to a given distribution
function point_estimate(data::Vector{T}, d::D) where {T<:Real, D<:UnivariateDistribution}
    sorted = ecdf(data).sorted_values
    edf = collect(range(0, stop=1, length=length(sorted)))
    max_dist(edf, cdf.(d, sorted))
end

get_interval(d::UnivariateDistribution, p::Float64=0.95) = quantile(d, 0.5-p/2), quantile(d, 0.5+p/2)
proba_effective(data, a, b) = sum(a .<= data .<= b) / length(data)

tolerance = 0.95

lib = glob("*.wav", "../samples/speech_lib")
cauchy_dist = Dict(
    :median => cauchy_median,
    :interquartile => cauchy_interquartile,
)
pt_estimate = Dict{Symbol, Array{Float64,1}}()
val_percent = Dict{Symbol, Array{Float64,1}}()
for mode in keys(cauchy_dist)
    pt_estimate[mode] = []
    val_percent[mode] = []
end
for file in lib
    data, fs = WAV.wavread(file)
    if ndims(data) > 1; data = size(data, 2) == 1 ? vec(data) : data[:, 1] end
    S = WCA1.stft(data, 1000, 900; fs=fs, window=WCA1.hanning)
    ν = vec(WCA1.compute_slopes(S))
    for mode in keys(cauchy_dist)
        dist = cauchy_dist[mode](ν)
        push!(pt_estimate[mode], point_estimate(ν, dist))
        a, b = get_interval(dist, tolerance)
        push!(val_percent[mode], proba_effective(ν, a, b))
    end
end

# IQR = interquartile range

modes = ["median" "IQR/2"]
function as_2d_array(dict::Dict)
    data = zeros(maximum(length.(values(dict))), length(dict))
    for (i, key) in enumerate(keys(dict)); data[:, i] = dict[key] end
    return data
end

pte = boxplot(modes, as_2d_array(pt_estimate), legend=false)
savefig("cauchy_pt_estimate.png")
vals = boxplot(modes, as_2d_array(val_percent), legend=false)
savefig("cauchy_values_percentage.png")
