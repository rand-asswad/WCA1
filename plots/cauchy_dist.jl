import Pkg
Pkg.activate("..")
import WCA1

import LinearAlgebra
using Distributions, StatsBase
using WAV, Plots

histo(data, nbins=Int(ceil(sqrt(length(data))))) = fit(Histogram, data; closed=:right, nbins=nbins)
mid_range(bins) = first(bins)+0.5*step(bins):step(bins):last(bins)-0.5*step(bins)

cauchy_median(data) = Cauchy(median(data), median(abs.(data)))
cauchy_interquartile(data) = Cauchy(median(data), quantile(data, 0.75) - quantile(data, 0.25)/2)

function plot_histogram_pdf(data::Vector{T}, d::D, nbins::Int=Int(ceil(sqrt(length(data))));
            lims=WCA1.auto_cut(data; p=0.99)) where {T<:Real, D<:UnivariateDistribution}
    hdata = (lims === nothing) ? data : data[lims[1] .<= data .<= lims[2]]
    h = histo(hdata, nbins)
    mid_bins = mid_range(h.edges[1])
    plot(LinearAlgebra.normalize(h, mode=:pdf), label="normalized histogram")
    plot!(mid_bins, pdf.(d, mid_bins), color="red", label="probability distribution function")
end

function plot_data_cdf(data::Vector{T}, d::D;
            lims=WCA1.auto_cut(data; p=0.99)) where {T<:Real, D<:UnivariateDistribution}
    sorted = ecdf(data).sorted_values
    plot(sorted, range(0, stop=1, length=length(sorted)), label="empirical distribution function", legend=:bottomright)
    plot!(sorted, cdf.(d, sorted), color="red", label="cumulative distribution function")
    if lims !== nothing; xlims!(lims) end
end

x, fs = WAV.wavread("../samples/speech_signal_example.wav")
x = vec(x)

X = WCA1.stft(x, 1000, 900; fs=fs, window=WCA1.hanning)
ν = vec(WCA1.compute_slopes(X))

a, b = WCA1.auto_cut(ν; p=0.99)
d = cauchy_median(ν)

p_pdf = plot_histogram_pdf(ν, d, 150; lims=(a/2, b/2))
p_cdf = plot_data_cdf(ν, d; lims=(a/2, b/2))
plot(p_pdf, p_cdf, layout=grid(1, 2), size=(1300, 500))

savefig("cauchy_dist.png")