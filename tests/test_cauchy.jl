using Distributions, StatsBase, HypothesisTests, LinearAlgebra
using Glob, WAV, Plots

function ks_pt_estimate(data::Vector{T}, d::UnivariateDistribution) where T<:Real
    n = length(data)
    cdf_vals = cdf.(d, ecdf(data).sorted_values)
    pos = maximum((1:n)/n - cdf_vals)
    neg = maximum(cdf_vals - (0:n-1)/n)
    return max(pos, neg)
end

function ks_test(data::Vector{T}, d::UnivariateDistribution; α=0.95) where T<:Real
    ks_stat = sqrt(length(data)) * ks_pt_estimate(data, d)
    return !(ks_stat > cdf(Kolmogorov(), α))
end

cauchy_median(data) = Cauchy(median(data), median(abs.(data)))
cauchy_iqr(data) = Cauchy(median(data), percentile(data, 75) - percentile(data, 25)/2)
cauchy_mle(data) = fit(Cauchy, data)

function ks_test_stft(x::Vector{T}, fs, nperseg, overlap) where T<:Real
    count = 0
    #success = 0
    for width in nperseg, ratio in overlap, win in [WCA1.hanning, nothing]
        noverlap = round(Int, ratio * width)
        X = WCA1.stft(x, width, noverlap; fs=fs, window=win)
        ν = vec(WCA1.brdcast_chirpiness(X))
        for d in [cauchy_median(ν), cauchy_iqr(ν)]
            count += 1
            if ks_test(ν, d; α=0.95); return true end#success += 1 end
        end
    end
    return false
    #println("Success rate: $(success)/$(count)")
    #return success > 0
end

function ks_test_lib(samples_dir::String, nperseg, overlap)
    lib = glob("*.wav", samples_dir)
    success = 0
    for file in lib
        data, fs = wavread(file)
        if ndims(data) > 1; data = size(data, 2) == 1 ? reshape(data, length(data)) : data[:, 1] end
        success += Int(ks_test_stft(data, fs, nperseg, overlap))
    end
    println("Success rate: $(success)/$(length(lib))")
    return success == length(lib)
end

# test parameters
nperseg = [250, 500, 1000, 1500, 2000, 5000, 10000]
overlap = [1/2, 2/3, 3/4, 4/5, 9/10]

# test results

# testing on the recorded sample of my voice
```
julia> ks_test_stft(x, fs, nperseg, overlap)
Success rate: 0/120
false
```

# testing on the speech library
```
julia> ks_test_lib("samples/speech_lib", nperseg, overlap)
Success rate: 0/49
false
```