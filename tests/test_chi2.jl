using Distributions
using Glob
using WAV
import DSP

"""
    chi2_test(data::Vector{Real}, α::Real=0.05; df::Int=3)

Verifies that a given vector of value is a random distribution
using the Χ² test with a significance level of α (default 5%)
with `df` degrees of freedom (default `df=3` for normal distributed values).

## Sources
- [Rosetta Code](https://rosettacode.org/wiki/Verify_distribution_uniformity/Chi-squared_test#Julia)
- [Wikipedia article](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Other_distributions)
"""
function chi2_test(data::Vector{T}, α::Real=0.05; df::Int=3) where T <: Real
    if ! (0 ≤ α ≤ 1); error("α must be in [0, 1]") end
    m = mean(data)
    chisqval = sum((x - m)^2 for x in data) / m
    pval = ccdf(Chisq(df), chisqval)
    return pval > α
end

function test_chirpiness_normality(samples_dir::String="./", tolerance::Real=0.05)
    for file in glob("*.wav", samples_dir)
        data, fs = wavread(file)
        if ndims(data) > 1
            data = size(data, 2) == 1 ? reshape(data, size(data)) : data[:, 1]
        end
        S = stft(data, 1000, 900; onesided=onesided, fs=fs, window=DSP.hanning)
        V = compute_chirpiness(S)
        is_normal = chi2_test(V, tolerance)
        if !is_normal
            println("Chirpiness in '$(file)' is not normally distrubuted")
            return false
        end
    end
    return true
end


@testset "Chi-squared test on speech signal chirpiness" begin
    for tol in [0.05, 0.01, 1e-5, 1e-10]
        @test test_chirpiness_normality("samples/speech_lib", tol)
    end
end