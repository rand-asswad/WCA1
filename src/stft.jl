
# Type definition -----------------------------------------------------------------------

"""
    Window=Union{Function,AbstractVector,Nothing}

Window type for STFT methods.
"""
const Window=Union{Function,AbstractVector,Nothing}

"""
    STFT{T<:Complex} <: AbstractMatrix{T}

Short-Time Fourier Transform object type.

### Fields
- `data::Matrix{Complex}`: stft matrix STFT[ω,τ].
- `freq::Union{Frequencies, AbstractRange}`: frequency bins vector.
- `time::FloatRange{Real}`: time samples vector.
- `width::Int`: the number of samples per window.
- `sig_length::Int`: the original signal length.
- `window::Window`: callable window function or window values vector.
"""
struct STFT{T<:Real} <: AbstractMatrix{Complex{T}}
    data::Matrix{Complex{T}}
    freq::FloatRange
    time::FloatRange
    width::Int
    sig_length::Int
    window::Window
end

data(S::STFT) = S.data
freq(S::STFT) = S.freq
Libc.time(S::STFT) = S.time
width(S::STFT) = S.width
sig_length(S::STFT) = S.sig_length
window(S::STFT) = S.window

fs(S::STFT) = round(Int, width(S) / (2 * first(time(S))))
noverlap(S::STFT) = round(Int, width(S) - fs(S) * step(time(S)))

# Base functions for interface inheritance
Base.size(S::STFT, args...) = size(data(S), args...)
Base.getindex(S::STFT, args...) = getindex(data(S), args...)
Base.setindex!(S::STFT, v, args...) = setindex!(data(S), v, args...)
Base.similar(S::STFT{T}) where {T} = similar(S, T)
Base.similar(S::STFT, ::Type{T}) where {T<:Real} = STFT(similar(data(S), Complex{T}), freq(S), time(S), width(S), sig_length(S), window(S)) 
Base.showarg(io::IO, S::STFT, toplevel) = toplevel && print(io, "STFT{$(eltype(S))} with fft width $(width)")

# STFT implementation -------------------------------------------------------------------

"""
    stft(s, nperseg=div(length(s),8), noverlap=div(nperseg,2); onesided=eltype(s)<:Real, nfft=DSP.nextfastfft(nperseg), fs=1, window=hanning)

Wrapper around `DSP.Periodograms.stft` function that computes the STFT of a signal
using the overlap-add method.
Returns an `STFT` object.

# Extended help

### Arguments
- `s::AbstractVector{T<:Real}`: the input signal array.
- `nperseg::Int=div(length(s), 64)`: the number of samples per window.
- `noverlap::Int=div(nperseg, 1)`: the number of overlapping samples.

### Keyword Arguments
- `onesided::Bool=eltype(s)<:Real`: if `true`, return a one-sided spectrum
  for real input signal. If `false` return a two-sided spectrum.
- `nfft::Int=DSP.nextfastfft(nperseg)`: the number of samples to use
  for the Fourier Transform. if `n` < `nfft`, the window is padded with zeros.
- `fs::Real=1`: the sample rate of the input signal.
- `window::Window=hanning`: the window function or vector to use.
  If `nothing`, a rectangular window is used.

"""
function stft(s::AbstractVector{T}, nperseg::Int=length(s)>>6, noverlap::Int=nperseg>>1;
                onesided::Bool=eltype(s)<:Real, nfft::Int=DSP.nextfastfft(nperseg), 
                fs::Real=1, window::Window=hanning) where T
    hopsize = nperseg - noverlap
    out = DSP.stft(s, nperseg, noverlap; onesided=onesided, fs=fs, window=window)
    STFT(out,
        onesided ? rfftfreq(nfft, fs) : fftfreq(nfft, fs),
        (nperseg/2 : hopsize : (size(out,2)-1)*hopsize + nperseg/2) / fs,
        nperseg,
        length(s),
        window)
end

"""
    stft(s::Signal, nperseg=div(length(s),8), noverlap=div(nperseg,2); onesided=eltype(s)<:Real, nfft=DSP.nextfastfft(nperseg), window=hanning)

Wrapper around `DSP.Periodograms.stft` function that computes the STFT of a `Signal` object
using the overlap-add method.
Returns an `STFT` object.
"""
function stft(s::Signal, nperseg::Int=length(s)>>6, noverlap::Int=nperseg>>1;
                onesided::Bool=eltype(s)<:Real, nfft::Int=DSP.nextfastfft(nperseg), 
                window::Window=hanning)
    stft(data(s), nperseg, noverlap; onesided=onesided, nfft=nfft, fs=fs(s), window=window)
end

"""
    istft(S::STFT)

Wrapper around `istft` that takes `STFT` object as input to calculate the
inverse Short-Time Fourier Transform.
Returns `Signal` object.
"""
function istft(X::STFT)
    x = istft(data(X), sig_length(X), width(X), noverlap(X);
            onesided=!any(freq(X) .< 0), window=window(X))
    Signal(x, fs(X))
end

"""
    istft(S, siglength, nperseg=onesided ? div(size(X,1)-1,2) : size(X,1), noverlap=div(nperseg,1); onesided=eltype(s)<:Real, nfft=DSP.nextfastfft(nperseg), fs=1, window=hanning)

Calculates the inverse Short-Time Fourier Transform (iSTFT).
The function implements the least-squares minimization method to find a signal
that minimizes the error between its STFT and the given STFT.

# Extended help

### Arguments
- `S::AbstractMatrix{Complex}`: the signal STFT.
- `siglength::Int`: the number of samples in original signal.
- `nperseg::Int=onesided ? 2 * (size(X, 1) - 1) : size(X, 1)`:
   the number of samples per window.
- `noverlap::Int=div(nperseg, 2)`: the number of overlapping samples.

### Keyword Arguments
- `onesided::Bool=eltype(s)<:Real`: if `true`, return a one-sided spectrum for real input signal.
  If `false` return a two-sided spectrum.
- `window::Window=hanning`: the window function or vector to use.
  If `nothing`, a rectangular window is used.
"""
function istft(X::AbstractMatrix{T}, siglength::Int,
                nperseg::Int=onesided ? (size(X, 1) - 1) << 1 : size(X, 1),
                noverlap::Int=nperseg >> 1; onesided::Bool=true,
                window::Window=hanning) where T<:Complex
    # check that sizes make sense
    hopsize = nperseg - noverlap
    window = (window === nothing) ? DSP.rect : window
    # if window is a vector verify size
    # if window is function generate vector
    w = window(nperseg)
    w2 = w .* w # for optimisation purposes
    x = zeros(siglength)
    scale = zeros(siglength)

    # apply inverse fft with respect to frequency dimension
    y = onesided ? irfft(X, nperseg, 1) : real(ifft(X, 1))

    # overlap-add windowed inverse fft
    for (j, i) in enumerate(1:hopsize:siglength-nperseg)
        x[i:i+nperseg-1] += y[:, j] .* w
        scale[i:i+nperseg-1] += w2
    end
    
    # return scaled signal
    ifelse.(scale .> 1e-10, x ./ scale, x)
end

# Plotting methods ----------------------------------------------------------------------

@recipe function plot_stft(S::STFT; mode=:spectrogram, yscale=:identity, axislabels=true)
    seriestype := :heatmap
    xguide --> (axislabels ? "Time (s)" : "")
    yguide --> (axislabels ? "Frequency (Hz)" : "")
    if mode == :spectrogram
        func = abs
    elseif mode == :phase
        func = angle
    else
        @error("Unsupported mode $(mode). Choose from (:spectrogram, :phase)")
    end
    if yscale in [:ln, :log2, :log10]
        non_negative = freq(S) .> 0
        return time(S), freq(S)[non_negative], func.(S[non_negative,:])
    else
        return time(S), freq(S), func.(S)
    end
end

"""
    plot(S::STFT; mode::Symbol=:spectrogram, yscale::Symbol=:identity, axislabels::Bool=true, kwargs...)

Plot recipe for STFT object. Plots spectrogram or phase heatmap.

`mode` is either `:spectrogram` (default) or `:phase`
If `axislabels` is true, the default axis labels are displayed.
Can be overriden using the appropriate keyword arguments.
The `yscale` parameter allows plotting logarithmic frequency scale.
"""
plot(::STFT)