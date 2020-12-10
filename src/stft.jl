
# Type definition -----------------------------------------------------------------------

const FloatRange{T} = StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T}}

"""
    STFT

Short-Time Fourier Transform object type.

### Fields
- `stft::Matrix{Complex}`: stft matrix STFT[ω,τ].
- `freq::F<:Union{Frequencies, AbstractRange}`: frequency bins vector.
- `time::FloatRange{Real}`: time samples vector.
- `width::Int`: the number of samples per window.
- `sig_length::Int`: the original signal length.
- `window::Union{Function,AbstractVector,Nothing}`: callable window function
  or window values vector.
"""
struct STFT{T<:Complex, F<:Union{Frequencies, AbstractRange}} 
    stft::Matrix{T}
    freq::F
    time::FloatRange{Float64}
    width::Int
    sig_length::Int
    window::Union{Function,AbstractVector,Nothing}
end

Libc.time(s::STFT) = s.time
freq(s::STFT) = s.freq
vals(s::STFT) = s.stft
width(s::STFT) = s.width
or_length(s::STFT) = s.sig_length
window(s::STFT) = s.window

Base.eltype(m::STFT) = eltype(vals(m))
Base.size(m::STFT, args...) = size(vals(m), args...)
Base.getindex(m::STFT, args...) = getindex(vals(m), args...)
Base.extrema(m::STFT) = extrema(vals(m))

fs(s::STFT) = round(Int, width(s)/(2*first(time(s))))
noverlap(s::STFT) = round(Int, width(s) - fs(s)*step(time(s)))

# STFT implementation -------------------------------------------------------------------

"""
    stft(s, nperseg=div(length(s),8), noverlap=div(nperseg,2); onesided=eltype(s)<:Real, nfft=DSP.nextfastfft(nperseg), fs=1, window=nothing)

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
- `window::Union{Function,AbstractVector,Nothing}=nothing`: the window function to use.
  If `nothing`, a rectangular window is used.

"""
function stft(s::AbstractVector{T}, nperseg::Int=length(s)>>6, noverlap::Int=nperseg>>1;
                onesided::Bool=eltype(s)<:Real, nfft::Int=DSP.nextfastfft(nperseg), 
                fs::Real=1, window::Union{Function,AbstractVector,Nothing}=nothing) where T
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
    istft(S, siglength, nperseg=onesided ? div(size(X,1)-1,2) : size(X,1), noverlap=div(nperseg,1); onesided=eltype(s)<:Real, nfft=DSP.nextfastfft(nperseg), fs=1, window=nothing)

Calculates the inverse Short-Time Fourier Transform (iSTFT).
The function implements the least-squares minimization method to find a signal
that minimizes the error between its STFT and the given STFT.

# Extended help

### Arguments
- `S::Matrix{Complex}`: the signal STFT.
- `siglength::Int`: the number of samples in original signal.
- `nperseg::Int=onesided ? (size(X, 1) - 1) << 1 : size(X, 1)`:
   the number of samples per window.
- `noverlap::Int=div(nperseg, 1)`: the number of overlapping samples.

### Keyword Arguments
- `onesided::Bool=eltype(s)<:Real`: if `true`, return a one-sided spectrum for real input signal.
  If `false` return a two-sided spectrum.
- `nfft::Int=DSP.nextfastfft(n)`: the number of samples to use for the Fourier Transform.
  If `n` < `nfft`, the window is padded with zeros.
- `fs::Real=1`: the sample rate of the input signal.
- `window::Union{Function,AbstractVector,Nothing}=nothing`: the window function to use.
  If `nothing`, a rectangular window is used.
"""
function istft(X::Matrix{T}, siglength::Int,
                nperseg::Int=onesided ? (size(X, 1) - 1) << 1 : size(X, 1),
                noverlap::Int=nperseg >> 1; onesided::Bool=true,
                window::Union{Function,AbstractVector,Nothing}=nothing) where T<:Complex
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

"""
    istft(S::STFT)

Wrapper around `istft` that takes `STFT` object as input to calculate the
inverse Short-Time Fourier Transform.
"""
function istft(X::STFT)
    istft(vals(X), or_length(X), width(X), noverlap(X);
            onesided=!any(freq(X) .< 0), window=window(X))
end

# Plotting methods ----------------------------------------------------------------------

function show_stft(m::STFT, pre = x->x, args...; kw...)
	# Function to plot an STFT
    T = map(x->round(x, digits = 2), range(first(time(m)), last(time(m)), length= 4))
    F = map(x->round(Int, x), range(first(freq(m)), last(freq(m)), length= 5))
    heatmap(pre.(abs.(vals(m))), xaxis = ("Time (s)"), yaxis = ("Frequencies (Hz)"),
        xticks = (range(1,size(m,2), length= 4), T),
        yticks = (range(1,size(m,1), length= 5), F)
    )
end


function show_istft(m::STFT, args...; kw...)
    s = istft(m)
    max_amp = maximum( abs.(extrema(s)) )
    T = map(x->round(x, digits = 2), range(first(time(m)), last(time(m)), length= 4))
    A = map(x->round(x, digits = 3), range(minimum(s), maximum(s), length= 5))
    plot(s, xaxis = ("Time (s)"), yaxis = ("Amplitude"),
        xticks = (range(1, length(s), length= 4), T),
        yticks = (range(-max_amp, max_amp, length= 5), range(-1, 1, length= 5)),
        ylims = (-max_amp, max_amp),
        label = ""
    )
end