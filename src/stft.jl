# We will use the already defined functions in DSP, 
# adapting the spectrogram and stft functions.
# 
# First of all we define a container type for our stft.

const FloatRange{T} = StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T}}

struct STFT{T, F<:Union{Frequencies, AbstractRange}} 
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
Base.size(m::STFT, etc...) = size(vals(m), etc...)
Base.getindex(m::STFT, etc...) = getindex(vals(m), etc...)
Base.extrema(m::STFT) = extrema(vals(m))

# Functions to recover informations about the stft from the 
# quantities we saved inside the type
fs(s::STFT) = round(Int, width(s)/(2*first(time(s))))
noverlap(s::STFT) = round(Int, width(s) - fs(s)*step(time(s)))

# We can now define the stft function, which is just a 
# wrapper around DSP.stft.

function stft(s::AbstractVector{T}, n::Int=length(s)>>3, 
					noverlap::Int=n>>1;
                    onesided::Bool=eltype(s)<:Real, nfft::Int=DSP.nextfastfft(n), 
                    fs::Real=1, window::Union{Function,AbstractVector,Nothing}=nothing) where T

    out = DSP.stft(s, n, noverlap; onesided=onesided, fs=fs, window=window)
    STFT(out,
        onesided ? rfftfreq(nfft, fs) : fftfreq(nfft, fs),
        (n/2 : n-noverlap : (size(out,2)-1)*(n-noverlap)+n/2) / fs,
        n,
        length(s),
        window)
end

function show_stft(m::STFT, pre = x->x) 
	# Function to plot an STFT
    T = map(x->round(x, digits = 2), range(first(time(m)), last(time(m)), length= 4))
    F = map(x->round(Int, x), range(first(freq(m)), last(freq(m)), length= 5))
    heatmap(pre.(abs.(vals(m))), xaxis = ("Time (s)"), yaxis = ("Frequencies (Hz)"),
        xticks = (range(1,size(m,2), length= 4), T),
        yticks = (range(1,size(m,1), length= 5), F))
end

# We now define the inverse STFT (using overlap-add method).

function istft(X::Matrix{T}, siglength::Int,
                nperseg::Int=onesided ? (size(X, 1) - 1) << 1 : size(X, 1),
                noverlap::Int=nperseg >> 1; onesided::Bool=true,
                window::Union{Function,AbstractVector,Nothing}=nothing) where T
    # check that sizes make sense
    hopsize = nperseg - noverlap
    window = (window == nothing) ? DSP.rect : window
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

function istft(X::STFT)
    istft(vals(X), or_length(X), width(X), noverlap(X);
        onesided=!any(freq(X) .< 0), window=window(X))
end

function show_istft(m::STFT)
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