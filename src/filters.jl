
"""
    stft_filter(S::Matrix{Complex}, kernel=Kernel.gaussian(3); mode="spectrum")

Applies filter using `ImageFiltering.imfilter` on STFT matrix using given kernel.
Returns Matrix{Complex}.
Provides 3 modes of computation:
- **spectrum:** filters the spectrum and leaves the phase unchanged (default mode).
- **polar:** applies the filter on the spectrum and on the phase separately.
- **cartesian:** applies the filter on the real part and imaginary part separately.
Throws warning otherwise and returns the input STFT.
"""
function stft_filter(S::Matrix{T}, kernel=Kernel.gaussian(3); mode::String="spectrum") where {T<:Complex}
    if mode == "cartesian"
        Z = complex.(imfilter(real(S), kernel), imfilter(imag(S), kernel))
    elseif mode == "spectrum"
        Z = imfilter(abs.(S), kernel) .* exp.(im * angle.(S))
    elseif mode == "polar"
        Z = imfilter(abs.(S), kernel) .* exp.(im * imfilter(angle.(S), kernel))
    else
        @warn "Undefined mode `$(mode)`"
        Z = copy(S)
    end
end

"""
    stft_filter(S::STFT, kernel=Kernel.gaussian(3); mode="spectrum")

Applies filter using `ImageFiltering.imfilter` on STFT matrix using given kernel.
Returns STFT object.
"""
function stft_filter(S::STFT, kernel=Kernel.gaussian(3); mode="spectrum")
    Z = stft_filter(S.stft, kernel; mode=mode)
    STFT(Z, S.freq, S.time, S.width, S.sig_length, S.window)
end


"""
    signal_filter(s, fs=1, kernel=Kernel.gaussian(3); mode="spectrum", nperseg=1000, noverlap=round(Int, nperseg*0.9))

Calculates STFT of given signal and applies given filter, returns the inverse STFT
of the filtered spectrum.
"""
function signal_filter(s::AbstractVector{T}, fs::Real=1, kernel=Kernel.gaussian(3);
                        mode="spectrum", nperseg::Int=1000,
                        noverlap::Int=round(Int, nperseg*0.9)) where {T<:Real}
    X = stft(s, nperseg, noverlap; fs=fs, window=WCA1.hanning)
    Y = stft_filter(X, kernel; mode=mode)
    istft(Y)
end