import DSP
using DSP.Windows

rel_dist(x, y) = sum(abs.(x - y)) / sum(abs.(x))
max_dist(x, y) = maximum(abs.(x - y))

function test_stft(x::AbstractArray{T}, fs, nperseg, overlap;
            dist::Function=rel_dist, tolerance=1e-3) where T
    for width in nperseg, ratio in overlap, win in [hanning, nothing], onesided in [true, false]
        noverlap = round(Int, ratio * width)
        X = stft(x, width, noverlap; onesided=onesided, fs=fs, window=win)
        y = istft(X)

        err = dist(x, y)
        if err > tolerance
            println("STFT-ISTFT failed width")
            println("\twindow width:\t", width)
            println("\toverlapping n:\t", noverlap)
            println("\twindow:\t",win)
            println("\tonesided is:\t",onesided)
            println("Error value:\t", err)
            return false
        end
    end
    return true
end

# generate synthetic signals
fs = 16000
duration = 2

lc = linear_chirp(fs, duration)
nlc = nonlinear_chirp(fs, duration)
irc = interrupted_chirp(fs, duration)
isc = intersecting_chirp(fs, duration)

# test parameters
nperseg = [256, 512]
overlap = [1/2, 2/3, 3/4, 4/5]

@testset "Synthetic signals STFT-ISTFT test" begin
    for signal in [lc, nlc, irc, isc], tol in [1e-5, 1e-10, 1e-15]
        @test test_stft(signal, fs, nperseg, overlap; dist=rel_dist, tolerance=tol)
    end
end

# recorded speech signal
import WAV
ss, fs = WAV.wavread("../samples/speech_signal_example.wav")
ss = reshape(ss, length(ss))

nperseg = [256, 512, 1024, 2048]
overlap = [1/2, 2/3, 3/4, 4/5]

@testset "Speech signal STFT-ISTFT test" begin
    for tol in [1e-3, 1e-4, 1e-5]
        @test test_stft(ss, fs, nperseg, overlap; dist=rel_dist, tolerance=tol)
    end
end