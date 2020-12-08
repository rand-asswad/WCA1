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

# synthetic signals
fs = 16000.
duration = 2
n = round(Int, duration * fs)

lc = [ n/7 <= t <= 8*n/14 ? sin(2000*2*π*t/fs+1000*t/fs*2*π*t/fs) : 0. for t in 1:n ]
nlc = [ n/7 <= t <= 8*n/14 ? sin(3000*2*π*t/fs+2pi*150*sin(2*pi*t/fs)) : 0. for t in 1:n ]

x1 = [ n/7 <= t <= 2.5*n/7 ? sin(2000*2*π*t/fs+1000*t/fs*2*π*t/fs) : 0. for t in 1:n ]
x2= [ 3*n/7 <= t <= 4.5*n/7 ? sin(2000*2*π*t/fs+1000*t/fs*2*π*t/fs) : 0. for t in 1:n ]
irc = x1+x2

x1 = [ n/7 <= t <= 6*n/14 ? sin(2000*2*π*t/fs + 1000*t/fs*2*π*t/fs)  : 0. for t in 1:n ]
x2 = [ n/7 <= t <= 6*n/14 ? sin(6000*2*π*t/fs - 1000*t/fs*2*π*t/fs)  : 0. for t in 1:n ]
isc = x1 + x2

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