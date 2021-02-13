# Input sound

#x, fs = wavread("../../samples/speech_signal_example.wav")
x, fs = WCA1.wavread("../../samples/speech_lib/calo.wav")
x = vec(x)


## Short time Fourier transform
S = stft(x, 500, 450; fs=fs, window=hanning)


## Lift
L = lift(S; N=100)

## WC evolution

χ = 20

α = 500
β = 1
γ = 200

τ = χ * step(time(L))

b = 1

mynormalize(f) = range(first(f)/last(f), 1.0; step=step(f)/last(f))

k = Kern(mynormalize(freq(L)), slopes(L), KernParams(τ, b, 1e-6));

W = wc_delay(L, α, β, γ, K=k,τdx = χ) |> project

## Save results

try
    mkpath("speech-signal-results")
catch
end

cd("speech-signal-results")

save_result(S, W, α, β, γ, χ; rate=fs)

cd("..")
