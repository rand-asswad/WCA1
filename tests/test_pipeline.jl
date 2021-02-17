using WCA1, Plots

function cut_signal(x::Signal, t0=0.0, Δt=0.1)
    n0 = round(Int, t0 * x.fs)
    n1 = n0 + round(Int, Δt * x.fs) - 1
    y = copy(data(x))
    y[n0:n1] .= 0
    Signal(y, x.fs)
end

function noise_signal(x::Signal, noise=:salt_pepper)
    y = copy(data(x))
    # add noise
    Signal(y, fs(x))
end

function reconstruct_signal(x::Signal; α=1000, β=1, γ=500, fft_width=500, overlap=9//10, liftsamples=100)
    noverlap = round(Int, fft_width * overlap)
    X = stft(x, fft_width, noverlap; window=WCA1.hanning)
    L = lift(X, N=liftsamples)
    W = wc_delay(L, α, β, γ)
    istft(project(W))
end

function normalize_signal(x::Signal, ref::Union{Signal,T}) where {T<:Real}
    if typeof(ref) <: Real && (ref > 0)
        return Signal(x.data / ref, x.fs)
    else
        ratio = sum(abs.(ref.data)) / sum(abs.(x.data))
        return Signal(x.data * ratio, x.fs)
    end
end

function test_cut(x::Signal; t0=0.0, Δt=0.1, α=1000, β=1, γ=500,
        fft_width=500, overlap=9//10, liftsamples=100)
    params = "dt=$(round(Δt, digits=3))_a=$(α)_b=$(β)_c=$(γ)"
    y = cut_signal(x, t0, Δt)
    #w = reconstruct_signal(y; α=α, β=β, γ=γ)
    Y = stft(y, fft_width, round(Int, fft_width * overlap); window=WCA1.hanning)
    L = lift(Y, N=liftsamples)
    W = wc_delay(L, α, β, γ)
    Wp = project(W)
    w = istft(Wp)
    w1 = normalize_signal(w, x)

    # audio outputs
    wavwrite(x, "tests/results/speech_$(params)_original.wav")
    wavwrite(y, "tests/results/speech_$(params)_cut.wav")
    wavwrite(w1, "tests/results/speech_$(params)_reconstructed.wav")

    # plots
    plot(x, label="original signal")
    plot!(w1, normalize=true, label="reconstructed signal")
    vline!([t0], linewidth=0.5, color="red", label="")
    plot!(legend=true)
    savefig("tests/results/speech_$(params)_wave.png")
    plot(Y)
    vline!([t0], linewidth=0.5, color="yellow", legend=false)
    savefig("tests/results/speech_$(params)_cut.png")
    plot(Wp)
    vline!([t0], linewidth=0.5, color="yellow", legend=false)
    savefig("tests/results/speech_$(params)_reconstructed.png")

    return w1
end