
function linear_chirp(fs = 16000, duration = 2)
    n = round(Int, duration * fs)
    return [n/7 <= t <= 8*n/14 ? sin(2000*2*π*t/fs+1000*t/fs*2*π*t/fs) : 0. for t in 1:n]
end

function nonlinear_chirp(fs = 16000, duration = 2)
    n = round(Int, duration * fs)
    return [n/7 <= t <= 8*n/14 ? sin(3000*2*π*t/fs+2pi*150*sin(2*pi*t/fs)) : 0. for t in 1:n]
end

function interrupted_chirp(fs = 16000, duration = 2)
    n = round(Int, duration * fs)
    x1 = [n/7 <= t <= 2.5*n/7 ? sin(2000*2*π*t/fs+1000*t/fs*2*π*t/fs) : 0. for t in 1:n]
    x2 = [3*n/7 <= t <= 4.5*n/7 ? sin(2000*2*π*t/fs+1000*t/fs*2*π*t/fs) : 0. for t in 1:n]
    return x1 + x2
end

function intersecting_chirp(fs = 16000, duration = 2)
    n = round(Int, duration * fs)
    x1 = [n/7 <= t <= 6*n/14 ? sin(2000*2*π*t/fs + 1000*t/fs*2*π*t/fs)  : 0. for t in 1:n]
    x2 = [n/7 <= t <= 6*n/14 ? sin(6000*2*π*t/fs - 1000*t/fs*2*π*t/fs)  : 0. for t in 1:n]
    return x1 + x2
end