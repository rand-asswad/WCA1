
# Sparse 3D Array using DOK storage
struct Sparse3DArray{T,Ti<:Integer} <: AbstractSparseArray{T,Ti,3}
    dims::Dims{3}
    data::Dict{NTuple{3,Ti},T}

    function Sparse3DArray{T,Ti}(::UndefInitializer, dims::Dims{3}) where {T,Ti<:Integer}
        return new{T,Ti}(dims, Dict{Tuple{3,Ti},T}())
    end
    function Sparse3DArray(a::Sparse3DArray{T,Ti}) where {T,Ti<:Integer}
        return new{T,Ti}(a.dims, copy(a.data))
    end
end

# Constructors
Sparse3DArray{T,Ti}(::UndefInitializer, dims...) where {T,Ti<:Integer} = Sparse3DArray{T,Ti}(undef, dims)
Sparse3DArray{T,Ti}(args...) where {T,Ti<:Integer} = Sparse3DArray{T,Ti}(undef, args...)
@inline function Sparse3DArray{T}(dims::Dims{3}) where {T}
    d = maximum(dims)
    for Ti in [Int8,Int16,Int32,Int64]
        if d <= typemax(Ti); return Sparse3DArray{T,Ti}(undef, dims) end
    end
    @error "invalid Array size (dimensions do not fit in Int64)"
end
Sparse3DArray{T}(dims...) where {T} = Sparse3DArray{T}(dims)

# Base functions for interface inheritance
Base.size(s::Sparse3DArray) = s.dims
Base.similar(s::Sparse3DArray, ::Type{T}, ::Type{Ti}, dims::Dims{3}=s.dims) where {T,Ti<:Integer} = Sparse3DArray{T,Ti}(dims)
Base.similar(s::Sparse3DArray{T,Ti}, ::Type{Tnew}, dims::Dims{3}=s.dims) where {T,Tnew,Ti<:Integer} = Sparse3DArray{Tnew,Ti}(dims)
Base.copy(s::Sparse3DArray) = Sparse3DArray(s)

# getindex
Base.@propagate_inbounds Base.getindex(s::Sparse3DArray, I::Vararg{Integer,3}) = getindex(s, I)
@inline function Base.getindex(s::Sparse3DArray{T,Ti}, I::NTuple{3,Integer}) where {T,Ti<:Integer}
    @boundscheck checkbounds(s, I...)
    return get(s.data, I, zero(T))
end
Base.@propagate_inbounds Base.getindex(s::Sparse3DArray, I::CartesianIndex{3}) = getindex(s, (I[1], I[2], I[3]))

# setindex
Base.@propagate_inbounds Base.setindex!(s::Sparse3DArray, v, I::Vararg{Integer,3}) = setindex!(s, v, I)
@inline function Base.setindex!(s::Sparse3DArray{T}, v, I::NTuple{3,Integer}) where T
    @boundscheck checkbounds(s, I...)
    if v != zero(v)
        s.data[I] = v
    else
        delete!(s.data, I)
    end
    return v
end
Base.@propagate_inbounds Base.setindex!(s::Sparse3DArray, v, I::CartesianIndex{3}) = setindex!(s, v, (I[1], I[2], I[3]))

# SparseArrays functions
SparseArrays.nnz(s::Sparse3DArray) = length(s.data)
SparseArrays.nonzeros(s::Sparse3DArray) = collect(values(s.data))

"""
    sparse(::AbstractArray{T,3} [, indexType])

Convert a 3-dimensional [`AbstractArray`](@ref) into a sparse 3D-array.
"""
function SparseArrays.sparse(A::AbstractArray{T,3}) where {T}
    d = maximum(size(A))
    for Ti in [Int8,Int16,Int32,Int64]
        if d <= typemax(Ti); return sparse(A, Ti) end
    end
    @warn "Array is too large, cannot sparsify"
    return A
end
function SparseArrays.sparse(A::AbstractArray{T,3}, ::Type{Ti}) where {T,Ti<:Integer}
    s = Sparse3DArray{T,Ti}(size(A))
    for I in CartesianIndices(A)
        s[I] = A[I]
    end
    return s
end

function SparseArrays.sparse(I::AbstractVector{Ti}, J::AbstractVector{Ti}, K::AbstractVector{Ti},
                             V::AbstractVector{T}, n1::Integer=maximum(I), n2::Integer=maximum(J),
                             n3::Integer=maximum(K)) where {T,Ti<:Integer}
    coolen = length(I)
    if length(J) != coolen || length(K) != coolen || length(V) != coolen
        throw(ArgumentError(string("the first four arguments' lengths must match, ",
              "length(I) (= $(length(I))) == length(J) (= $(length(J))) == length(K) (= ",
              "$(length(K))) == length(V) (= $(length(V)))")))
    end
    if Base.hastypemax(Ti) && coolen >= typemax(Ti)
        throw(ArgumentError("the index type $Ti cannot hold $coolen elements; use a larger index type"))
    end
    s = Sparse3DArray{T,Ti}(n1, n2, n3)
    for i in 1:coolen
        s[I[i], J[i], K[i]] = V[i]
    end
    return s
end

"""
    spzeros([type,] n1, n2, n3)

Create a sparse 3D-array of size `n1×n2×n3`.
This sparse array will not contain any nonzero values.
No storage will be allocated for nonzero values during construction.
The type defaults to [`Float64`](@ref) if not specified.
"""
SparseArrays.spzeros(n1::Integer, n2::Integer, n3::Integer) = spzeros(Float64, n1, n2, n3)
SparseArrays.spzeros(::Type{T}, n1::Integer, n2::Integer, n3::Integer) where T = Sparse3DArray{T}(n1,n2,n3)
SparseArrays.spzeros(::Type{T}, ::Type{Ti}, n1::Integer, n2::Integer, n3::Integer) where {T,Ti<:Integer} = Sparse3DArray{T,Ti}(n1,n2,n3)

function SparseArrays.sprand(n1::Integer, n2::Integer, n3::Integer, density::AbstractFloat)
    z = zeros(n1, n2, n3)
    for i = 1:n1
        z[i,:,:] = sprand(n2, n3, density)
    end
    return sparse(z)
end

"""
    findnz(A)
Return a tuple `(I, J, K, V)` where `I`, `J`, and `K` are the indices of the stored
("structurally non-zero") values in sparse 3D-array `A`, and `V` is a vector of the values.
"""
function SparseArrays.findnz(s::Sparse3DArray{T,Ti}) where {T,Ti<:Integer}
    N = nnz(s)
    I = Vector{Ti}(undef, N)
    J = Vector{Ti}(undef, N)
    K = Vector{Ti}(undef, N)
    V = Vector{T}(undef, N)

    i = 1
    for (index, val) in s.data
        I[i] = index[1]
        J[i] = index[2]
        K[i] = index[3]
        V[i] = val
        i += 1
    end
    return (I, J, K, V)
end

# to-do
# SparseArrays._sparse_findnextnz(s, i)
# https://github.com/JuliaLang/julia/blob/69fcb5745bda8a5588c089f7b65831787cffc366/stdlib/SparseArrays/src/abstractsparse.jl
# check lines 76-77

function Base.show(io::IO, ::MIME"text/plain", S::Sparse3DArray)
    xnnz = nnz(S)
    n, m, l = size(S)
    print(io, n, "×", m, "×", l, " ", typeof(S), " with ", xnnz, " stored ",
              xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        print(io, ":")
        show(IOContext(io, :typeinfo => eltype(S)), S)
    end
end

Base.show(io::IO, S::Sparse3DArray) = show(convert(IOContext, io), S::Sparse3DArray)
function Base.show(io::IOContext, S::Sparse3DArray)
    nnz(S) == 0 && return show(io, MIME("text/plain"), S)

    I, J, K, V = findnz(S)

    ioc = IOContext(io, :compact => true)
    function _format_line(i, pad1, pad2, pad3)
        print(ioc, "\n  [", rpad(I[i], pad1), ", ", rpad(J[i], pad2), ", ", lpad(K[i], pad3), "]  =  ", V[i])
    end

    rows = displaysize(io)[1] - 4 # -4 from [Prompt, header, newline after elements, new prompt]
    if !get(io, :limit, false) || rows >= nnz(S) # Will the whole matrix fit when printed?
        pad1, pad2, pad3 = ndigits.((maximum(I), maximum(J), maximum(K)))
        _format_line.(1:nnz(S), pad1, pad2, pad3)
    else
        if rows <= 2
            print(io, "\n  \u22ee")
            return
        end
        s1, e1 = 1, div(rows - 1, 2) # -1 accounts for \vdots
        s2, e2 = nnz(S) - (rows - 1 - e1) + 1, nnz(S)
        pad1 = ndigits(max(maximum(I[s1:e1]), maximum(I[s2:e2])))
        pad2 = ndigits(max(maximum(J[s1:e1]), maximum(J[s2:e2])))
        pad3 = ndigits(max(maximum(K[s1:e1]), maximum(K[s2:e2])))
        _format_line.(s1:e1, pad1, pad2, pad3)
        print(io, "\n  \u22ee")
        _format_line.(s2:e2, pad1, pad2, pad3)
    end
    return
end

Base.sizeof(d::Dict{K,V}) where {K,V} = (sizeof(K) + sizeof(V)) * length(d)
Base.sizeof(s::AbstractSparseArray) = sum(sizeof(getfield(s, i)) for i in 1:fieldcount(typeof(s)))
_sizeof_dense(s::AbstractSparseArray) = sizeof(eltype(s)) * length(s)
sparsity_size_ratio(s::AbstractSparseArray) = sizeof(s) / _sizeof_dense(s)
sparsity_saved_size(s::AbstractSparseArray) = _sizeof_dense(s) - sizeof(s)