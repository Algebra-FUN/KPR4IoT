#=
KPR4IoT.jl
Code for simulation of KPR Game for Resource Allocation in IoT
© 2022 Algebra-FUN(Y. Fan). All rights reserved.
=#

using Distributions
import Base

Base.getindex(A::AbstractArray, ::Colon, s::Symbol) = getproperty.(A,s)

mutable struct RB
    id::Integer
    rank::Integer
    usage::Integer
    RB(id::Integer) = new(id,id,0)
end

mutable struct IoT
    id::Integer
    coord::Vector{Real}
    rb::Union{RB,Nothing}
    neighbors::Vector{IoT}
    IoT(id::Integer,x::Real,y::Real) = new(id,[x;y],nothing,[])
end

Base.show(io::IO, x::IoT)=print(io,"IoT[id=$(x.id)]")
Base.show(io::IO, x::RB)=print(io,"RB[id=$(x.id)]")

function rank(iot::IoT)
    if iot.rb == nothing
        return nothing
    end
    return iot.rb.rank
end

function status(iot::IoT)
    if iot.rb == nothing
        return nothing
    end
    return iot.rb == 1
end

function update!(iot::IoT,rb::Union{RB,Nothing})
    iot.rb = rb
end

function deployIoTs(λ::Real,R::Real)::Vector{IoT}
    N = rand(Poisson(λ*R^2))
    x = rand(Uniform(0,R),N)
    y = rand(Uniform(0,R),N)
    return IoT.(1:N,x,y)
end

function initRBs(N::Integer)::Vector{RB}
    return RB.(1:N)
end

distance(a::IoT,b::IoT) = √sum((a.coord .- b.coord).^2)

function match_neighbors!(IoTs::Vector{IoT},r::Real)
    N = length(IoTs)
    for i in 1:N
        IoTs[i].neighbors = []
        for j in 1:N
            if i != j && distance(IoTs[i],IoTs[j]) < r
                push!(IoTs[i].neighbors,IoTs[j])
            end
        end
    end
end

choose_randomly(iot::IoT; RBs::Vector{RB}) = rand(RBs)

service_rate(RBs::Vector{RB}) = mean(RBs[:,:usage] .== 1)

function count_usage!(rb::RB; choices::Vector{RB})
    rb.usage = 0
    for choice in choices
        if choice.id == rb.id
            rb.usage += 1
        end
    end
end

function cond_neighbor(iot::IoT, cond::Bool)
    neighbors = []
    for neighbor in iot.neighbors
        if status(neighbor) == cond
            push!(neighbors,neighbor)
        end
    end
    return neighbors
end

function choose_by_rank(iot::IoT; RBs::Vector{RB})
    FNs = cond_neighbor(iot, false)
    if !isempty(FNs)
        min_rank = minimum(rank, FNs)
        lower_RBs = filter(rb->rb.rank<=min_rank,RBs)
        return choose_randomly(iot;RBs=lower_RBs)
    end
    SNs = cond_neighbor(iot, true)
    if !isempty(SNs)
        max_rank = maximum(rank, SNs)
        higher_RBs = filter(rb->rb.rank>=max_rank,RBs)
        return choose_randomly(iot;RBs=higher_RBs)
    end
    return choose_randomly(iot;RBs=RBs)
end

function simu!(IoTs::Vector{IoT},RBs::Vector{RB};T::Integer=1000,p::Real=0.01,choose=choose_randomly)
    N = length(IoTs)
    rate = zeros(T)
    for t in 1:T
        issends = rand(Binomial(1,p),N) .|> Bool
        choices = choose.(IoTs[issends];RBs=RBs)
        count_usage!.(RBs;choices=choices)
        update!.(IoTs[issends],choices)
        update!.(IoTs[.!issends],nothing)
        rate[t] = service_rate(RBs)
    end
    return mean(rate)
end
