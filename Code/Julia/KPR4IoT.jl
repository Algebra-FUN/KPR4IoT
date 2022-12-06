#=
KPR4IoT.jl
Code for simulation of KPR Game for Resource Allocation in IoT
© 2022 Algebra-FUN(Y. Fan). 
All rights reserved.
=#

using Distributions, Underscores
using Plots, DataFrames
import Base, Random

Base.getindex(A::AbstractVector, slice::Any, key::Symbol) = getproperty.(A[slice], key)

function Base.setindex!(A::AbstractVector, X::Any, slice::Any, key::Symbol)
    setproperty!.(A[slice], key, X)
end

ones_like(x::Array) = ones(eltype(x), size(x))

mutable struct RB
    id::Int
    rank::Int
    usage::Int
    RB(id::Int) = new(id, id, 0)
end

mutable struct IoT
    id::Int
    coord::Vector{Real}
    rb::Union{RB,Nothing}
    neighbors::Vector{IoT}
    IoT(id::Int, x::Real, y::Real) = new(id, [x; y], nothing, [])
end

Base.show(io::IO, x::IoT) = print(io, "IoT[id=$(x.id)]")
Base.show(io::IO, x::RB) = print(io, "RB[id=$(x.id)]")

function rank(iot::IoT)
    if iot.rb === nothing
        return nothing
    end
    return iot.rb.rank
end

function status(iot::IoT)
    if iot.rb === nothing
        return nothing
    end
    return iot.rb == 1
end

function deployIoTs(λ::Real, R::Real)::Vector{IoT}
    N = rand(Poisson(λ * R^2))
    x = rand(Uniform(0, R), N)
    y = rand(Uniform(0, R), N)
    return IoT.(1:N, x, y)
end

initRBs(N::Int) = RB.(1:N)

distance(a::IoT, b::IoT) = √sum((a.coord .- b.coord) .^ 2)

function match_neighbors!(IoTs::Vector{IoT}, r::Real)
    for iot in IoTs
        iot.neighbors = @_ filter(0 < distance(iot, _) < r, IoTs)
    end
end

choose_randomly(iot::IoT; RBs::Vector{RB}) = rand(RBs)

service_rate(RBs::Vector{RB}) = mean(RBs[:, :usage] .== 1)

count_usage!(rb::RB; chosen_rbs::Vector{RB}) = rb.usage = count(chosen_rbs[:, :id] .== rb.id)

cond_neighbor(iot::IoT, cond::Bool) = @_ filter(status(_) == cond, iot.neighbors)

function choose_by_rank(iot::IoT; RBs::Vector{RB})
    FNs = cond_neighbor(iot, false)
    if !isempty(FNs)
        min_rank = minimum(rank, FNs)
        lower_RBs = @_ filter(_.rank <= min_rank, RBs)
        return choose_randomly(iot; RBs=lower_RBs)
    end
    SNs = cond_neighbor(iot, true)
    if !isempty(SNs)
        max_rank = maximum(rank, SNs)
        higher_RBs = @_ filter(_.rank >= max_rank, RBs)
        return choose_randomly(iot; RBs=higher_RBs)
    end
    return choose_randomly(iot; RBs=RBs)
end

function simu!(IoTs::Vector{IoT}, RBs::Vector{RB}; T::Int=1000, p::Real=0.01, choose=choose_randomly)
    N = length(IoTs)
    rate = zeros(T)
    for t in 1:T
        issends = rand(Binomial(1, p), N) .|> Bool
        chosen_rbs = choose.(IoTs[issends]; RBs=RBs)
        count_usage!.(RBs; chosen_rbs=chosen_rbs)
        IoTs[issends, :rb] = chosen_rbs
        IoTs[.!issends, :rb] = nothing
        rate[t] = service_rate(RBs)
    end
    return mean(rate)
end

function simulation(; λ=2.5, R=20, b=5, r=0, T=1000, p=0.01, choose=choose_randomly, trials=5, seed=5003666)
    Random.seed!(seed)
    rates = zeros(trials)
    for i in 1:trials
        IoTs = deployIoTs(λ, R)
        RBs = initRBs(b)
        r != 0 && match_neighbors!(IoTs, r)
        rates[i] = simu!(IoTs, RBs; T=T, p=p, choose=choose)
    end
    return mean(rates)
end

function experiment(; p=0.01, rs=1:6, T=100, trials=5)
    rs = collect(rs)
    baseline_25 = 100 * simulation(λ=2.5, R=20, b=5, T=T, p=p, trials=trials)
    baseline_50 = 100 * simulation(λ=5, R=20, b=5, T=T, p=p, trials=trials)
    learning_25 = 100 .* @_ map(simulation(λ=2.5, R=20, b=5, r=_, T=T, p=p, choose=choose_by_rank, trials=trials), rs)
    learning_50 = 100 .* @_ map(simulation(λ=5, R=20, b=5, r=_, T=T, p=p, choose=choose_by_rank, trials=trials), rs)
    baseline_25 = baseline_25 .* ones_like(rs)
    baseline_50 = baseline_50 .* ones_like(rs)
    return DataFrame(r=rs, baseline25=baseline_25, baseline50=baseline_50, learning25=learning_25, learning50=learning_50)
end

function create_experiment_plot(df::DataFrame; p=0.01, ylim=(-0.2, 30))
    fig = plot(df.r, df.learning25, label="Learning with \$\\lambda=2.5\$", line=(:red))
    plot!(fig, df.r, df.baseline25, label="Baseline with \$\\lambda=2.5\$", line=(:blue))
    plot!(fig, df.r, df.learning50, label="Learning with \$\\lambda=5\$", line=(:dash, :red))
    plot!(fig, df.r, df.baseline50, label="Baseline with \$\\lambda=5\$", line=(:dash, :blue))
    xlabel!(fig, "Communication range \$r_c\$ (m)")
    ylabel!(fig, "Service rate (%)")
    title!(fig, "Average service rate of RBs (p=$p)")
    ylims!(fig, ylim)
    return fig
end