struct ModelSnapshot <: AbstractModelSnapshop
    V::Vector{Float64}
end
function ModelSnapshot(model::M) where {M<:AbstractModel}
    ModelSnapshot(copy(model.V))
end

mutable struct ModelRecord{E <: AbstractEnv, P <: AbstractPolicy} <: AbstractRecord
    env::E
    policy::P
    V::Matrix{Float64}
    n::Int
end
function ModelRecord(agent::Agent{E,M,P}, maxsize::Int)::ModelRecord where {E <: AbstractEnv, M <: AbstractModel, P <: AbstractPolicy}
    ModelRecord(
        agent.env,
        agent.policy,
        zeros(maxsize, length(agent.env)),
        0)
end
Base.firstindex(record::ModelRecord) = 1
Base.lastindex(record::ModelRecord) = length(record)
Base.length(record::ModelRecord) = record.n
function Base.push!(record::ModelRecord, model::AbstractModel)
    record.n += 1
    (sx, sy) = size(record.V)
    if record.n > sx
        new_V = zeros(sx * 2, sy)
        new_V[1:sx, :] .= record.V
        record.V = new_V
    end
    record.V[record.n, :] = model.V[:]
end
function Base.iterate(record::ModelRecord, state=1)
    if state > length(record)
        nothing
    else
        (ModelSnapshot(record.V[state, :]), state+1)
    end
end
function Base.getindex(record::ModelRecord, i::Int)
    1 <= i <= length(record) || throw(BoundsError(record, i))
    ModelSnapshot(record.V[i, :])
end
Base.getindex(record::ModelRecord, I) = ModelRecord(record.env, record.policy, record.V[I, :], length(I))

function Record(agent::Agent{E, M, P}, maxsize::Int)::ModelRecord where {E, M <: AbstractModel, P}
    ModelRecord(agent, maxsize)
end