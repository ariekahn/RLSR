struct SRMBTD0TD1WeightedModel <: AbstractWeightedModel
    TD0Model::MFTDModel
    TD1Model::MFTDModel
    SRModel::SRModel
    MBModel::MBModel

    V::Vector{Float64}
    QTD0::Vector{Float64}
    QTD1::Vector{Float64}
    QSR::Vector{Float64}
    QMB::Vector{Float64}
end
function SRMBTD0TD1WeightedModel(TD0Model, TD1Model, SRModel, MBModel)
    SRMBTD0TD1WeightedModel(
        TD0Model, TD1Model, SRModel, MBModel,
        zeros(length(TD0Model.V)),
        zeros(length(TD0Model.V)),
        zeros(length(TD0Model.V)),
        zeros(length(TD0Model.V)),
        zeros(length(TD0Model.V)),
    )
    
end
function model_name(model::M; kwargs...) where {M <: SRMBTD0TD1WeightedModel}
    "SRMBTD0TD1WeightedModel: [βSR: $(model.βSR) βMB: $(model.βMB) βTD0: $(model.βTD0) βTD1: $(βTD1)]"
end

function SRMBTD0TD1WeightedAgent(env, TD0Model, TD1Model, SRModel, MBModel, βTD0, βTD1, βSR, βMB, βBoat)
    model = SRMBTD0TD1WeightedModel(TD0Model, TD1Model, SRModel, MBModel)
    policy = PolicySRMBMFTD0TD1TwoStepSoftmax(βTD0, βTD1, βSR, βMB, βBoat)
    Agent(env, model, policy)
end

function update_model_start!(agent::Agent{E, M, P}) where {E, M <: SRMBTD0TD1WeightedModel, P}
    update_model_start!(Agent(agent.env, agent.model.TD0Model, agent.policy))
    update_model_start!(Agent(agent.env, agent.model.TD1Model, agent.policy))
    update_model_start!(Agent(agent.env, agent.model.SRModel, agent.policy))
    update_model_start!(Agent(agent.env, agent.model.MBModel, agent.policy))
    agent.model.QTD0[:] = agent.model.TD0Model.Q
    agent.model.QTD1[:] = agent.model.TD1Model.Q
    agent.model.QSR[:] = agent.model.SRModel.Q
    agent.model.QMB[:] = agent.model.MBModel.Q
    agent.model.V[:] = agent.model.TD0Model.V
end

function update_model_step!(agent::Agent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: SRMBTD0TD1WeightedModel, P}
    update_model_step!(Agent(agent.env, agent.model.SRModel, agent.policy), s, reward, s′)
    update_model_step!(Agent(agent.env, agent.model.MBModel, agent.policy), s, reward, s′)
    update_model_step!(Agent(agent.env, agent.model.TD0Model, agent.policy), s, reward, s′)
    update_model_step!(Agent(agent.env, agent.model.TD1Model, agent.policy), s, reward, s′)
    agent.model.QTD0[:] = agent.model.TD0Model.Q
    agent.model.QTD1[:] = agent.model.TD1Model.Q
    agent.model.QSR[:] = agent.model.SRModel.Q
    agent.model.QMB[:] = agent.model.MBModel.Q
    agent.model.V[:] = agent.model.TD0Model.V
end

function update_model_step_blind!(agent::Agent{E, M, P}, s::Int, s′::Int) where {E, M <: SRMBTD0TD1WeightedModel, P}
    update_model_step_blind!(Agent(agent.env, agent.model.TD0Model, agent.policy), s, s′)
    update_model_step_blind!(Agent(agent.env, agent.model.TD1Model, agent.policy), s, s′)
    update_model_step_blind!(Agent(agent.env, agent.model.SRModel, agent.policy), s, s′)
    update_model_step_blind!(Agent(agent.env, agent.model.MBModel, agent.policy), s, s′)
    agent.model.QTD0[:] = agent.model.TD0Model.Q
    agent.model.QTD1[:] = agent.model.TD1Model.Q
    agent.model.QSR[:] = agent.model.SRModel.Q
    agent.model.QMB[:] = agent.model.MBModel.Q
    agent.model.V[:] = agent.model.TD0Model.V
end

function update_model_end!(agent::Agent{E, M, P}, episode::Episode) where {E, M <: SRMBTD0TD1WeightedModel, P}
    update_model_end!(Agent(agent.env, agent.model.TD0Model, agent.policy), episode)
    update_model_end!(Agent(agent.env, agent.model.TD1Model, agent.policy), episode)
    update_model_end!(Agent(agent.env, agent.model.SRModel, agent.policy), episode)
    update_model_end!(Agent(agent.env, agent.model.MBModel, agent.policy), episode)
    agent.model.QTD0[:] = agent.model.TD0Model.Q
    agent.model.QTD1[:] = agent.model.TD1Model.Q
    agent.model.QSR[:] = agent.model.SRModel.Q
    agent.model.QMB[:] = agent.model.MBModel.Q
    agent.model.V[:] = agent.model.TD0Model.V
end

# Snapshot code
struct SRMBTD0TD1WeightedModelSnapshot <: AbstractModelSnapshop
    V::Vector{Float64}
    QTD0::Vector{Float64}
    QTD1::Vector{Float64}
    QSR::Vector{Float64}
    QMB::Vector{Float64}
    M::Matrix{Float64}
    
end
function SRMBTD0TD1WeightedModelSnapshot(model::SRMBTD0TD1WeightedModel)
    SRMBTD0TD1WeightedModelSnapshot(
        copy(model.V),
        copy(model.QTD0),
        copy(model.QTD1),
        copy(model.QSR),
        copy(model.QMB),
        copy(model.SRModel.M))
end
mutable struct SRMBTD0TD1WeightedModelRecord{E <: AbstractEnv, P <: AbstractPolicy} <: AbstractRecord
    env::E
    policy::P
    V::Matrix{Float64}
    QTD0::Matrix{Float64}
    QTD1::Matrix{Float64}
    QSR::Matrix{Float64}
    QMB::Matrix{Float64}
    M::Array{Float64, 3}
    n::Int
end
function SRMBTD0TD1WeightedModelRecord(agent::Agent{E,M,P}, maxsize::Int)::SRMBTD0TD1WeightedModelRecord where {E <: AbstractEnv, M <: SRMBTD0TD1WeightedModel, P <: AbstractPolicy}
    SRMBTD0TD1WeightedModelRecord(
        agent.env,
        agent.policy,
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env), length(agent.env)),
        0)
end
Base.firstindex(record::SRMBTD0TD1WeightedModelRecord) = 1
Base.lastindex(record::SRMBTD0TD1WeightedModelRecord) = length(record)
Base.length(record::SRMBTD0TD1WeightedModelRecord) = record.n
function Base.push!(record::SRMBTD0TD1WeightedModelRecord, model::SRMBTD0TD1WeightedModel)
    record.n += 1
    (sx, sy) = size(record.V)
    if record.n > sx
        new_V = zeros(sx * 2, sy)
        new_V[1:sx, :] .= record.V
        record.V = new_V

        new_QTD0 = zeros(sx * 2, sy)
        new_QTD0[1:sx, :] .= record.QTD0
        record.QTD0 = new_QTD0

        new_QTD1 = zeros(sx * 2, sy)
        new_QTD1[1:sx, :] .= record.QTD1
        record.QTD1 = new_QTD1

        new_QSR = zeros(sx * 2, sy)
        new_QSR[1:sx, :] .= record.QSR
        record.QSR = new_QSR

        new_QMB = zeros(sx * 2, sy)
        new_QMB[1:sx, :] .= record.QMB
        record.QMB = new_QMB

        new_M = zeros(sx * 2, sy, sy)
        new_M[1:sx, :, :] .= record.M
        record.M = new_M
    end
    record.V[record.n, :] = model.V[:]
    record.QTD0[record.n, :] = model.QTD0[:]
    record.QTD1[record.n, :] = model.QTD1[:]
    record.QSR[record.n, :] = model.QSR[:]
    record.QMB[record.n, :] = model.QMB[:]
    record.M[record.n, :, :] = model.SRModel.M[:, :]
end
function Base.iterate(record::SRMBTD0TD1WeightedModelRecord, state=1)
    if state > length(record)
        nothing
    else
        (SRMBTD0TD1WeightedModelSnapshot(
            record.V[state, :],
            record.QTD0[state, :],
            record.QTD1[state, :],
            record.QSR[state, :],
            record.QMB[state, :],
            record.M[state, :, :]), state+1)
    end
end
function Base.getindex(record::SRMBTD0TD1WeightedModelRecord, i::Int)
    1 <= i <= length(record) || throw(BoundsError(record, i))
    SRMBTD0TD1WeightedModelSnapshot(
        record.V[i, :], 
        record.QTD0[i, :],
        record.QTD1[i, :],
        record.QSR[i, :],
        record.QMB[i, :],
        record.M[i, :, :])
end
Base.getindex(record::SRMBTD0TD1WeightedModelRecord, I) = SRMBTD0TD1WeightedModelRecord(
    record.env,
    record.policy,
    record.V[I, :],
    record.QTD0[I, :],
    record.QTD1[I, :],
    record.QSR[I, :],
    record.QMB[I, :],
    record.M[I, I, :],
    length(I))

function Record(agent::Agent{E, M, P}, maxsize::Int)::SRMBTD0TD1WeightedModelRecord where {E, M <: SRMBTD0TD1WeightedModel, P}
    SRMBTD0TD1WeightedModelRecord(agent, maxsize)
end