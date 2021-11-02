abstract type AbstractWeightedModel <: AbstractModel end
struct WeightedModel <: AbstractWeightedModel
    V::Vector{Float64}
    Q::Vector{Float64}
    models
    weights::Vector{Float64}
    function WeightedModel(V, Q, models, weights)
        if any(weights .< 0) || !isapprox(sum(weights), 1)
            error("invalid model weights: $weights")
        end
        if length(models) ≠ length(weights)
            error("number of models ($length(models)) ≠ number of weights ($length(weights))")
        end
        for m ∈ models
            if length(m.V) ≠ length(V)
                error("Different number of states: $m")
            end
        end
        new(V, Q, models, weights)
    end
end
function WeightedModel(models, weights)
    WeightedModel(zeros(length(models[1].V)), zeros(length(models[1].V)), models, weights)
end
function model_name(model::M; show_weights=false, kwargs...) where {M <: AbstractWeightedModel}
    name = "WeightedModel: ["
    for (i, (m, w)) ∈ enumerate(zip(model.models, model.weights))
        if i > 1
            name *= " / "
        end
        if show_weights
            name *= string(w) * " " * model_name(m, kwargs...)
        else
            name *= model_name(m, kwargs...)
        end
    end
    name * "]"
end

function WeightedModel_ϵ_Greedy(env, models, weights; ϵ)
    model = WeightedModel(models, weights)
    policy = Policy_ϵ_Greedy(ϵ)
    Agent(env, model, policy)
end

function WeightedModelSoftmax(env, models, weights; β)
    model = WeightedModel(models, weights)
    policy = PolicySoftmax(β)
    Agent(env, model, policy)
end

function WeightedModelGreedy(env, models, weights)
    model = WeightedModel(models, weights)
    policy = PolicyGreedy()
    Agent(env, model, policy)
end

function reaverage_models!(agent::Agent{E, M, P}) where {E, M <: WeightedModel, P}
    agent.model.V[:] = sum([m.V * w for (m, w) in zip(agent.model.models, agent.model.weights)])
    agent.model.Q[:] = sum([m.Q * w for (m, w) in zip(agent.model.models, agent.model.weights)])
end

function update_model_start!(agent::Agent{E, M, P}) where {E, M <: WeightedModel, P}
    for m ∈ agent.model.models
        update_model_start!(Agent(agent.env, m, agent.policy))
    end
    reaverage_models!(agent)
end

function update_model_step!(agent::Agent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: WeightedModel, P}
    for m ∈ agent.model.models
        update_model_step!(Agent(agent.env, m, agent.policy), s, reward, s′)
    end
    reaverage_models!(agent)
end

function update_model_step_blind!(agent::Agent{E, M, P}, s::Int, s′::Int) where {E, M <: WeightedModel, P}
    for m ∈ agent.model.models
        update_model_step_blind!(Agent(agent.env, m, agent.policy), s, s′)
    end
    reaverage_models!(agent)
end

function update_model_end!(agent::Agent{E, M, P}, episode::Episode) where {E, M <: WeightedModel, P}
    for m ∈ agent.model.models
        update_model_end!(Agent(agent.env, m, agent.policy), episode)
    end
    reaverage_models!(agent)
end