struct Policy_ϵ_Greedy <: AbstractPolicy
    ϵ::Float64
end
function Policy_ϵ_Greedy(; ϵ)
    Policy_ϵ_Greedy(ϵ)
end
function sample_successor(env::AbstractEnv, model::AbstractModel, policy::Policy_ϵ_Greedy, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        weights = model.Q[neighbors]
        if rand() < policy.ϵ || sum(weights) == 0
            rand(neighbors)
        else
            rand(neighbors[weights .== maximum(weights)])
        end
    end
end
function policy_name(policy::Policy_ϵ_Greedy) "ϵ-Greedy" end

struct PolicyGreedy <: AbstractPolicy
end
function sample_successor(env::AbstractEnv, model::AbstractModel, ::PolicyGreedy, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        weights = model.Q[neighbors]
        rand(neighbors[weights .== maximum(weights)])
    end
end
function policy_name(policy::PolicyGreedy) "Greedy" end

struct PolicySoftmax <: AbstractPolicy
    β::Float64
end
function PolicySoftmax(; β)
    PolicySoftmax(β)
end
function sample_successor(env::AbstractEnv, model::AbstractModel, policy::PolicySoftmax, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        neighbor_values = exp.(policy.β * model.Q[neighbors])
        weights = Weights(neighbor_values ./ sum(neighbor_values))
        sample(neighbors, weights)
    end
end
function policy_name(policy::PolicySoftmax) "Softmax" end