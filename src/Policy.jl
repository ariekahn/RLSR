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

struct PolicyTwoStepSoftmax <: AbstractPolicy
    β1::Float64
    β2::Float64
end
function PolicyTwoStepSoftmax(; β1, β2)
    PolicyTwoStepSoftmax(β1, β2)
end
function sample_successor(env::AbstractEnv, model::AbstractModel, policy::PolicyTwoStepSoftmax, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        if (s == 1)
            neighbor_values = exp.(policy.β1 * model.Q[neighbors])
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        else
            neighbor_values = exp.(policy.β2 * model.Q[neighbors])
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        end
    end
end
function policy_name(policy::PolicyTwoStepSoftmax) "Softmax" end


struct PolicySRMBMFTD0TD1TwoStepSoftmax <: AbstractPolicy
    βSR::Float64
    βMB::Float64
    βTD0::Float64
    βTD1::Float64
    βBoat::Float64
end
function sample_successor(env::AbstractEnv, model::AbstractModel, policy::PolicySRMBMFTD0TD1TwoStepSoftmax, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        if (s == 1)
            neighbor_values = exp.(policy.βTD0 * model.QTD0[neighbors] + policy.βTD1 * model.QTD1[neighbors] + policy.βSR * model.QSR[neighbors] + policy.βMB * model.QMB[neighbors])
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        else
            neighbor_values = exp.(policy.βBoat * model.QTD0[neighbors])
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        end
    end
end
function policy_name(policy::PolicySRMBMFTD0TD1TwoStepSoftmax) "SRMBMFTD0TD1TwoStepSoftmax" end