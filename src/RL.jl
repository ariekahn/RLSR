module RL

using LinearAlgebra
using LightGraphs
using Random
using StatsBase
using Distributions
using SpecialFunctions

using DataFrames
using ShiftedArrays

using Printf

using Plots
using GraphPlot
using Compose
using Colors
using ColorSchemes

abstract type AbstractEnv end
abstract type AbstractGraphEnv <: AbstractEnv end
abstract type AbstractPolicy end
abstract type AbstractModel end
abstract type AbstractRecord end

Base.length(model::M) where {M <: AbstractModel} = length(model.V)

include("Env.jl")
include("Episode.jl")
include("Policy.jl")
include("Agent.jl")
include("ModelSR.jl")
include("ModelSRTD.jl")
include("ModelMB.jl")
include("ModelMFTD.jl")
include("ModelWeighted.jl")
include("Snapshots.jl")
include("Dataframes.jl")
include("Plotting.jl")

export AbstractEnv, AbstractGraphEnv, AbstractPolicy, AbstractModel
export GraphEnv, GraphEnvStochastic, GraphEnvStochasticBinary
export sample_reward, get_reward_state, update_rewards!, drift_rewards!
export Observation, Episode
export PolicyGreedy, Policy_ϵ_Greedy, PolicySoftmax
export Agent, model_name, policy_name, active_episode!, passive_episode!, blind_episode
export plot_graph, plot_graph_full, plot_values, plot_values_full

export SRModel, SR_ϵ_Greedy, SRGreedy, SRSoftmax
export SRTDModel, SRTD_ϵ_Greedy, SRTDGreedy, SRTDSoftmax
export MBModel, MB_ϵ_Greedy, MBGreedy, MBSoftmax
export MFTDModel, MFTD_ϵ_Greedy, MFTDGreedy, MFTDSoftmax
export WeightedModel, WeightedModel_ϵ_Greedy, WeightedModelGreedy, WeightedModelSoftmax
export ModelSnapshot, ModelRecord
export RunToDataFrame

function make_env(rewards::Vector{Float64}=[0.5, 0.5, 0.5, 0.5])
    A = [
        0 1 1 0 0 0 0 0 0 0 0
        0 0 0 1 1 0 0 0 0 0 0
        0 0 0 0 0 1 1 0 0 0 0
        0 0 0 0 0 0 0 1 0 0 0
        0 0 0 0 0 0 0 0 1 0 0
        0 0 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 0 1
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
    ]

    locs_y = [0.0,
              1.0, 1.0,
              2.0, 2.0, 2.0, 2.0,
              3.0, 3.0, 3.0, 3.0]
    locs_x = [0.5,
              1/12, 11/12,
              0.0, 1/3, 2/3, 1.0,
              0.0, 1/3, 2/3, 1.0]
    isrewarded = [0,
                  0, 0,
                  0, 0, 0, 0,
                  1, 1, 1, 1]

    rewards = [0,
         0, 0,
         0, 0, 0, 0,
         rewards[1], rewards[2], rewards[3], rewards[4]]
    
    GraphEnvStochasticBinary(A, locs_x, locs_y, isrewarded, rewards)
end
export make_env

function make_env_big(rewards::Vector{Float64}=[0.5, 0.5, 0.5, 0.5])
    A = [
        0 1 1 0 0 0 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0 0 0 0 0
        0 0 0 0 1 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 1 1 0 0 0 0
        0 0 0 0 0 0 0 0 0 1 0 0 0
        0 0 0 0 0 0 0 0 0 0 1 0 0
        0 0 0 0 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 0 0 0 1
        0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0 0 0
    ]

    locs_y = [0.0,
              1.0, 1.0,
              2.0, 2.0,
              3.0, 3.0, 3.0, 3.0,
              4.0, 4.0, 4.0, 4.0]
    locs_x = [0.5,
              1/12, 11/12,
              1/12, 11/12,
              0.0, 1/3, 2/3, 1.0,
              0.0, 1/3, 2/3, 1.0]
    isrewarded = [0,
                  0, 0,
                  0, 0,
                  0, 0, 0, 0,
                  1, 1, 1, 1]

    rewards = [0,
               0, 0,
               0, 0,
               0, 0, 0, 0,
         rewards[1], rewards[2], rewards[3], rewards[4]]
    
    GraphEnvStochasticBinary(A, locs_x, locs_y, isrewarded, rewards)
end
export make_env_big

function make_env_normal(μ::Vector{Float64}=[0.0, 0.0, 0.0, 0.0], σ::Vector{Float64}=[1.0, 1.0, 1.0, 1.0])
    A = [
        0 1 1 0 0 0 0 0 0 0 0
        0 0 0 1 1 0 0 0 0 0 0
        0 0 0 0 0 1 1 0 0 0 0
        0 0 0 0 0 0 0 1 0 0 0
        0 0 0 0 0 0 0 0 1 0 0
        0 0 0 0 0 0 0 0 0 1 0
        0 0 0 0 0 0 0 0 0 0 1
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
		
    ]

    locs_y = [0.0,
              1.0, 1.0,
              2.0, 2.0, 2.0, 2.0,
              3.0, 3.0, 3.0, 3.0]
    locs_x = [0.5,
              1/12, 11/12,
              0.0, 1/3, 2/3, 1.0,
              0.0, 1/3, 2/3, 1.0]

    R_μ = [0,
           0, 0,
           0, 0, 0, 0,
           μ[1], μ[2], μ[3], μ[4]]
    R_σ = [0,
           0, 0,
           0, 0, 0, 0,
           σ[1], σ[2], σ[3], σ[4]]
    
    GraphEnvStochastic(A, locs_x, locs_y, R_μ, R_σ)
end
export make_env_normal

function make_alternate_starts(ntrials::Int)
    # Balance trials sampled
    # Initial shuffle randomizes which state is overexpressed
    ntop = Int(floor(ntrials/2))
    bottomstarts = shuffle(repeat(shuffle([4, 5, 6, 7]), Int(ceil(ntop / 4)))[1:ntop])
    topstarts = ones(Int, length(bottomstarts))
    [topstarts bottomstarts]'[:]
end

function make_bottom_starts(ntrials::Int)
    shuffle(repeat(shuffle([4, 5, 6, 7]), Int(ceil(ntrials / 4)))[1:ntrials])
end

function run_trials(agent::Agent, startstates::Vector{Int})
    agentRecord = Record(agent, length(startstates))
    episodeRecord = Vector{Episode}()   
    for s in startstates
        push!(agentRecord, agent.model)
        episode = active_episode!(agent, s)
        push!(episodeRecord, episode)
    end
    (episodeRecord, agentRecord)
end
function run_trials(agent::Agent, ntrials::Int)
    startstates = make_alternate_starts(ntrials)
    run_trials(agent, startstates)
end

function run_trials!(agent::Agent, startstates::Vector{Int}, agentRecord::R, episodeRecord::Vector{Episode}) where {R<:AbstractRecord}
    for s in startstates
        push!(agentRecord, agent.model)
        episode = active_episode!(agent, s)
        push!(episodeRecord, episode)
    end
end
function run_trials!(agent::Agent, ntrials::Int, agentRecord::R, episodeRecord::Vector{Episode}) where {R<:AbstractRecord}
    startstates = make_alternate_starts(ntrials)
    run_trials!(agent, startstates, agentRecord, episodeRecord)
end

function passive_trials!(agent::Agent, episodeRecord::Vector{Episode})
    agentRecord = Record(agent, length(episodeRecord))
    for episode in episodeRecord
        push!(agentRecord, agent.model)
        passive_episode!(agent, episode)
    end
    agentRecord
end

function passive_trials!(agent::Agent, agentRecord::R, episodeRecord::Vector{Episode}) where {R<:AbstractRecord}
    for episode in episodeRecord
        push!(agentRecord, agent.model)
        passive_episode!(agent, episode)
    end
end

export make_alternate_starts, make_bottom_starts, run_trials!, run_trials, passive_trials!

function run_trials_drift(agent::Agent, startstates::Vector{Int}, σ; lb, ub)
    agentRecord = ModelRecord(agent, length(startstates))
    episodeRecord = Vector{Episode}()   
    envRecord = zeros(length(startstates), 4)
    for (i, s) in enumerate(startstates)
        push!(agentRecord, agent.model)
        episode = active_episode!(agent, s)
        push!(episodeRecord, episode)
        envRecord[i, :] = get_reward_state(agent.env)
        drift_rewards!(agent.env, σ; lb=lb, ub=ub)
    end
    (episodeRecord, agentRecord, envRecord)
end

function run_trials_drift(agent::Agent, ntrials::Int, σ; lb, ub)
    startstates = make_alternate_starts(ntrials)
    run_trials_drift(agent, startstates, σ; lb=lb, ub=ub)
end

export run_trials_drift

function zscore(x)
    (x .- mean(x)) ./ std(x)
end

function softmaximum(a, b, β)
    p= 1 / (1 + exp(-β*(a - b)))
    p*a + (1-p)*b
end

function unitnorm(x)
    0.5 + 0.5 * erf(x / sqrt(2))
end

export zscore, softmaximum, unitnorm

end