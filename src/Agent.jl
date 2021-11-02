struct Agent{E <: AbstractEnv, M <: AbstractModel, P <: AbstractPolicy}
    env::E
    model::M
    policy::P
end

function Agent(model, policy)
    Agent(model.env, model, policy)
end

function model_name(agent::Agent; kwargs...) model_name(agent.model; kwargs...) end
function policy_name(agent::Agent; kwargs...) policy_name(agent.policy; kwargs...) end

"""
    sample_successor(agent, s)

Choose a successor state `s'` to state `s` according to the agent's model and policy.

Return either a state index, or `Nothing` if there are no valid successors."""
function sample_successor(agent::Agent, s::Int)::Union{Int, Nothing}
    sample_successor(agent.env, agent.model, agent.policy, s)
end


"""
    active_episode!(agent, s)

Simulate an episode starting from state `s` and until reaching a terminal state.

Return an `episode` of the sequence of states and rewards
"""
function active_episode!(agent::Agent, s::Int)
    S = [s]
    R = zeros(1)
    update_model_start!(agent)
    while true
        # Choose an action
        s′ = sample_successor(agent, s)
        if isnothing(s′)
            break
        end
        append!(S, s′)

        # Reward
        r = sample_reward(agent.env, s′)
        append!(R, r)

        update_model_step!(agent, s, r, s′)
        s = s′
    end
    episode = Episode(S, R)
    update_model_end!(agent, episode)
    episode
end


"""
    passive_episode!(agent, episode)

Replay the provided episode, updating the agent's model appropriately.
"""
function passive_episode!(agent::Agent, episode::Episode)
    s = episode.S[1]
    r = episode.R[1]
    update_model_start!(agent)
    for (s′, r) in episode[2:end]
        update_model_step!(agent, s, r, s′)
        s = s′
    end
    update_model_end!(agent, episode)
end

"""
    blind_episode(agent, s)

Simulate an episode starting from state `s` and until reaching a terminal state.

However, the agent is not updated with reward information.

Return an `episode` of the sequence of states and rewards
"""
function blind_episode(agent::Agent, s::Int)
    S = [s]
    R = zeros(1)
    while true
        # Choose an action
        s′ = sample_successor(agent, s)
        if isnothing(s′)
            break
        end
        append!(S, s′)

        # Reward
        r = sample_reward(agent.env, s′)
        append!(R, r)

        update_model_step_blind!(agent, s, s′)
        s = s′
    end
    Episode(S, R)
end

function update_rewards!(agent::Agent, R::Vector{Float64})
    update_rewards!(agent.env, R)
end

function update_rewards!(agent::Agent, R_μ::Vector{Float64}, R_σ::Vector{Float64})
    update_rewards!(agent.env, R_μ, R_σ)
end