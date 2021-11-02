abstract type AbstractMFTD <: AbstractModel end
struct MFTDModel <: AbstractMFTD
    V::Vector{Float64}
    Q::Vector{Float64}
    trace::Vector{Float64}
    α::Float64
    λ::Float64
    γ::Float64
end
function MFTDModel(env, α, λ, γ)
    n = length(env)
    trace = zeros(length(env))
    MFTDModel(zeros(n), zeros(n), zeros(n), α, λ, γ)
end
MFTDModel(env; α, λ, γ)::MFTDModel = MFTDModel(env, α, λ, γ)
function model_name(model::M) where {M <: AbstractMFTD} "MFTD" end

function MFTD_ϵ_Greedy(env; α, λ, γ, ϵ)
    MFTD = MFTDModel(env, α, λ, γ)
    policy = Policy_ϵ_Greedy(ϵ)
    Agent(env, MFTD, policy)
end

function MFTDSoftmax(env; α, λ, γ, β)
    MFTD = MFTDModel(env, α, λ, γ)
    policy = PolicySoftmax(β)
    Agent(env, MFTD, policy)
end

function MFTDGreedy(env; α, λ, γ)
    MFTD = MFTDModel(env, α, λ, γ)
    policy = PolicyGreedy()
    Agent(env, MFTD, policy)
end

function update_model_start!(agent::Agent{E, M, P}) where {E, M <: AbstractMFTD, P}
    agent.model.trace[:] .= 0
end

function update_model_step_blind!(agent::Agent{E, M, P}, s::Int, s′::Int) where {E, M <: AbstractMFTD, P} end

function update_model_step!(agent::Agent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: AbstractMFTD, P}
    MFTD = agent.model
    # Update eligibility trace
    MFTD.trace[:] = MFTD.trace * (MFTD.γ * MFTD.λ)
    MFTD.trace[s] = MFTD.trace[s] + 1

    δ = reward + MFTD.γ * MFTD.V[s′] - MFTD.V[s]
    MFTD.V[:] = MFTD.V + (MFTD.trace * MFTD.α * δ)

    # Update Q-values 
    # Similar to the SR, this is valid as long as rewards are terminal
    MFTD.Q[:] = MFTD.γ * MFTD.V
end

function update_model_end!(::Agent{E, M, P}, ::Episode) where {E, M <: AbstractMFTD, P}

end