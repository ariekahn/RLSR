function RunToDataFrame(episode_record::Vector{Episode}; subject=0)
    #      1
    #  2       3
    # 4 5     6 7
    # 8 9    10 11
    df = DataFrame()

    df.state1 = [ep.S[1] for ep in episode_record]
    df.state2 = [ep.S[2] < 8 ? ep.S[2] : missing for ep in episode_record]
    df.state3 = [length(ep.S) > 2 && ep.S[3] < 8 ? ep.S[3] : missing for ep in episode_record]

    # endState is 2-7 (2/3 for island-only trials, 4-7 for other trials)
    df.endState = [ep.S[end] > 3 ? ep.S[end-1] : ep.S[end] for ep in episode_record]
    df.endBranchLeft = [(ep.S[end] == 2) || (ep.S[end-1] == 4) || (ep.S[end-1] == 5) for ep in episode_record]
    df.reward = [ep.R[end] for ep in episode_record]

    # As it stands, an even state means we went left, odd right
    df.action1Left = mod.(df[!, :state2], 2) .== 0
    df.action2Left = mod.(df[!, :state3], 2) .== 0
    df.rewardₜ₋₁ = lag(df[!, :reward])
    df.rewardₜ₋₂ = lag(df[!, :rewardₜ₋₁])
    df.state1ₜ₋₁ = lag(df[!, :state1])
    df.state1ₜ₋₂ = lag(df[!, :state1ₜ₋₁])
    df.state1ₜ₋₃ = lag(df[!, :state1ₜ₋₂])
    df.state2ₜ₋₁ = lag(df[!, :state2])
    df.state2ₜ₋₂ = lag(df[!, :state2ₜ₋₁])
    df.state3ₜ₋₁ = lag(df[!, :state3])
    df.state3ₜ₋₂ = lag(df[!, :state3ₜ₋₁])
    df.endStateₜ₋₁ = lag(df[!, :endState])
    df.endStateₜ₋₂ = lag(df[!, :endStateₜ₋₁])
    df.endBranchLeftₜ₋₁ = lag(df[!, :endBranchLeft])
    df.endBranchLeftₜ₋₂ = lag(df[!, :endBranchLeftₜ₋₁])
    
    df.trial = 1:nrow(df)
    df[!, :subject] .= string(subject)
    
    # Crucial prediction variable: Is our action towards the same side as the prior sampled state?
    df.action1TowardsPrevEnd = df[!, :action1Left] .== df[!, :endBranchLeftₜ₋₁]

    # df.endBranchₜ₋₁SameAsLastChoice = df[!, :endBranchLeftₜ₋₁] .== df[!, :endBranchLeftₜ₋₂]
    
    #######
    # Prior to the current episode, what was the most recent move at each state?
    priorMoveMat = Matrix{Union{Missing, Int}}(missing, length(episode_record), 3)
    priorMoveVec = Vector{Union{Missing, Int}}(missing, 3)
    for (i, ep) in enumerate(episode_record)
        if ep.S[1] == 1
            # If a full traversal, update moves for start and island
            if length(ep.S) > 2
                endstate = ep.S[end]
                choice_1 = 2 + (endstate > 9)  # 2 or 3
                choice_2 = endstate - 4  # 4,5,6,7
                priorMoveVec[1] = choice_1
                priorMoveVec[choice_1] = choice_2
            # Otherwise just update moves for start
            else
                priorMoveVec[1] = ep.S[2]
            end
        end
        priorMoveMat[i, :] = priorMoveVec[:]
    end
    df.priorMoveAt1 = lag(priorMoveMat[:, 1])
    df.priorMoveAt2 = lag(priorMoveMat[:, 2])
    df.priorMoveAt3 = lag(priorMoveMat[:, 3])
    # df.priorMoveAt1Left = mod.(df[!, :priorMoveAt1], 2) .== 0
    # df.priorMoveAt2Left = mod.(df[!, :priorMoveAt2], 2) .== 0
    # df.priorMoveAt3Left = mod.(df[!, :priorMoveAt3], 2) .== 0
    # Prior to the current episode, what was the most recent move at the parent
    # of the state we finished this episode in?
    parentMap = Dict(2 => :priorMoveAt1,
                     3 => :priorMoveAt1,
                     4 => :priorMoveAt2,
                     5 => :priorMoveAt2,
                     6 => :priorMoveAt3,
                     7 => :priorMoveAt3)
    df.parentPriorMove = [df[i, parentMap[df[i, :endState]]] for i in 1:nrow(df)]
    # df.priorMoveAtParentLeft = mod.(df[!, :priorMoveAtParent], 2) .== 0
    # df.priorMoveAtParentLeftₜ₋₁ = lag(df[!, :priorMoveAtParentLeft])
    df.parentPriorMoveToEndState = df[!, :parentPriorMove] .== df[!, :endState]
    df.parentPriorMoveToEndStateₜ₋₁ = lag(df[!, :parentPriorMoveToEndState])
    
    ######
    # Prior to the current episode, what was the reward observed at a given state?
    priorRewardMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 6)
    priorRewardVec = Vector{Union{Missing, Float64}}(missing, 6)
    for (i, ep) in enumerate(episode_record)
        if ep.S[end] > 3  # For trials that end on boats, 8 -> 3, 9 -> 4, 10 -> 5, 11 -> 6
            ind = ep.S[end] - 5
            priorRewardVec[ind] = ep.R[end]
        else  # For trials that end on islands, 2 -> 1, 3 -> 2
            ind = ep.S[end] - 1
            priorRewardVec[ind] = ep.R[end]
        end
        priorRewardMat[i, :] = priorRewardVec[:]
    end
    # If at trial 10, priorRewardAtX contains the last update including trial 9
    df.priorRewardAt2 = lag(priorRewardMat[:, 1])
    df.priorRewardAt3 = lag(priorRewardMat[:, 2])
    df.priorRewardAt4 = lag(priorRewardMat[:, 3])
    df.priorRewardAt5 = lag(priorRewardMat[:, 4])
    df.priorRewardAt6 = lag(priorRewardMat[:, 5])
    df.priorRewardAt7 = lag(priorRewardMat[:, 6])
    # Which index in recentRewardMat should each end-state look into?
    rewardSiblingMap = Dict(2 => :priorRewardAt3,
                            3 => :priorRewardAt2,
                            4 => :priorRewardAt5,
                            5 => :priorRewardAt4,
                            6 => :priorRewardAt7,
                            7 => :priorRewardAt6)
    # If on trial 10 we ended on 
    df.endStateSiblingPriorReward = [df[i, rewardSiblingMap[df[i, :endState]]] for i in 1:nrow(df)]
    df.endStateSiblingPriorRewardₜ₋₁ = lag(df[!, :endStateSiblingPriorReward])
    # 
    rewardMap = Dict(2 => :priorRewardAt2,
                     3 => :priorRewardAt3,
                     4 => :priorRewardAt4,
                     5 => :priorRewardAt5,
                     6 => :priorRewardAt6,
                     7 => :priorRewardAt7)
    df.endStatePriorReward = [df[i, rewardMap[df[i, :endState]]] for i in 1:nrow(df)]
    df.endStatePriorRewardₜ₋₁ = lag(df[!, :endStatePriorReward])
    df.endStatePriorRewardₜ₋₂ = lag(df[!, :endStatePriorRewardₜ₋₁])
    

    #####
    # Prior reward observed on the left/right branches
    #
    priorRewardBranchMat = Matrix{Union{Missing, Float64}}(missing, length(episode_record), 2)
    priorRewardBranchVec = Vector{Union{Missing, Float64}}(missing, 2)
    for (i, ep) in enumerate(episode_record)
        if ep.S[end] > 3
            ind = 1 + (ep.S[end] > 9)
            priorRewardBranchVec[ind] = ep.R[end]
        end
        priorRewardBranchMat[i, :] = priorRewardBranchVec[:]
    end
    df.priorRewardLeftBranch = lag(priorRewardBranchMat[:, 1])
    df.priorRewardRightBranch = lag(priorRewardBranchMat[:, 2])
    # Which index in priorRewardBranchMat should each end-state look into?
    rewardBranchMap = Dict(2 => :priorRewardLeftBranch,
                           3 => :priorRewardRightBranch,
                           4 => :priorRewardLeftBranch,
                           5 => :priorRewardLeftBranch,
                           6 => :priorRewardRightBranch,
                           7 => :priorRewardRightBranch)
    df.endStateBranchPriorReward = [df[i, rewardBranchMap[df[i, :endState]]] for i in 1:nrow(df)]
    df.endStateBranchPriorRewardₜ₋₁ = lag(df[!, :endStateBranchPriorReward])
    df.endStateBranchPriorRewardₜ₋₂ = lag(df[!, :endStateBranchPriorRewardₜ₋₁])
   
    # Last Sampled Reward
    # priorSampledRewardMat = Matrix{Union{Missing, Int}}(missing, length(episode_record), 4)
    # priorSampledRewardVec = Vector{Union{Missing, Int}}(missing, 4)
    # for (i, ep) in enumerate(episode_record)
    #     if ep.S[1] > 3
    #         ind = ep.S[end] - 7
    #         priorSampledRewardVec[ind] = ep.R[end]
    #     end
    #     priorSampledRewardMat[i, :] = recentSampledRewardVec[:]
    # end
    # df.priorSampledRewardAt4 = lag(recentSampledRewardMat[:, 1])
    # df.priorSampledRewardAt5 = lag(recentSampledRewardMat[:, 2])
    # df.priorSampledRewardAt6 = lag(recentSampledRewardMat[:, 3])
    # df.priorSampledRewardAt7 = lag(recentSampledRewardMat[:, 4])
    # sampledRewardSiblingMap = Dict(4 => :priorSampledRewardAt5,
    #                                5 => :priorSampledRewardAt4,
    #                                6 => :priorSampledRewardAt7,
    #                                7 => :priorSampledRewardAt6)
    # df.endSiblingPriorSampledReward = [df[i, rewardSiblingMap[df[i, :endState]]] for i in 1:nrow(df)]
    # df.endSiblingPriorSampledRewardₜ₋₁ = lag(df[!, :endSiblingRecentSampledReward])
    
    df
end


function RunToDataFrameBig(episode_record::Vector{Episode}; subject=0)
    #      1
    #  2       3
    # 4 5     6 7
    # 8 9    10 11
    df = DataFrame()

    df.state1 = [ep.S[1] for ep in episode_record]
    df.state2 = [ep.S[2] < 8 ? ep.S[2] : missing for ep in episode_record]
    df.state3 = [length(ep.S) > 2 && ep.S[3] < 8 ? ep.S[3] : missing for ep in episode_record]
    df.state4 = [length(ep.S) > 3 && ep.S[3] < 16 ? ep.S[4] : missing for ep in episode_record]

    # endState is 2-7 (2/3 for island-only trials, 4-7 for other trials)
    df.endState = [ep.S[end] > 3 ? ep.S[end-1] : ep.S[end] for ep in episode_record]
    df.reward = [ep.R[end] for ep in episode_record]

    df.trial = 1:nrow(df)
    df[!, :subject] .= string(subject)

    df
end