function bound(v::Vector{Float64})::Vector{Float64}
    v_bounded = v .- minimum(v)
    v_bounded = v_bounded ./ maximum(v_bounded)
end

function plot_graph(env::AbstractGraphEnv; title="")
    nonterminal = findall(.!env.terminal_states)
    p = gplot(env.graph[nonterminal], env.x_coords[nonterminal], 
                env.y_coords[nonterminal], nodelabel=1:nv(env.graph[nonterminal]))
    p = compose(context(),
                (context(), Compose.text(0,0, title), fontsize(18pt)),
                (context(0, 0.05, 1, 0.95), p))
    compose(context(),
        (context(), Compose.rectangle(), fill(nothing), Compose.stroke("black")),
        (context(0, 0.05, 1, 0.95), p))
end

function plot_graph_full(env::AbstractGraphEnv; title="")
    p = gplot(env.graph, env.x_coords, env.y_coords, nodelabel=1:nv(env.graph))
    p = compose(context(),
        (context(), Compose.text(0,0, title), fontsize(18pt)),
        (context(0, 0.05, 1, 0.95), p))
    compose(context(),
        (context(), Compose.rectangle(), fill(nothing), Compose.stroke("black")),
        (context(0, 0.05, 1, 0.95), p))
end

function plot_values(env::AbstractGraphEnv, values; title="")
    nonterminal = findall(.!env.terminal_states)
    colors = get(colorschemes[:viridis], bound(values[nonterminal]))
    labels = [@sprintf("%.2f", f) for f in values[nonterminal]]
    p = gplot(env.graph[nonterminal], env.x_coords[nonterminal], 
              env.y_coords[nonterminal], nodefillc=colors, nodelabel=labels, 
              nodelabelc=colorant"black", NODELABELSIZE=4, nodelabeldist=2, 
              nodelabelangleoffset=pi/2)
    p = compose(context(),
                (context(), Compose.text(0,0, title), fontsize(18pt)),
                (context(0, 0.05, 1, 0.95), p))
    compose(context(),
        (context(), Compose.rectangle(), fill(nothing), Compose.stroke("black")),
        (context(0, 0.05, 1, 0.95), p))
end

function plot_values_full(env::AbstractGraphEnv, values; title="")
    w = colorschemes[:viridis]
    colors = get(colorschemes[:viridis], bound(values))
    labels = [@sprintf("%.2f", f) for f in values]
    p = gplot(env.graph, env.x_coords, env.y_coords, nodefillc=colors, nodelabel=labels,
              nodelabelc=colorant"black", NODELABELSIZE=4, nodelabeldist=2, nodelabelangleoffset=pi/2)
    p = compose(context(),
                (context(), Compose.text(0,0, title), fontsize(18pt)),
                (context(0, 0.05, 1, 0.95), p))
    compose(context(),
        (context(), Compose.rectangle(), fill(nothing), Compose.stroke("black")),
        (context(0, 0.05, 1, 0.95), p))
end

function get_extended_ylims(p::Plots.Plot; factor=0.2)
    yl = ylims(p)
    yrange = yl[2] - yl[1]
    (yl[1], yl[2] + factor * yrange)
end