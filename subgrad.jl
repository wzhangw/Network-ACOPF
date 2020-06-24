using PowerModels, JuMP, Ipopt, Gurobi, Plasmo, LinearAlgebra, MathOptInterface
const MOI = MathOptInterface

function get_index_from_var_name(str::String)::Tuple
    first_idx = 0
    second_idx = 0
    third_idx = 0
    first_begin = 6
    second_begin = 0
    third_begin = 0

    for i in first_begin+1:length(str)
        if !isdigit(str[i])
            first_idx = parse(Int, str[first_begin:i-1])
            second_begin = i+2
            break
        end
    end

    for i in second_begin+1:length(str)
        if !isdigit(str[i])
            second_idx = parse(Int, str[second_begin:i-1])
            third_begin = i+2
            break
        end
    end

    for i in third_begin+1:length(str)
        if !isdigit(str[i])
            third_idx = parse(Int, str[third_begin:i-1])
            break
        end
    end

    return (first_idx, second_idx, third_idx)
end

function build_subgraph_model(
    N_gs::Vector{Vector{Int64}},
    L_gs::Vector{Vector{Tuple{Int64, Int64}}},
    cut_lines::Vector{Tuple{Int64, Int64}},
    load_bus::Vector{Int64},
    Pd::Vector{Float64},
    Qd::Vector{Float64},
    gen_bus::Vector{Int64},
    Pmax::Vector{Float64},
    Pmin::Vector{Float64},
    Qmax::Vector{Float64},
    Qmin::Vector{Float64},
    gen_cost_type::Int64,
    costs::Vector{Vector{Float64}},
    shunt_node::Vector{Int64},
    gs::Dict,
    bs::Dict,
    smax::Dict,
    pm_model::Model,
    λs::Vector{Vector{Float64}}
    )

    N_partitions = length(N_gs)

    cref_type_list = list_of_constraint_types(pm_model)
    crefs = Dict()
    for (i,j) in cref_type_list
        crefs[(i,j)] = all_constraints(pm_model, i, j)
    end

    dm = ModelGraph()

    @node(dm, nodes[1:N_partitions])
    @node(dm, global_node) # dummy node that stores global variables (on cut)

    shared_vars_dict = Dict()

    for k in 1:N_partitions
        gen_idx_k = [i for i in eachindex(gen_bus) if gen_bus[i] in N_gs[k]]
        lines_in_cut = intersect(L_gs[k], cut_lines)
        relevant_idx = union(N_gs[k], vcat([collect(i) for i in L_gs[k]]...))
        expanded_idx = union(relevant_idx, relevant_idx .+ N)

        @variable(nodes[k], W[expanded_idx, expanded_idx])
        @variable(nodes[k], v[expanded_idx])
        @variable(nodes[k], plf[L_gs[k]])
        @variable(nodes[k], plt[L_gs[k]])
        @variable(nodes[k], qlf[L_gs[k]])
        @variable(nodes[k], qlt[L_gs[k]])
        @variable(nodes[k], Pmin[i] <= pg[i in gen_idx_k] <= Pmax[i])
        @variable(nodes[k], Qmin[i] <= qg[i in gen_idx_k] <= Qmax[i])

        for i in expanded_idx, j in expanded_idx
            @constraint(nodes[k], W[i,j] == v[i] * v[j])
        end

        # constraints 1d, 1e for each node
        for i in N_gs[k]
            if i in shunt_node
                @constraint(nodes[k], sum(plf[j] for j in L_gs[k] if j[1] == i)
                             + sum(plt[j] for j in L_gs[k] if j[2] == i)
                             - sum(pg[j] for j in 1:N_gen if gen_bus[j] == i)
                             + sum(Pd[j] for j in 1:N_load if load_bus[j] == i)
                             - gs[i] * (W[i,i] + W[i+N,i+N]) == 0)
                @constraint(nodes[k], sum(qlf[j] for j in L_gs[k] if j[1] == i)
                             + sum(qlt[j] for j in L_gs[k] if j[2] == i)
                             - sum(qg[j] for j in 1:N_gen if gen_bus[j] == i)
                             + sum(Qd[j] for j in 1:N_load if load_bus[j] == i)
                             - bs[i] * (W[i,i] + W[i+N,i+N]) == 0)
            else
                @constraint(nodes[k], sum(plf[j] for j in L_gs[k] if j[1] == i)
                             + sum(plt[j] for j in L_gs[k] if j[2] == i)
                             - sum(pg[j] for j in 1:N_gen if gen_bus[j] == i)
                             + sum(Pd[j] for j in 1:N_load if load_bus[j] == i) == 0)

                @constraint(nodes[k], sum(qlf[j] for j in L_gs[k] if j[1] == i)
                             + sum(qlt[j] for j in L_gs[k] if j[2] == i)
                             - sum(qg[j] for j in 1:N_gen if gen_bus[j] == i)
                             + sum(Qd[j] for j in 1:N_load if load_bus[j] == i) == 0)
            end
        end

        # constraints 1b, 1c
        # extract coefficients for constraints 1b, 1c
        for cref in crefs[cref_type_list[2]]
            aff_expr = constraint_object(cref).func.aff
            var = first(keys(aff_expr.terms))
            # check if cref is actually one of constraints 1b, 1c
            expr = aff_expr - var
            drop_zeros!(expr)
            if (expr == zero(AffExpr))
                quad_terms = constraint_object(cref).func.terms
                p_or_q = name(var)[3]
                (_, f_bus, t_bus) = get_index_from_var_name(name(var))
                if (f_bus, t_bus) in lines
                    is_f = true
                    line = (f_bus, t_bus)
                else
                    is_f = false
                    line = (t_bus, f_bus)
                end
                pairs = [UnorderedPair(variable_by_name(pm_model, "0_vr[$(f_bus)]"), variable_by_name(pm_model, "0_vr[$(f_bus)]")),
                         UnorderedPair(variable_by_name(pm_model, "0_vi[$(f_bus)]"), variable_by_name(pm_model, "0_vi[$(f_bus)]")),
                         UnorderedPair(variable_by_name(pm_model, "0_vr[$(f_bus)]"), variable_by_name(pm_model, "0_vr[$(t_bus)]")),
                         UnorderedPair(variable_by_name(pm_model, "0_vr[$(f_bus)]"), variable_by_name(pm_model, "0_vi[$(t_bus)]")),
                         UnorderedPair(variable_by_name(pm_model, "0_vi[$(f_bus)]"), variable_by_name(pm_model, "0_vr[$(t_bus)]")),
                         UnorderedPair(variable_by_name(pm_model, "0_vi[$(f_bus)]"), variable_by_name(pm_model, "0_vi[$(t_bus)]"))]
                if p_or_q == 'p'
                    if line in L_gs[k]
                        if is_f
                            new_expr = zero(QuadExpr) + plf[line]
                        else
                            new_expr = zero(QuadExpr) + plt[line]
                        end
                        vrefs = [W[f_bus, f_bus], W[f_bus+N, f_bus+N], W[f_bus, t_bus],
                                 W[f_bus, t_bus+N], W[f_bus+N, t_bus], W[f_bus+N, t_bus+N]]
                        @constraint(nodes[k], new_expr + sum(quad_terms[pairs[i]] * vrefs[i] for i in eachindex(pairs) if pairs[i] in keys(quad_terms)) == 0)
                    end
                else # p_or_q == 'q'
                    if line in L_gs[k]
                        if is_f
                            new_expr = zero(QuadExpr) + qlf[line]
                        else
                            new_expr = zero(QuadExpr) + qlt[line]
                        end
                        vrefs = [W[f_bus, f_bus], W[f_bus+N, f_bus+N], W[f_bus, t_bus],
                                 W[f_bus, t_bus+N], W[f_bus+N, t_bus], W[f_bus+N, t_bus+N]]
                        @constraint(nodes[k], new_expr + sum(quad_terms[pairs[i]] * vrefs[i] for i in eachindex(pairs) if pairs[i] in keys(quad_terms)) == 0)
                    end
                end
            end
        end

        # constraints 1g
#        for i in N_gs[k]
        for i in relevant_idx
            @constraint(nodes[k], Vmin[i]^2 <= W[i,i] + W[i+N, i+N] <= Vmax[i]^2)
        end

        # constraints 1h
        for i in L_gs[k]
            if i in keys(smax)
                @constraint(nodes[k], plf[i]^2 + qlf[i]^2 <= smax[i]^2)
                @constraint(nodes[k], plt[i]^2 + qlt[i]^2 <= smax[i]^2)
            end
        end

        # collect shared (global) variables of the node
        shared_vars = Vector{VariableRef}(undef, 8 * length(lines_in_cut))
        for i in eachindex(lines_in_cut)
            line = lines_in_cut[i]
            shared_vars[8*(i-1)+1] = W[line[1], line[2]]
            shared_vars[8*(i-1)+2] = W[line[1], line[2]+N]
            shared_vars[8*(i-1)+3] = W[line[1]+N, line[2]]
            shared_vars[8*(i-1)+4] = W[line[1]+N, line[2]+N]
            shared_vars[8*(i-1)+5] = plf[line]
            shared_vars[8*(i-1)+6] = plt[line]
            shared_vars[8*(i-1)+7] = qlf[line]
            shared_vars[8*(i-1)+8] = qlt[line]
        end

        shared_vars_dict[k] = shared_vars

        # objective function
        if gen_cost_type == 2
            @objective(nodes[k], Min, sum(sum(costs[i][j] * pg[i]^(length(costs[i])-j) for j in 1:length(costs[i])) for i in gen_idx_k)
                                    - sum(λs[k] .* shared_vars) )
        end
    end

    # global variables
    @variable(global_node, wrr[cut_lines])
    @variable(global_node, wri[cut_lines])
    @variable(global_node, wir[cut_lines])
    @variable(global_node, wii[cut_lines])
    @variable(global_node, plf[cut_lines])
    @variable(global_node, qlf[cut_lines])
    @variable(global_node, plt[cut_lines])
    @variable(global_node, qlt[cut_lines])

    for i in cut_lines
        for k in 1:N_partitions
            if i in L_gs[k]
                @linkconstraint(dm, global_node[:wrr][i] == nodes[k][:W][i[1], i[2]])
                @linkconstraint(dm, global_node[:wri][i] == nodes[k][:W][i[1], i[2]+N])
                @linkconstraint(dm, global_node[:wir][i] == nodes[k][:W][i[1]+N, i[2]])
                @linkconstraint(dm, global_node[:wii][i] == nodes[k][:W][i[1]+N, i[2]+N])
                @linkconstraint(dm, global_node[:plf][i] == nodes[k][:plf][i])
                @linkconstraint(dm, global_node[:plt][i] == nodes[k][:plt][i])
                @linkconstraint(dm, global_node[:qlf][i] == nodes[k][:qlf][i])
                @linkconstraint(dm, global_node[:qlt][i] == nodes[k][:qlt][i])
            end
        end
    end

    return dm, shared_vars_dict
end

# Main code
file = "case5.m"
data = parse_matpower(file)
pm = instantiate_model(file, ACRPowerModel, PowerModels.build_opf)

N = length(data["bus"])
L = length(data["branch"])

# collect node data
Vmin = [data["bus"]["$(i)"]["vmin"] for i in 1:N]
Vmax = [data["bus"]["$(i)"]["vmax"] for i in 1:N]

# collect load data
N_load = length(data["load"])
load_bus = [data["load"]["$(i)"]["load_bus"] for i in 1:N_load]
Pd = [data["load"]["$(i)"]["pd"] for i in 1:N_load]
Qd = [data["load"]["$(i)"]["qd"] for i in 1:N_load]

# collect generator data
N_gen = length(data["gen"])
gen_bus = [data["gen"]["$(i)"]["gen_bus"] for i in 1:N_gen]
Pmax = [data["gen"]["$(i)"]["pmax"] for i in 1:N_gen]
Qmax = [data["gen"]["$(i)"]["qmax"] for i in 1:N_gen]
Pmin = [data["gen"]["$(i)"]["pmin"] for i in 1:N_gen]
Qmin = [data["gen"]["$(i)"]["qmin"] for i in 1:N_gen]

# collect generator cost data
gen_cost_type = data["gen"]["1"]["model"] # assume all generators have the same cost model type
if gen_cost_type == 2
    # neglect startup/shutdown costs
    costs = [data["gen"]["$(i)"]["cost"] for i in 1:N_gen]
end

# collect shunt data
N_shunt = length(data["shunt"])
shunt_node = Int64[data["shunt"]["$(i)"]["shunt_bus"] for i in 1:N_shunt]
gs = Dict()
bs = Dict()
for i in 1:N_shunt
    gs[shunt_node[i]] = data["shunt"]["$(i)"]["gs"]
    bs[shunt_node[i]] = data["shunt"]["$(i)"]["bs"]
end

# Partition data
# ieee case 9
#N_gs = [[1, 2, 4, 8, 9], [3, 5, 6, 7]]
#N_gs = [[1,4,9],[3,5,6],[2,7,8]]
N_gs = [[1,5],[2,3,4]]
N_partitions = length(N_gs)
lines = [(data["branch"]["$(i)"]["f_bus"], data["branch"]["$(i)"]["t_bus"]) for i in 1:L]
L_gs = [[i for i in lines if i[1] in N_g || i[2] in N_g] for N_g in N_gs]
cut_lines = [i for i in lines if sum(i in L_gs[j] for j in 1:length(L_gs)) >= 2]
smax = Dict()
for i in 1:L
    if "rate_a" in keys(data["branch"]["$(i)"])
        smax[(data["branch"]["$(i)"]["f_bus"], data["branch"]["$(i)"]["t_bus"])] = data["branch"]["$(i)"]["rate_a"]
    end
end


# run the subgradient algorithm
# TODO: implement subgradient method
λ_dims = [length(intersect(i, cut_lines))*8 for i in L_gs]
α = 0.5 # fixed step size for now
lambdas = [zeros(Float64, i) for i in λ_dims]
d_lambda_norm = 1000
itr_count = 1
max_itr = 1
mg_and_dict = ()
solve_times = []
lambda_norms = []
obj_vals = zeros(max_itr)

global mg_and_dict = build_subgraph_model(N_gs, L_gs, cut_lines, load_bus, Pd, Qd, gen_bus,
                                            Pmax, Pmin, Qmax, Qmin, gen_cost_type, costs,
                                            shunt_node, gs, bs, smax, pm.model, lambdas)
#=
while d_lambda_norm > 1e-6 && itr_count <= max_itr
    println("================ Iteration $(itr_count) ================")
    global mg_and_dict = build_subgraph_model(N_gs, L_gs, cut_lines, load_bus, Pd, Qd, gen_bus,
                                                Pmax, Pmin, Qmax, Qmin, gen_cost_type, costs,
                                                shunt_node, gs, bs, smax, pm.model, lambdas)
    mg, shared_vars_dict = mg_and_dict
    # for node_idx in 1:length(mg.modelnodes)-1
    #     node = mg.modelnodes[node_idx]
    for node in mg.modelnodes
        set_start_value.(all_variables(node.model), 1)
        JuMP.set_optimizer(node.model, Ipopt.Optimizer)
        optimize!(node.model)
        t1 = solve_time(node.model)

        start_vals = value.(all_variables(node.model))
        JuMP.set_optimizer(node.model, Gurobi.Optimizer)
        set_optimizer_attributes(node.model, "NonConvex" => 2)
        set_start_value.(all_variables(node.model), start_vals)
        optimize!(node.model)
        t2 = solve_time(node.model)
        append!(solve_times, t1 + t2)
        global obj_vals[itr_count] += objective_value(node.model)
    end
    new_lambdas = [lambdas[k] - α * value.(shared_vars_dict[k]) for k in 1:N_partitions]
    # Projection for new_lambdas
    for line in cut_lines
        f_bus = line[1]
        t_bus = line[2]
        k_f = 0
        k_t = 0
        f_idx_in_shared_var_list = 0
        t_idx_in_shared_var_list = 0
        for i in eachindex(N_gs)
            if f_bus in N_gs[i]
                k_f = i
                lines_in_cut = intersect(L_gs[i], cut_lines)
                f_idx_in_shared_var_list = first(findall(x->x==line, lines_in_cut))
            end
            if t_bus in N_gs[i]
                k_t = i
                lines_in_cut = intersect(L_gs[i], cut_lines)
                t_idx_in_shared_var_list = first(findall(x->x==line, lines_in_cut))
            end
            if k_f != 0 && k_t != 0
                break
            end
        end
        for i in 1:8
            a = new_lambdas[k_f][8*(f_idx_in_shared_var_list-1)+i]
            b = new_lambdas[k_t][8*(t_idx_in_shared_var_list-1)+i]
            new_lambdas[k_f][8*(f_idx_in_shared_var_list-1)+i] = (a - b) / 2
            new_lambdas[k_t][8*(t_idx_in_shared_var_list-1)+i] = (b - a) / 2
        end
    end
    d_lambda_norm = norm(vcat(lambdas...) .- vcat(new_lambdas...))
    append!(lambda_norms, d_lambda_norm)
    println("Δλ l2 norm: $(d_lambda_norm)")
    global lambdas = new_lambdas
    global itr_count += 1
end
println("================ End of Solution Process ================")
=#
