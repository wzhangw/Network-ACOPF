using PowerModels, JuMP, Ipopt, Gurobi, Plasmo, LinearAlgebra, MathOptInterface, DataStructures
const MOI = MathOptInterface

using Random
Random.seed!(0)

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
        @variable(nodes[k], gen_cost)

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
        @constraint(nodes[k], gen_cost == sum(sum(costs[i][j] * pg[i]^(length(costs[i])-j) for j in 1:length(costs[i])) for i in gen_idx_k))
        if gen_cost_type == 2
            @objective(nodes[k], Min, sum(sum(costs[i][j] * pg[i]^(length(costs[i])-j) for j in 1:length(costs[i])) for i in gen_idx_k)
                                    - sum(λs[k] .* shared_vars) )
        end
    end

    return dm, shared_vars_dict
end

function lambda_projection(lambda::Vector{Vector{Float64}})
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
            a = lambda[k_f][8*(f_idx_in_shared_var_list-1)+i]
            b = lambda[k_t][8*(t_idx_in_shared_var_list-1)+i]
            lambda[k_f][8*(f_idx_in_shared_var_list-1)+i] = (a - b) / 2
            lambda[k_t][8*(t_idx_in_shared_var_list-1)+i] = (b - a) / 2
        end
    end
    return
end

# Main code
file = "case9.m"
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
N_gs = [[1,4,9],[3,5,6],[2,7,8]]
#N_gs = [[2, 3, 4], [1, 5]]
lines = [(data["branch"]["$(i)"]["f_bus"], data["branch"]["$(i)"]["t_bus"]) for i in 1:L]
L_gs = [[i for i in lines if i[1] in N_g || i[2] in N_g] for N_g in N_gs]
cut_lines = [i for i in lines if sum(i in L_gs[j] for j in 1:length(L_gs)) >= 2]
smax = Dict()
for i in 1:L
    if "rate_a" in keys(data["branch"]["$(i)"])
        smax[(data["branch"]["$(i)"]["f_bus"], data["branch"]["$(i)"]["t_bus"])] = data["branch"]["$(i)"]["rate_a"]
    end
end

λ_dims = [length(intersect(i, cut_lines))*8 for i in L_gs]
#lambdas = [zeros(Float64, i) for i in λ_dims]
#lambdas = [ones(Float64, i) for i in λ_dims]
lambdas = [ones(Float64, 16), vcat(-ones(Float64, 8), ones(Float64, 8)), -ones(Float64, 16)]
mg, shared_vars_dict = build_subgraph_model(N_gs, L_gs, cut_lines, load_bus, Pd, Qd, gen_bus,
                                            Pmax, Pmin, Qmax, Qmin, gen_cost_type, costs,
                                            shunt_node, gs, bs, smax, pm.model, lambdas)
lag_mp = Model(Gurobi.Optimizer)
lag_θ = @variable(lag_mp, θ[1:length(N_gs)])
lag_λ = [@variable(lag_mp, [eachindex(shared_vars_dict[i])], base_name = "λ$(i)") for i in eachindex(N_gs)]
@objective(lag_mp, Max, sum(lag_θ))
solve_times = []
obj_vals = zeros(max_itr)
another_lambdas = copy(lambdas)
stop_statuses = SortedDict()
pg_vals = SortedDict()
λs = Dict()

curr_obj_vals = []
for node in mg.modelnodes
    JuMP.set_optimizer(node.model, Gurobi.Optimizer)
    set_optimizer_attributes(node.model, "NonConvex" => 2)
    optimize!(node.model)
    append!(curr_obj_vals, objective_value(node.model))
end

ϵ = 0.1
lambdas_pos = [ϵ * value.(shared_vars_dict[i]) + lambdas[i] for i in 1:length(N_gs)]
lambdas_neg = [-ϵ * value.(shared_vars_dict[i]) + lambdas[i] for i in 1:length(N_gs)]
lambda_projection(lambdas_pos)
lambda_projection(lambdas_neg)

obj_vals_pos = []
mg, shared_vars_dict = build_subgraph_model(N_gs, L_gs, cut_lines, load_bus, Pd, Qd, gen_bus,
                                            Pmax, Pmin, Qmax, Qmin, gen_cost_type, costs,
                                            shunt_node, gs, bs, smax, pm.model, lambdas_pos)
for node in mg.modelnodes
    JuMP.set_optimizer(node.model, Gurobi.Optimizer)
    set_optimizer_attributes(node.model, "NonConvex" => 2)
    optimize!(node.model)
    append!(obj_vals_pos, objective_value(node.model))
end

obj_vals_neg = []
mg, shared_vars_dict = build_subgraph_model(N_gs, L_gs, cut_lines, load_bus, Pd, Qd, gen_bus,
                                            Pmax, Pmin, Qmax, Qmin, gen_cost_type, costs,
                                            shunt_node, gs, bs, smax, pm.model, lambdas_neg)
for node in mg.modelnodes
    JuMP.set_optimizer(node.model, Gurobi.Optimizer)
    set_optimizer_attributes(node.model, "NonConvex" => 2)
    optimize!(node.model)
    append!(obj_vals_neg, objective_value(node.model))
end
