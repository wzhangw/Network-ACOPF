# second version of model code: aims to incorporate multiple lines between nodes,
# and arbitrary indexing of buses

using PowerModels, JuMP, Ipopt, Gurobi, Plasmo, LinearAlgebra, MathOptInterface, DataStructures, SCIP, Juniper
using HDF5, JLD
const MOI = MathOptInterface

using Random
Random.seed!(0)

include("../partition/spec_cluster.jl")

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
    lines::Dict,
    L_gs::Vector{Vector{String}},
    cut_lines::Vector{String},
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
    angmin::Dict,
    angmax::Dict,
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
        relevant_idx = union(N_gs[k], vcat([collect(lines[i]) for i in L_gs[k]]...))
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
                @constraint(nodes[k], sum(plf[j] for j in L_gs[k] if lines[j][1] == i)
                             + sum(plt[j] for j in L_gs[k] if lines[j][2] == i)
                             - sum(pg[j] for j in 1:N_gen if gen_bus[j] == i)
                             + sum(Pd[j] for j in 1:N_load if load_bus[j] == i)
                             - gs[i] * (W[i,i] + W[i+N,i+N]) == 0)
                @constraint(nodes[k], sum(qlf[j] for j in L_gs[k] if lines[j][1] == i)
                             + sum(qlt[j] for j in L_gs[k] if lines[j][2] == i)
                             - sum(qg[j] for j in 1:N_gen if gen_bus[j] == i)
                             + sum(Qd[j] for j in 1:N_load if load_bus[j] == i)
                             - bs[i] * (W[i,i] + W[i+N,i+N]) == 0)
            else
                @constraint(nodes[k], sum(plf[j] for j in L_gs[k] if lines[j][1] == i)
                             + sum(plt[j] for j in L_gs[k] if lines[j][2] == i)
                             - sum(pg[j] for j in 1:N_gen if gen_bus[j] == i)
                             + sum(Pd[j] for j in 1:N_load if load_bus[j] == i) == 0)

                @constraint(nodes[k], sum(qlf[j] for j in L_gs[k] if lines[j][1] == i)
                             + sum(qlt[j] for j in L_gs[k] if lines[j][2] == i)
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
                p_or_q = JuMP.name(var)[3]
                (line, f_bus, t_bus) = get_index_from_var_name(JuMP.name(var))
                line = string(line)
                if (f_bus, t_bus) == lines[line]
                    is_f = true
                else
                    is_f = false
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
            @constraint(nodes[k], Vmin[bus_idx_dict[i]]^2 <= W[i,i] + W[i+N, i+N] <= Vmax[bus_idx_dict[i]]^2)
            @constraint(nodes[k], -Vmax[bus_idx_dict[i]] <= v[i] <= Vmax[bus_idx_dict[i]]) # should be redundant
            @constraint(nodes[k], -Vmax[bus_idx_dict[i]] <= v[i+N] <= Vmax[bus_idx_dict[i]]) # should be redundant
        end

        # constraints 1h
        for i in L_gs[k]
            if i in keys(smax)
                @constraint(nodes[k], plf[i]^2 + qlf[i]^2 <= smax[i]^2)
                @constraint(nodes[k], plt[i]^2 + qlt[i]^2 <= smax[i]^2)
                @constraint(nodes[k], -smax[i] <= plf[i] <= smax[i]) # should be redundant
                @constraint(nodes[k], -smax[i] <= qlf[i] <= smax[i]) # should be redundant
                @constraint(nodes[k], -smax[i] <= plt[i] <= smax[i]) # should be redundant
                @constraint(nodes[k], -smax[i] <= qlt[i] <= smax[i]) # should be redundant
            end
        end

        # branch voltage angle difference bounds
        for i in L_gs[k]
            f_bus = lines[i][1]
            t_bus = lines[i][2]
            @constraint(nodes[k], W[f_bus+N, t_bus] - W[f_bus, t_bus+N] <= tan(angmax[i]) * (W[f_bus, t_bus] + W[f_bus+N, t_bus+N]))
            @constraint(nodes[k], W[f_bus+N, t_bus] - W[f_bus, t_bus+N] >= tan(angmin[i]) * (W[f_bus, t_bus] + W[f_bus+N, t_bus+N]))
        end

        # collect shared (global) variables of the node
        shared_vars = Vector{VariableRef}(undef, 8 * length(lines_in_cut))
        for i in eachindex(lines_in_cut)
            line = lines_in_cut[i]
            f_bus = lines[line][1]
            t_bus = lines[line][2]
            shared_vars[8*(i-1)+1] = W[f_bus, t_bus]
            shared_vars[8*(i-1)+2] = W[f_bus, t_bus+N]
            shared_vars[8*(i-1)+3] = W[f_bus+N, t_bus]
            shared_vars[8*(i-1)+4] = W[f_bus+N, t_bus+N]
            shared_vars[8*(i-1)+5] = plf[line]
            shared_vars[8*(i-1)+6] = plt[line]
            shared_vars[8*(i-1)+7] = qlf[line]
            shared_vars[8*(i-1)+8] = qlt[line]
        end
        # shared_vars = Vector{VariableRef}(undef, 4 * length(lines_in_cut))
        # for i in eachindex(lines_in_cut)
        #     line = lines_in_cut[i]
        #     f_bus = lines[line][1]
        #     t_bus = lines[line][2]
        #     shared_vars[4*(i-1)+1] = W[f_bus, t_bus]
        #     shared_vars[4*(i-1)+2] = W[f_bus, t_bus+N]
        #     shared_vars[4*(i-1)+3] = W[f_bus+N, t_bus]
        #     shared_vars[4*(i-1)+4] = W[f_bus+N, t_bus+N]
        # end

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

function build_tr_mp(
    model::Model,
    radius::Float64,
    center::Vector{Vector{Float64}}
    )
    tr_model = copy(model)
    for i in eachindex(center)
        base_name = "λ$(i)"
        center_vals = center[i]
        for j in eachindex(center_vals)
            var_name = base_name * "[$(j)]"
            vref = variable_by_name(tr_model, var_name)
            center_val = center_vals[j]
            @constraint(tr_model, vref - center_val >= -radius)
            @constraint(tr_model, vref - center_val <= radius)
        end
    end
    JuMP.set_optimizer(tr_model, gurobi_optimizer)
    return tr_model
end

function update_lambda_from_model!(lambdas::Vector{Vector{Float64}}, model::Model)::Nothing
    for i in eachindex(lambdas)
        base_name = "λ$(i)"
        for j in eachindex(lambdas[i])
            var_name = base_name * "[$(j)]"
            lambdas[i][j] = value(variable_by_name(model, var_name))
        end
    end
    return
end

# Main code
# file = "../../matpower/data/case30.m"
# file = "../../pglib-opf/api/pglib_opf_case14_ieee__api.m"
# file = "../../pglib-opf/sad/pglib_opf_case14_ieee__sad.m"
# file = "../../pglib-opf/api/pglib_opf_case30_as__api.m"
file = "case5.m"
# file = "../../pglib-opf/api/pglib_opf_case5_pjm__api.m"
# file = "/home/weiqizhang/anl/pglib-opf/api/pglib_opf_case24_ieee_rts__api.m"
data = parse_file(file)
pm = instantiate_model(file, ACRPowerModel, PowerModels.build_opf)

N = length(data["bus"])
buses = collect(keys(data["bus"]))
buses = sort([parse(Int, i) for i in buses])
bus_idx_dict = Dict(buses[i] => i for i in eachindex(buses))
L = length(data["branch"])

# collect node data
Vmin = [data["bus"]["$(i)"]["vmin"] for i in buses]
Vmax = [data["bus"]["$(i)"]["vmax"] for i in buses]

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
N_gs = [[i] for i in buses]
# N_gs = [[1, 2, 4, 8, 9], [3, 5, 6, 7]]
# N_gs = [[2, 3, 4], [1, 5]]
# N_gs = [[2, 3], [1, 4, 5]] # bad iteration
# N_gs = [[1,4,9],[3,5,6],[2,7,8]]
# N_gs = [[1,4,100],[3,5,6],[2,7,8]]
# N_gs = [[1,2,3,4,5],[7,8,9,10,14],[6,11,12,13]]
# N_gs = [[1,4,9],[3,5,6,7],[2,8]]
# N_gs = [[17,18,21,22],[3,24,15,16,19],[12,13,20,23],[9,11,14],[7,8],[10,5,6],[1,2,4]]
# N_gs = [[1,2,3,4,5,6,7],[9,10,11,21,22],[12,13,16,17],[14,15,18,19,20],[23,24,25,26],[27,29,30,8,28]]
# N_gs = compute_cluster(file, 12)
lines = Dict(string(i) => (data["branch"]["$(i)"]["f_bus"], data["branch"]["$(i)"]["t_bus"]) for i in 1:L)
# lines = [(data["branch"]["$(i)"]["f_bus"], data["branch"]["$(i)"]["t_bus"]) for i in 1:L]
# L_gs = [[i for i in lines if i[1] in N_g || i[2] in N_g] for N_g in N_gs]
L_gs = [[string(i) for i in eachindex(lines) if lines[i][1] in N_g || lines[i][2] in N_g] for N_g in N_gs]
cut_lines = [string(i) for i in eachindex(lines) if sum(string(i) in L_gs[j] for j in 1:length(L_gs)) >= 2]
smax = Dict()
for i in 1:L
    if "rate_a" in keys(data["branch"]["$(i)"])
        smax["$(i)"] = data["branch"]["$(i)"]["rate_a"]
    end
end
angmin = Dict()
angmax = Dict()
for i in 1:L
    angmin["$(i)"] = data["branch"]["$(i)"]["angmin"]
    angmax["$(i)"] = data["branch"]["$(i)"]["angmax"]
end

# run the bundle-trust-region algorithm
λ_dims = [length(intersect(i, cut_lines))*8 for i in L_gs]
# λ_dims = [length(intersect(i, cut_lines))*4 for i in L_gs]
lambdas = [zeros(Float64, i) for i in λ_dims]
tr_center = [zeros(Float64, i) for i in λ_dims]
itr_count = 1
max_itr = 1000
mg_and_dict = build_subgraph_model(N_gs, lines, L_gs, cut_lines, load_bus, Pd, Qd, gen_bus,
                                   Pmax, Pmin, Qmax, Qmin, gen_cost_type, costs,
                                   shunt_node, gs, bs, smax, angmin, angmax, pm.model, lambdas)
shared_vars_dict = mg_and_dict[2]
lag_mp = Model(Gurobi.Optimizer)
lag_θ = @variable(lag_mp, θ[1:length(N_gs)])
lag_λ = [@variable(lag_mp, [eachindex(shared_vars_dict[i])], base_name = "λ$(i)") for i in eachindex(N_gs)]
@objective(lag_mp, Max, sum(lag_θ))
solve_times = []
obj_vals = zeros(max_itr)
λs = Dict()

Δ_ub = 20
Δ_lb = 0
Δ = (Δ_ub + Δ_lb) / 2
ξ = 0.4
ϵ = 1e-6

# Δ_ub = 50
# Δ_lb = 0
# Δ = (Δ_ub + Δ_lb) / 2
# ξ = 0.4
# ϵ = 1e-12

# nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
# optimizer = optimizer_with_attributes(Juniper.Optimizer, "nl_solver" => nl_solver)
# optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
optimizer = optimizer_with_attributes(Gurobi.Optimizer, "NonConvex" => 2, "OutputFlag" => 0)
gurobi_optimizer = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

curr_obj_vals = []
mg, shared_vars_dict = mg_and_dict
for node in mg.modelnodes
    set_start_value.(all_variables(node.model), 1)
    JuMP.set_optimizer(node.model, optimizer)
    optimize!(node.model)

   # start_vals = value.(all_variables(node.model))
   # JuMP.set_optimizer(node.model, Gurobi.Optimizer)
   # set_optimizer_attributes(node.model, "NonConvex" => 2)
   # set_start_value.(all_variables(node.model), start_vals)
   # optimize!(node.model)
    append!(curr_obj_vals, objective_value(node.model))
end
major_obj_val = sum(curr_obj_vals)

for line in cut_lines
    f_bus = lines[line][1]
    t_bus = lines[line][2]
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
        @constraint(lag_mp, lag_λ[k_f][8*(f_idx_in_shared_var_list-1)+i] + lag_λ[k_t][8*(t_idx_in_shared_var_list-1)+i] == 0)
    end
    # for i in 1:4
    #     @constraint(lag_mp, lag_λ[k_f][4*(f_idx_in_shared_var_list-1)+i] + lag_λ[k_t][4*(t_idx_in_shared_var_list-1)+i] == 0)
    # end
end
for i in 1:length(N_gs)
    @constraint(lag_mp, lag_θ[i] <= curr_obj_vals[i] - sum( value.(shared_vars_dict[i]) .* (lag_λ[i] - lambdas[i]) ) )
end

history = Dict()
history["m_kl"] = zeros(max_itr)
history["D_kl"] = zeros(max_itr)
history["major_obj_val"] = zeros(max_itr)
history["step"] = ["" for i in 1:max_itr]
history["TR size"] = zeros(max_itr)
history["time"] = zeros(max_itr)
history["termination_status"] = Dict()
history["y_values"] = Dict()

while itr_count <= max_itr
    println("====================== Iteration $(itr_count) ======================")
    term_statuses = []
    history["TR size"][itr_count] = Δ
    tr_mp = build_tr_mp(lag_mp, Δ, tr_center)
    itr_time = 0
    optimize!(tr_mp)
    push!(term_statuses, raw_status(tr_mp))
    itr_time += solve_time(tr_mp)
    update_lambda_from_model!(lambdas, tr_mp)
    m_kl = objective_value(tr_mp)
    history["m_kl"][itr_count] = m_kl
    history["major_obj_val"][itr_count] = major_obj_val
    println("mkl: $(m_kl). Dk: $(major_obj_val)")
    if m_kl - major_obj_val <= ϵ * (1 + abs(major_obj_val))
        println("Termination is reached.")
        history["step"][itr_count] = "termination"
        save("history.jld", "history", history)
        break
    end
    global mg, shared_vars_dict = build_subgraph_model(N_gs, lines, L_gs, cut_lines, load_bus, Pd, Qd, gen_bus,
                                                Pmax, Pmin, Qmax, Qmin, gen_cost_type, costs,
                                                shunt_node, gs, bs, smax, angmin, angmax, pm.model, lambdas)
    curr_obj_vals = []
    for node in mg.modelnodes
        set_start_value.(all_variables(node.model), 1)
        JuMP.set_optimizer(node.model, optimizer)
        optimize!(node.model)
        while !(termination_status(node.model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED])
            # set_start_value.(all_variables(node.model), rand(Int) % 100)
            println("Subgraph not solved to local optimality. Restart with different initial values.")
            set_start_value.(all_variables(node.model), rand())
            JuMP.set_optimizer(node.model, optimizer)
            optimize!(node.model)
        end
        itr_time += solve_time(node.model)
        push!(term_statuses, raw_status(node.model))
        # start_vals = value.(all_variables(node.model))
        # JuMP.set_optimizer(node.model, Gurobi.Optimizer)
        # set_optimizer_attributes(node.model, "NonConvex" => 2)
        # set_start_value.(all_variables(node.model), start_vals)
        # optimize!(node.model)
        # itr_time += solve_time(node.model)
        push!(term_statuses, raw_status(node.model))
        append!(curr_obj_vals, objective_value(node.model))
    end
    history["time"][itr_count] = itr_time
    history["D_kl"][itr_count] = sum(curr_obj_vals)
    history["termination_status"][itr_count] = term_statuses
    history["y_values"][itr_count] = [value.(shared_vars_dict[i]) for i in eachindex(N_gs)]

    # solve subgraphs to local optimality multiple times
    # curr_obj_vals = []
    # term_statuses = []
    # itr_time = 0
    # y_vals = []
    # for i in eachindex(mg.modelnodes)
    #     node = mg.modelnodes[i]
    #     best_node_obj_val = Inf
    #     temp_term_status = ""
    #     temp_y_values = []
    #     for _ in 1:5
    #         set_start_value.(all_variables(node.model), rand())
    #         JuMP.set_optimizer(node.model, optimizer)
    #         optimize!(node.model)
    #         itr_time += solve_time(node.model)
    #         while termination_status(node.model) != LOCALLY_SOLVED
    #             # set_start_value.(all_variables(node.model), rand(Int) % 100)
    #             println("Subgraph not solved to local optimality. Restart with different initial values.")
    #             set_start_value.(all_variables(node.model), rand())
    #             JuMP.set_optimizer(node.model, optimizer)
    #             optimize!(node.model)
    #             itr_time += solve_time(node.model)
    #         end
    #         if objective_value(node.model) < best_node_obj_val
    #             best_node_obj_val = objective_value(node.model)
    #             temp_term_status = raw_status(node.model)
    #             temp_y_values = value.(shared_vars_dict[i])
    #         end
    #     end
    #     push!(term_statuses, raw_status(node.model))
    #     append!(curr_obj_vals, best_node_obj_val)
    #     push!(y_vals, temp_y_values)
    # end
    # history["time"][itr_count] = itr_time
    # history["D_kl"][itr_count] = sum(curr_obj_vals)
    # history["termination_status"][itr_count] = term_statuses
    # history["y_values"][itr_count] = y_vals

    if sum(curr_obj_vals) >= major_obj_val + ξ * (m_kl - major_obj_val)
        println("Take a serious step.")
        global tr_center = [copy(i) for i in lambdas]
        global major_obj_val = sum(curr_obj_vals)
        global Δ = (Δ + Δ_ub) / 2
        history["step"][itr_count] = "serious"
    else
        println("Take a null step.")
        for i in 1:length(N_gs)
            @constraint(lag_mp, lag_θ[i] <= curr_obj_vals[i] - sum( value.(shared_vars_dict[i]) .* (lag_λ[i] - lambdas[i]) ) )
        end
        global Δ = (Δ_lb + Δ) / 2
        history["step"][itr_count] = "null"
    end
    global itr_count += 1
    save("intermediary.jld", "Δ", Δ, "tr_center", tr_center)
    write_to_file(lag_mp, "lag_mp.mps")
    save("history.jld", "history", history)
end

#=
# Subproblems are too slow... (> 25 min for each)
ipopt_stop_count = copy(itr_count)
itr_count += 1
Δ = (Δ_ub + Δ_lb) / 2
while itr_count <= max_itr
    term_statuses = []
    history["TR size"][itr_count] = Δ
    tr_mp = build_tr_mp(lag_mp, Δ, tr_center)
    itr_time = 0
    optimize!(tr_mp)
    push!(term_statuses, raw_status(tr_mp))
    itr_time += solve_time(tr_mp)
    update_lambda_from_model!(lambdas, tr_mp)
    m_kl = objective_value(tr_mp)
    history["m_kl"][itr_count] = m_kl
    history["major_obj_val"][itr_count] = major_obj_val
    println("mkl: $(m_kl). Dk: $(major_obj_val)")
    # if m_kl - major_obj_val <= ϵ * (1 + abs(major_obj_val))
    #     history["step"][itr_count] = "termination"
    #     save("history.jld", "history", history)
    #     break
    # end
    global mg, shared_vars_dict = build_subgraph_model(N_gs, L_gs, cut_lines, load_bus, Pd, Qd, gen_bus,
                                                Pmax, Pmin, Qmax, Qmin, gen_cost_type, costs,
                                                shunt_node, gs, bs, smax, pm.model, lambdas)
    curr_obj_vals = []
    for node in mg.modelnodes
        set_start_value.(all_variables(node.model), 1)
        JuMP.set_optimizer(node.model, Ipopt.Optimizer)
        optimize!(node.model)
        while termination_status(node.model) != LOCALLY_SOLVED
            set_start_value.(all_variables(node.model), rand(Int) % 100)
            JuMP.set_optimizer(node.model, Ipopt.Optimizer)
            optimize!(node.model)
        end
        itr_time += solve_time(node.model)

        start_vals = value.(all_variables(node.model))
        JuMP.set_optimizer(node.model, Gurobi.Optimizer)
        set_optimizer_attributes(node.model, "NonConvex" => 2)
        set_start_value.(all_variables(node.model), start_vals)
        optimize!(node.model)
        itr_time += solve_time(node.model)
        push!(term_statuses, raw_status(node.model))
        append!(curr_obj_vals, objective_value(node.model))
    end
    history["time"][itr_count] = itr_time
    history["D_kl"][itr_count] = sum(curr_obj_vals)
    history["termination_status"][itr_count] = term_statuses
    history["y_values"][itr_count] = [value.(shared_vars_dict[i]) for i in eachindex(N_gs)]
    if sum(curr_obj_vals) >= major_obj_val + ξ * (m_kl - major_obj_val)
        global tr_center = [copy(i) for i in lambdas]
        global major_obj_val = sum(curr_obj_vals)
        global Δ = (Δ + Δ_ub) / 2
        history["step"][itr_count] = "serious"
    else
        for i in 1:length(N_gs)
            @constraint(lag_mp, lag_θ[i] <= curr_obj_vals[i] - sum( value.(shared_vars_dict[i]) .* (lag_λ[i] - lambdas[i]) ) )
        end
        global Δ = (Δ_lb + Δ) / 2
        history["step"][itr_count] = "null"
    end
    global itr_count += 1
    save("intermediary.jld", "Δ", Δ, "tr_center", tr_center)
    write_to_file(lag_mp, "lag_mp.mps")
    save("history.jld", "history", history)
end
=#

if findfirst(x->x=="", history["step"]) isa Nothing
    final_itr = length(history["step"])
else
    final_itr = findfirst(x->x=="", history["step"]) - 1
end
avg_time = sum(history["time"][1:final_itr]) / final_itr
