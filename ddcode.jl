# second version of model code: aims to incorporate multiple lines between nodes,
# and arbitrary indexing of buses

using PowerModels, JuMP, Ipopt, Gurobi, Plasmo, LinearAlgebra, MathOptInterface, DataStructures, Juniper
using DualDecomposition, BundleMethod
using HDF5, JLD
const MOI = MathOptInterface
const DD = DualDecomposition
const BM = BundleMethod

using Random
Random.seed!(0)

include("./partition/spec_cluster.jl")

nl_optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
# nl_optimizer = optimizer_with_attributes(Gurobi.Optimizer, "NonConvex" => 2)
optimizer = optimizer_with_attributes(Juniper.Optimizer, "nl_solver" => nl_optimizer)

# customary trust region size update
# function BM.update_Δ_serious_step!(method::BM.TrustRegionMethod)
#     method.Δ = 1/4 * method.Δ + 3/4 * method.Δ_ub
# end
#
# function BM.update_Δ_null_step!(method::BM.TrustRegionMethod)
#     method.Δ = 3/4 * method.Δ + 1/4 * method.Δ_lb
# end

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
    pm_model::Model
    )

    N = maximum(maximum.(N_gs))
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
        for i in N_gs[k]
        # for i in relevant_idx
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
        shared_vars = Dict{String, Vector{VariableRef}}()
        for i in eachindex(lines_in_cut)
            line = lines_in_cut[i]
            f_bus = lines[line][1]
            t_bus = lines[line][2]
            shared_vars[line] = [
                W[f_bus, t_bus], W[f_bus, t_bus+N],
                W[f_bus+N, t_bus], W[f_bus+N, t_bus+N],
                plf[line], plt[line], qlf[line], qlt[line]
            ]
        end

        shared_vars_dict[k] = shared_vars

        # objective function
        if gen_cost_type == 2
            @objective(nodes[k].model, Min, sum(sum(costs[i][j] * pg[i]^(length(costs[i])-j) for j in 1:length(costs[i])) for i in gen_idx_k))
        end

        JuMP.set_optimizer(dm.modelnodes[k].model, nl_optimizer)
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
# file = "../pglib-opf/api/pglib_opf_case14_ieee__api.m"
# file = "../../pglib-opf/sad/pglib_opf_case14_ieee__sad.m"
# file = "../pglib-opf/api/pglib_opf_case30_as__api.m"
file = "case5.m"
# file = "../pglib-opf/api/pglib_opf_case5_pjm__api.m"
# file = "../pglib-opf/api/pglib_opf_case24_ieee_rts__api.m"
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
# N_gs = compute_cluster(file, 3)
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
mg, shared_vars_dict =
    build_subgraph_model(N_gs, lines, L_gs, cut_lines, load_bus, Pd, Qd, gen_bus,
                         Pmax, Pmin, Qmax, Qmin, gen_cost_type, costs,
                         shunt_node, gs, bs, smax, angmin, angmax, pm.model)

algo = DD.LagrangeDual(BM.TrustRegionMethod)
# algo = DD.LagrangeDual()
for i in eachindex(N_gs)
    DD.add_block_model!(algo, i, mg.modelnodes[i].model)
end

coupling_vars = Vector{DD.CouplingVariableRef}()
subname = ["Wrr", "Wri", "Wir", "Wii", "pf", "pt", "qf", "qt"]
for i in eachindex(N_gs)
    vars_dict = shared_vars_dict[i]
    for line in keys(vars_dict)
        for j in eachindex(vars_dict[line])
            id = line * "_" * subname[j]
            push!(coupling_vars, DD.CouplingVariableRef(i, id, vars_dict[line][j]))
        end
    end
end

DD.set_coupling_variables!(algo, coupling_vars)
DD.run!(algo, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))

# Build DD model and algorithm
