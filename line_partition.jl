# second version of model code: aims to incorporate multiple lines between nodes,
# and arbitrary indexing of buses

using PowerModels, JuMP, Ipopt, Gurobi, Plasmo, LinearAlgebra, MathOptInterface, DataStructures, SCIP, Juniper
using DualDecomposition, BundleMethod
using HDF5, JLD
const DD = DualDecomposition
const BM = BundleMethod

using Random
Random.seed!(0)

include("./partition/spec_cluster.jl")

nl_optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
# nl_optimizer = optimizer_with_attributes(Gurobi.Optimizer, "NonConvex" => 2)
optimizer = optimizer_with_attributes(Juniper.Optimizer, "nl_solver" => nl_optimizer)

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
    cut_nodes::Dict,
    virtual_lines::Dict,
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
        gen_idx_k = [i for i in eachindex(gen_bus) if gen_bus[i] in N_gs[k]] # index of generators in partition k
        relevant_idx = N_gs[k]
        expanded_idx = union(relevant_idx, relevant_idx .+ N)

        @variable(nodes[k], W[expanded_idx, expanded_idx])
        @variable(nodes[k], v[expanded_idx])
        @variable(nodes[k], plf[L_gs[k]])
        @variable(nodes[k], plt[L_gs[k]])
        @variable(nodes[k], qlf[L_gs[k]])
        @variable(nodes[k], qlt[L_gs[k]])
        @variable(nodes[k], Pmin[i] <= pg[i in gen_idx_k] <= Pmax[i])
        @variable(nodes[k], Qmin[i] <= qg[i in gen_idx_k] <= Qmax[i])
        pv = Dict()
        qv = Dict()
        for i in cut_nodes[k]
            pv[i] = @variable(nodes[k], [virtual_lines[i]], base_name = "pv_$(i)")
            qv[i] = @variable(nodes[k], [virtual_lines[i]], base_name = "qv_$(i)")
        end

        for i in expanded_idx, j in expanded_idx
            @constraint(nodes[k], W[i,j] == v[i] * v[j])
        end

        # constraints 1d, 1e for each node
        for i in N_gs[k]
            real_expr = @expression(nodes[k], sum(plf[j] for j in L_gs[k] if lines[j][1] == i)
                         + sum(plt[j] for j in L_gs[k] if lines[j][2] == i)
                         - sum(pg[j] for j in 1:N_gen if gen_bus[j] == i)
                         + sum(Pd[j] for j in 1:N_load if load_bus[j] == i))
            reac_expr = @expression(nodes[k], sum(qlf[j] for j in L_gs[k] if lines[j][1] == i)
                         + sum(qlt[j] for j in L_gs[k] if lines[j][2] == i)
                         - sum(qg[j] for j in 1:N_gen if gen_bus[j] == i)
                         + sum(Qd[j] for j in 1:N_load if load_bus[j] == i))
            if i in cut_nodes[k]
                real_expr += @expression(nodes[k], -sum(pv[i][j] for j in virtual_lines[i] if j[1] == k)
                                                   +sum(pv[i][j] for j in virtual_lines[i] if j[2] == k))
                reac_expr += @expression(nodes[k], -sum(qv[i][j] for j in virtual_lines[i] if j[1] == k)
                                                   +sum(qv[i][j] for j in virtual_lines[i] if j[2] == k))
            end
            if i in shunt_node
                real_expr -= gs[i] * (W[i,i] + W[i+N,i+N])
                reac_expr -= bs[i] * (W[i,i] + W[i+N,i+N])
            end
            @constraint(nodes[k], real_expr == 0)
            @constraint(nodes[k], reac_expr == 0)
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
        for i in relevant_idx
            @constraint(nodes[k], Vmin[bus_idx_dict[i]]^2 <= W[i,i] + W[i+N, i+N] <= Vmax[bus_idx_dict[i]]^2)
        end

        # constraints 1h
        for i in L_gs[k]
            if i in keys(smax)
                @constraint(nodes[k], plf[i]^2 + qlf[i]^2 <= smax[i]^2)
                @constraint(nodes[k], plt[i]^2 + qlt[i]^2 <= smax[i]^2)
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
        shared_vars = Dict{String, Dict}()
        shared_vars["W"] = Dict()
        shared_vars["lines"] = Dict()
        shared_vars["gens"] = Dict()
        for i in cut_nodes[k]
            shared_vars["W"][i] = [W[i,i], W[i+N, i+N]]
            for j in virtual_lines[i]
                shared_vars["lines"][(i,j)] = [pv[i][j], qv[i][j]]
            end
            idx = findall(x->gen_bus[x]==i, gen_idx_k)
            for j in idx
                shared_vars["gens"][gen_idx_k[j]] = [pg[gen_idx_k[j]], qg[gen_idx_k[j]]]
            end
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

# Main code
# file = "../../matpower/data/case30.m"
# file = "../../pglib-opf/api/pglib_opf_case14_ieee__api.m"
# file = "../../pglib-opf/sad/pglib_opf_case14_ieee__sad.m"
# file = "../../pglib-opf/api/pglib_opf_case30_as__api.m"
file = "case5.m"
# file = "../../pglib-opf/api/pglib_opf_case5_pjm__api.m"
# file = "../../pglib-opf/api/pglib_opf_case24_ieee_rts__api.m"
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
lines = Dict(string(i) => (data["branch"]["$(i)"]["f_bus"], data["branch"]["$(i)"]["t_bus"]) for i in 1:L)
# L_gs = [[string(i)] for i in eachindex(lines)]
L_gs = [["1", "2", "3"], ["4", "5", "6"]]
N_gs = [union([data["branch"][i]["f_bus"] for i in L], [data["branch"][i]["t_bus"] for i in L]) for L in L_gs]

# cut_nodes maps partitions to cut nodes
# virtual links are virtual power flow between the same cut nodes distributed
# across different partitions
# virtual_lines maps cut nodes to virtual lines represented by pair of partitions
cut_nodes = Dict()
virtual_lines = Dict()
virtual_gens = Dict()
for i in eachindex(N_gs)
    cut_nodes[i] = filter(x->sum(in.(x, N_gs)) > 1, N_gs[i])
end
all_cut_nodes = union(values(cut_nodes)...)
for i in all_cut_nodes
    in_partition = in.(i, N_gs)
    partitions = findall(x->x, in_partition)
    virtual_lines[i] = []
    for j in 1:length(partitions), k in (j+1):length(partitions)
        push!(virtual_lines[i], (partitions[j], partitions[k]))
    end
end

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
    build_subgraph_model(N_gs, lines, L_gs, cut_nodes, virtual_lines, load_bus, Pd, Qd, gen_bus,
                         Pmax, Pmin, Qmax, Qmin, gen_cost_type, costs,
                         shunt_node, gs, bs, smax, angmin, angmax, pm.model)

algo = DD.LagrangeDual(BM.TrustRegionMethod)
for i in eachindex(N_gs)
    DD.add_block_model!(algo, i, mg.modelnodes[i].model)
end

coupling_vars = Vector{DD.CouplingVariableRef}()
for i in eachindex(N_gs)
    vars_dict = shared_vars_dict[i]
    for node in keys(vars_dict["W"])
        id1 = "Wrr_$(node)"
        id2 = "Wii_$(node)"
        push!(coupling_vars, DD.CouplingVariableRef(i, id1, vars_dict["W"][node][1]))
        push!(coupling_vars, DD.CouplingVariableRef(i, id2, vars_dict["W"][node][2]))
    end
    for (bus, vl) in keys(vars_dict["lines"])
        id1 = "pvl_$(bus)_$(vl[1])$(vl[2])"
        id2 = "qvl_$(bus)_$(vl[1])$(vl[2])"
        push!(coupling_vars, DD.CouplingVariableRef(i, id1, vars_dict["lines"][(bus, vl)][1]))
        push!(coupling_vars, DD.CouplingVariableRef(i, id2, vars_dict["lines"][(bus, vl)][2]))
    end
    for gen in keys(vars_dict["gens"])
        id1 = "pg_$(gen)"
        id2 = "qg_$(gen)"
        push!(coupling_vars, DD.CouplingVariableRef(i, id1, vars_dict["gens"][gen][1]))
        push!(coupling_vars, DD.CouplingVariableRef(i, id2, vars_dict["gens"][gen][2]))
    end
end

DD.set_coupling_variables!(algo, coupling_vars)
DD.run!(algo, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))

# Build DD model and algorithm
