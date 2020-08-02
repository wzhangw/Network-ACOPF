# second version of model code: aims to incorporate multiple lines between nodes,
# and arbitrary indexing of buses

using PowerModels, JuMP, Ipopt, Gurobi, LinearAlgebra, MathOptInterface, DataStructures, Juniper
using DualDecomposition, BundleMethod
using HDF5, JLD
const MOI = MathOptInterface
const DD = DualDecomposition
const BM = BundleMethod
const PM = PowerModels
using Random
Random.seed!(0)

include("./partition/spec_cluster.jl")

nl_optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
# nl_optimizer = optimizer_with_attributes(Gurobi.Optimizer, "NonConvex" => 2)
optimizer = optimizer_with_attributes(Juniper.Optimizer, "nl_solver" => nl_optimizer)

#=
function build_subgraph_model(data::Dict, N::Int, method::Function)
end
=#
function find_neighbor_buses(data::Dict{String, Any}, sub_buses::Dict{Int64, Any})::Set{Int64}
    N_g = collect(keys(sub_buses))
    neighbors = Set{Int64}()
    for line in keys(data["branch"])
        if data["branch"][line]["f_bus"] in N_g || data["branch"][line]["t_bus"] in N_g
            push!(neighbors, data["branch"][line]["f_bus"], data["branch"][line]["t_bus"])
        end
    end
    return setdiff!(neighbors, N_g)
end

function generate_subgraph_data(data::Dict{String, Any}, N_g::Set{Int64})
    sub_data = deepcopy(data)
    nbus = find_neighbor_buses(data, N_g)
    N = Set(parse.(Int, collect(keys(data["bus"]))))
    # deactivate unrelated buses
    for i in setdiff!(N, N_g)
        sub_data["bus"]["$(i)"]["bus_type"] = PM.pm_component_status_inactive["bus"]
    end
    # deactivate lines between neighbor buses
    #=
    for line in keys(data["branch"])
        if sub_data["branch"][line]["f_bus"] in nbus && data["branch"][line]["t_bus"] in nbus
            sub_data["branch"][line]["br_status"] = PM.pm_component_status_inactive["branch"]
        end
    end
    =#
    return sub_data
end

function build_subgraph_model(
    data::Dict{String, Any},
    N_gs::Vector{Set{Int64}},
    modeltype::Type{T},
    build_function::Function
    ) where T <: PM.AbstractPowerModel

    models = T[]
    for N_g in N_gs
        sub_data = generate_subgraph_data(data, N_g)
        push!(models, instantiate_model(sub_data, modeltype, build_function, ref_extensions = [ref_add_cut_branch!, ref_add_cut_bus_arcs_refs!]))
    end

    # need to have some way of tracking split variables
    # shared_vars_dict = collect_split_vars(models)
    # return models, shared_vars_dict
    return models
end

function ref_add_cut_bus!(ref::Dict{Symbol, Any}, data::Dict{String, Any})
    for (nw, nw_ref) in ref[:nw]
        # set up neighboring nodes
        nw_ref[:cut_bus] = Dict(i => data["bus"]["$(i)"] for i in find_neighbor_buses(data, nw_ref[:bus]))
    end
end

function ref_add_cut_branch!(ref::Dict{Symbol, Any}, data::Dict{String, Any})
    for (nw, nw_ref) in ref[:nw]
        # set up cut branches and arcs
        nw_ref[:cut_branch] = Dict(parse(Int, x.first) => x.second for x in data["branch"] if (x.second["f_bus"] in keys(nw_ref[:bus])) + (x.second["t_bus"] in keys(nw_ref[:bus])) == 1)
        nw_ref[:cut_arcs_from] = [(i,branch["f_bus"],branch["t_bus"]) for (i,branch) in nw_ref[:cut_branch]]
        nw_ref[:cut_arcs_to]   = [(i,branch["t_bus"],branch["f_bus"]) for (i,branch) in nw_ref[:cut_branch]]
        nw_ref[:cut_arcs] = [nw_ref[:cut_arcs_from]; nw_ref[:cut_arcs_to]]
    end
end

function ref_add_cut_bus_arcs_refs!(ref::Dict{Symbol, Any}, data::Dict{String, Any})
    for (nw, nw_ref) in ref[:nw]
        cut_bus_arcs = Dict((i, Tuple{Int,Int,Int}[]) for (i,bus) in merge(nw_ref[:bus], nw_ref[:cut_bus]))
        for (l,i,j) in nw_ref[:cut_arcs]
            push!(cut_bus_arcs[i], (l,i,j))
        end
        nw_ref[:cut_bus_arcs] = cut_bus_arcs
    end
end

function variable_cut_bus_voltage(pm::AbstractACRModel; nw::Int=pm.cnw, bounded::Bool=true, kwargs...)
    variable_cut_bus_voltage_real(pm; nw=nw, bounded=bounded, kwargs...)
    variable_cut_bus_voltage_imaginary(pm; nw=nw, bounded=bounded, kwargs...)

    if bounded
        for (i,bus) in ref(pm, nw, :cut_bus)
            constraint_cut_voltage_magnitude_bounds(pm, i, nw=nw)
        end
    end
end

function variable_cut_bus_voltage_real(pm::AbstractPowerModel; nw::Int=pm.cnw, bounded::Bool=true, report::Bool=true)
    vcr = var(pm, nw)[:vcr] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :cut_bus)], base_name="$(nw)_vcr",
        start = comp_start_value(ref(pm, nw, :bus, i), "vcr_start", 1.0)
    )

    if bounded
        for (i, bus) in ref(pm, nw, :cut_bus)
            JuMP.set_lower_bound(vcr[i], -bus["vmax"])
            JuMP.set_upper_bound(vcr[i],  bus["vmax"])
        end
    end
end

function variable_cut_bus_voltage_imaginary(pm::AbstractPowerModel; nw::Int=pm.cnw, bounded::Bool=true, report::Bool=true)
    vci = var(pm, nw)[:vci] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :cut_bus)], base_name="$(nw)_vci",
        start = comp_start_value(ref(pm, nw, :bus, i), "vci_start")
    )

    if bounded
        for (i, bus) in ref(pm, nw, :cut_bus)
            JuMP.set_lower_bound(vi[i], -bus["vmax"])
            JuMP.set_upper_bound(vi[i],  bus["vmax"])
        end
    end
end

function constraint_cut_voltage_magnitude_bounds(pm::AbstractACRModel, n::Int, i, vmin, vmax)
    @assert vmin <= vmax
    vcr = var(pm, n, :vcr, i)
    vci = var(pm, n, :vci, i)

    JuMP.@constraint(pm.model, vmin^2 <= (vcr^2 + vci^2))
    JuMP.@constraint(pm.model, vmax^2 >= (vcr^2 + vci^2))
end

function constraint_voltage_magnitude_bounds(pm::AbstractPowerModel, i::Int; nw::Int=pm.cnw)
    bus = ref(pm, nw, :cut_bus, i)
    constraint_voltage_magnitude_bounds(pm, nw, i, bus["vmin"], bus["vmax"])
end


function variable_split_lines(pm::AbstractPowerModel)
    variable_cut_branch_power_real(pm)
    variable_cut_branch_power_imaginary(pm)
end

function variable_cut_branch_power_real(pm::AbstractPowerModel; nw::Int=pm.cnw, bounded::Bool=true, report::Bool=true)
    pl = var(pm, nw)[:pl] = JuMP.@variable(pm.model,
        [(l,i,j) in ref(pm, nw, :cut_arcs)], base_name="$(nw)_pl",
        start = comp_start_value(ref(pm, nw, :branch, l), "pl_start")
    )

    if bounded
        flow_lb, flow_ub = ref_calc_branch_flow_bounds(ref(pm, nw, :cut_branch), ref(pm, nw, :bus))

        for arc in ref(pm, nw, :cut_arcs)
            l,i,j = arc
            if !isinf(flow_lb[l])
                JuMP.set_lower_bound(pl[arc], flow_lb[l])
            end
            if !isinf(flow_ub[l])
                JuMP.set_upper_bound(pl[arc], flow_ub[l])
            end
        end
    end

    for (l,branch) in ref(pm, nw, :cut_branch)
        if haskey(branch, "pf_start")
            f_idx = (l, branch["f_bus"], branch["t_bus"])
            JuMP.set_start_value(pl[f_idx], branch["pf_start"])
        end
        if haskey(branch, "pt_start")
            t_idx = (l, branch["t_bus"], branch["f_bus"])
            JuMP.set_start_value(pl[t_idx], branch["pt_start"])
        end
    end

    # report && PM._IM.sol_component_value_edge(pm, nw, :branch, :pf, :pt, ref(pm, nw, :arcs_from), ref(pm, nw, :arcs_to), p)
end


function variable_cut_branch_power_imaginary(pm::AbstractPowerModel; nw::Int=pm.cnw, bounded::Bool=true, report::Bool=true)
    ql = var(pm, nw)[:ql] = JuMP.@variable(pm.model,
        [(l,i,j) in ref(pm, nw, :cut_arcs)], base_name="$(nw)_ql",
        start = comp_start_value(ref(pm, nw, :branch, l), "ql_start")
    )

    if bounded
        flow_lb, flow_ub = ref_calc_branch_flow_bounds(ref(pm, nw, :cut_branch), ref(pm, nw, :bus))

        for arc in ref(pm, nw, :cut_arcs)
            l,i,j = arc
            if !isinf(flow_lb[l])
                JuMP.set_lower_bound(ql[arc], flow_lb[l])
            end
            if !isinf(flow_ub[l])
                JuMP.set_upper_bound(ql[arc], flow_ub[l])
            end
        end
    end

    for (l,branch) in ref(pm, nw, :cut_branch)
        if haskey(branch, "qf_start")
            f_idx = (l, branch["f_bus"], branch["t_bus"])
            JuMP.set_start_value(ql[f_idx], branch["qf_start"])
        end
        if haskey(branch, "qt_start")
            t_idx = (l, branch["t_bus"], branch["f_bus"])
            JuMP.set_start_value(ql[t_idx], branch["qt_start"])
        end
    end

    # report && _IM.sol_component_value_edge(pm, nw, :branch, :qf, :qt, ref(pm, nw, :arcs_from), ref(pm, nw, :arcs_to), q)
end

function build_acopf_with_free_lines(pm::AbstractPowerModel)
    # modified from original constraint_power_balance for ACRPowerModel
    function constraint_power_balance(pm::AbstractACRModel, n::Int, i::Int, bus_arcs, bus_arcs_dc, bus_arcs_sw, bus_gens, bus_storage, bus_pd, bus_qd, bus_gs, bus_bs)
        vr = var(pm, n, :vr, i)
        vi = var(pm, n, :vi, i)
        # vcr = var(pm, n, :vcr, i)
        # vci = var(pm, n, :vci, i)
        p    = get(var(pm, n),    :p, Dict()); _check_var_keys(p, bus_arcs, "active power", "branch")
        q    = get(var(pm, n),    :q, Dict()); _check_var_keys(q, bus_arcs, "reactive power", "branch")
        pg   = get(var(pm, n),   :pg, Dict()); _check_var_keys(pg, bus_gens, "active power", "generator")
        qg   = get(var(pm, n),   :qg, Dict()); _check_var_keys(qg, bus_gens, "reactive power", "generator")
        ps   = get(var(pm, n),   :ps, Dict()); _check_var_keys(ps, bus_storage, "active power", "storage")
        qs   = get(var(pm, n),   :qs, Dict()); _check_var_keys(qs, bus_storage, "reactive power", "storage")
        psw  = get(var(pm, n),  :psw, Dict()); _check_var_keys(psw, bus_arcs_sw, "active power", "switch")
        qsw  = get(var(pm, n),  :qsw, Dict()); _check_var_keys(qsw, bus_arcs_sw, "reactive power", "switch")
        p_dc = get(var(pm, n), :p_dc, Dict()); _check_var_keys(p_dc, bus_arcs_dc, "active power", "dcline")
        q_dc = get(var(pm, n), :q_dc, Dict()); _check_var_keys(q_dc, bus_arcs_dc, "reactive power", "dcline")

        cstr_p = JuMP.@constraint(pm.model,
            sum(p[a] for a in bus_arcs)
            + sum(p_dc[a_dc] for a_dc in bus_arcs_dc)
            + sum(psw[a_sw] for a_sw in bus_arcs_sw)
            ==
            sum(pg[g] for g in bus_gens)
            - sum(ps[s] for s in bus_storage)
            - sum(pd for pd in values(bus_pd))
            - sum(gs for gs in values(bus_gs))*(vr^2 + vi^2)
        )
        cstr_q = JuMP.@constraint(pm.model,
            sum(q[a] for a in bus_arcs)
            + sum(q_dc[a_dc] for a_dc in bus_arcs_dc)
            + sum(qsw[a_sw] for a_sw in bus_arcs_sw)
            ==
            sum(qg[g] for g in bus_gens)
            - sum(qs[s] for s in bus_storage)
            - sum(qd for qd in values(bus_qd))
            + sum(bs for bs in values(bus_bs))*(vr^2 + vi^2)
        )

        if PM._IM.report_duals(pm)
            sol(pm, n, :bus, i)[:lam_kcl_r] = cstr_p
            sol(pm, n, :bus, i)[:lam_kcl_i] = cstr_q
        end
    end

    # before calling build_opf, add split variables here
    variable_cut_bus_voltage(pm)
    variable_split_lines(pm)

    # call build_opf with custom constraint_power_balance, which now
    # will take care of split variables
    build_opf(pm)

    # this is for adding w variables to the model
    # variable_bus_voltage_magnitude_sqr(pm)
    # variable_buspair_voltage_product(pm)

    w  = var(pm,  :w)
    vr = var(pm, :vr)
    vi = var(pm, :vi)

    for (i, bus) in ref(pm, :bus)
        JuMP.@constraint(pm.model, w[i] == vr[i]^2 + vi[i]^2)
    end



end

# Main code
#=
# file = "../../matpower/data/case30.m"
# file = "../pglib-opf/api/pglib_opf_case14_ieee__api.m"
# file = "../../pglib-opf/sad/pglib_opf_case14_ieee__sad.m"
# file = "../pglib-opf/api/pglib_opf_case30_as__api.m"
file = "case5.m"
# file = "../pglib-opf/api/pglib_opf_case5_pjm__api.m"
# file = "../pglib-opf/api/pglib_opf_case24_ieee_rts__api.m"
data = parse_file(file)

# Partition data
# ieee case 9
# N_gs = [[i] for i in buses]
# N_gs = [[1, 2, 4, 8, 9], [3, 5, 6, 7]]
N_gs = [Set((2, 3, 4)), Set((1, 5))]
# N_gs = [[2, 3], [1, 4, 5]] # bad iteration
# N_gs = [[1,4,9],[3,5,6],[2,7,8]]
# N_gs = [[1,4,100],[3,5,6],[2,7,8]]
# N_gs = [[1,2,3,4,5],[7,8,9,10,14],[6,11,12,13]]
# N_gs = [[1,4,9],[3,5,6,7],[2,8]]
# N_gs = [[17,18,21,22],[3,24,15,16,19],[12,13,20,23],[9,11,14],[7,8],[10,5,6],[1,2,4]]
# N_gs = [[1,2,3,4,5,6,7],[9,10,11,21,22],[12,13,16,17],[14,15,18,19,20],[23,24,25,26],[27,29,30,8,28]]
# N_gs = compute_cluster(file, 3)

models = build_subgraph_model(data, N_gs, ACRPowerModel, PM.build_opf)

#=
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
=#

# Build DD model and algorithm
=#

data = parse_file("case5.m")
sub_data = generate_subgraph_data(data, Set((2,3)))
# sub_pm = instantiate_model(sub_data, ACRPowerModel, build_acopf_with_free_lines, ref_extensions=[ref_add_cut_bus!, ref_add_cut_branch!, ref_add_cut_bus_arcs_refs!])
sub_pm = instantiate_model(sub_data, ACRPowerModel, build_opf, ref_extensions=[ref_add_cut_bus!, ref_add_cut_branch!, ref_add_cut_bus_arcs_refs!])
