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

function find_neighbor_buses(data::Dict{String, Any}, N_g::Set{Int64})::Set{Int64}
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
    for i in setdiff!(N, N_g, nbus)
        sub_data["bus"]["$(i)"]["bus_type"] = PM.pm_component_status_inactive["bus"]
    end
    # deactivate lines between neighbor buses
    for line in keys(data["branch"])
        if sub_data["branch"][line]["f_bus"] in nbus && data["branch"][line]["t_bus"] in nbus
            sub_data["branch"][line]["br_status"] = PM.pm_component_status_inactive["branch"]
        end
    end
    # deactivate unrelated generators
    for (_, gen) in sub_data["gen"]
        if !(gen["gen_bus"] in N_g)
            gen["gen_status"] = PM.pm_component_status_inactive["gen"]
        end
    end
    sub_data["cut_bus"] = nbus
    return sub_data
end

function build_subgraph_model(
    data::Dict{String, Any},
    N_gs::Vector{Set{Int64}},
    modeltype::Type{T},
    build_function::Function;
    ) where T <: PM.AbstractPowerModel

    models = T[]
    shared_vars_dict = Dict{Int64, Any}()
    for i in eachindex(N_gs)
        N_g = N_gs[i]
        sub_data = generate_subgraph_data(data, N_g)
        push!(models, instantiate_model(sub_data, modeltype, build_function, ref_extensions = [ref_add_cut_bus!, ref_add_cut_branch!]))
        shared_vars_dict[i] = collect_split_vars(models[i])
    end

    return models, shared_vars_dict
end

function ref_add_cut_bus!(ref::Dict{Symbol, Any}, data::Dict{String, Any})
    for (nw, nw_ref) in ref[:nw]
        nw_ref[:cut_bus] = Dict(i => data["bus"]["$(i)"] for i in data["cut_bus"])
    end
end

function ref_add_cut_branch!(ref::Dict{Symbol, Any}, data::Dict{String, Any})
    for (nw, nw_ref) in ref[:nw]
        # set up cut branches and arcs
        nw_ref[:cut_branch] = Dict(parse(Int, x.first) => x.second for x in data["branch"] if
            (x.second["f_bus"] in keys(nw_ref[:bus])) + (x.second["t_bus"] in keys(nw_ref[:bus])) +
            (x.second["f_bus"] in keys(nw_ref[:cut_bus])) + (x.second["t_bus"] in keys(nw_ref[:cut_bus])) == 3)
        nw_ref[:cut_arcs_from] = [(i,branch["f_bus"],branch["t_bus"]) for (i,branch) in nw_ref[:cut_branch]]
        nw_ref[:cut_arcs_to]   = [(i,branch["t_bus"],branch["f_bus"]) for (i,branch) in nw_ref[:cut_branch]]
        nw_ref[:cut_arcs] = [nw_ref[:cut_arcs_from]; nw_ref[:cut_arcs_to]]
    end
end

function ref_add_cut_bus_arcs_refs!(ref::Dict{Symbol, Any}, data::Dict{String, Any})
    #=
    for (nw, nw_ref) in ref[:nw]
        cut_bus_arcs = Dict((i, Tuple{Int,Int,Int}[]) for (i,bus) in merge(nw_ref[:bus], nw_ref[:cut_bus]))
        for (l,i,j) in nw_ref[:cut_arcs]
            push!(cut_bus_arcs[i], (l,i,j))
        end
        nw_ref[:cut_bus_arcs] = cut_bus_arcs
    end
    =#
end

function build_acopf_with_free_lines(pm::AbstractPowerModel)
    # modified from original build_opf
    # get rid of ref buses and apply power balance to only nodes in partition
    function build_opf_mod(pm::AbstractPowerModel)
        variable_bus_voltage(pm)
        variable_gen_power(pm)
        variable_branch_power(pm)
        variable_dcline_power(pm)

        objective_min_fuel_and_flow_cost_mod(pm)

        constraint_model_voltage(pm)

        for i in setdiff(ids(pm, :bus), ids(pm, :cut_bus))
            constraint_power_balance(pm, i)
        end

        for i in ids(pm, :branch)
            constraint_ohms_yt_from(pm, i)
            constraint_ohms_yt_to(pm, i)

            constraint_voltage_angle_difference(pm, i)

            constraint_thermal_limit_from(pm, i)
            constraint_thermal_limit_to(pm, i)
        end

        for i in ids(pm, :dcline)
            constraint_dcline_power_losses(pm, i)
        end
    end

    # modified from original objective_min_fuel_and_flow_cost to allow for
    # the case with no generators
    function objective_min_fuel_and_flow_cost_mod(pm::AbstractPowerModel; kwargs...)
        model = check_cost_models(pm)
        if model == 1
            return objective_min_fuel_and_flow_cost_pwl(pm; kwargs...)
        elseif model == 2
            return objective_min_fuel_and_flow_cost_polynomial(pm; kwargs...)
        elseif model != nothing
            Memento.error(_LOGGER, "Only cost models of types 1 and 2 are supported at this time, given cost model type of $(model)")
        end
    end

    build_opf_mod(pm)

    # this is for adding w variables to the model
    variable_bus_voltage_magnitude_sqr(pm)
    variable_buspair_voltage_product(pm)

    w  = var(pm,  :w)
    wr = var(pm, :wr)
    wi = var(pm, :wi)
    vr = var(pm, :vr)
    vi = var(pm, :vi)

    for (i, bus) in ref(pm, :bus)
        JuMP.@constraint(pm.model, w[i] == vr[i]^2 + vi[i]^2)
    end

    for (_, branch) in ref(pm, :branch)
        fbus = branch["f_bus"]
        tbus = branch["t_bus"]
        JuMP.@constraint(pm.model, wr[(fbus, tbus)] == vr[fbus] * vr[tbus] + vi[fbus] * vi[tbus])
        JuMP.@constraint(pm.model, wi[(fbus, tbus)] == vi[fbus] * vr[tbus] - vr[fbus] * vi[tbus])
    end
end

function collect_split_vars(pm::AbstractPowerModel)
    wr = var(pm, :wr)
    wi = var(pm, :wi)
    p  = var(pm,  :p)
    q  = var(pm,  :q)

    shared_vars_dict = Dict{String, Dict{Tuple, VariableRef}}()
    shared_vars_dict["wr"] = Dict{Tuple{Int64, Int64}, VariableRef}()
    shared_vars_dict["wi"] = Dict{Tuple{Int64, Int64}, VariableRef}()
    shared_vars_dict["p"] = Dict{Tuple{Int64, Int64, Int64}, VariableRef}()
    shared_vars_dict["q"] = Dict{Tuple{Int64, Int64, Int64}, VariableRef}()

    cut_arcs_from = ref(pm, :cut_arcs_from)
    for (l,i,j) in cut_arcs_from
        if !((i,j) in keys(shared_vars_dict["wr"]))
            shared_vars_dict["wr"][(i,j)] = wr[(i,j)]
            shared_vars_dict["wi"][(i,j)] = wi[(i,j)]
        end
        shared_vars_dict["p"][(l,i,j)] = p[(l,i,j)]
        shared_vars_dict["p"][(l,j,i)] = p[(l,j,i)]
        shared_vars_dict["q"][(l,i,j)] = q[(l,i,j)]
        shared_vars_dict["q"][(l,j,i)] = q[(l,j,i)]
    end
    return shared_vars_dict
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

# Partition data
# ieee case 9
# N_gs = [[parse(Int, i)] for i in keys(data["bus"])]
# N_gs = [[1, 2, 4, 8, 9], [3, 5, 6, 7]]
N_gs = [[2,3,4], [1,5]]
# N_gs = [[2, 3], [1, 4, 5]] # bad iteration
# N_gs = [[1,4,9],[3,5,6],[2,7,8]]
# N_gs = [[1,4,100],[3,5,6],[2,7,8]]
# N_gs = [[1,2,3,4,5],[7,8,9,10,14],[6,11,12,13]]
# N_gs = [[1,4,9],[3,5,6,7],[2,8]]
# N_gs = [[17,18,21,22],[3,24,15,16,19],[12,13,20,23],[9,11,14],[7,8],[10,5,6],[1,2,4]]
# N_gs = [[1,2,3,4,5,6,7],[9,10,11,21,22],[12,13,16,17],[14,15,18,19,20],[23,24,25,26],[27,29,30,8,28]]
# N_gs = compute_cluster(file, 3)
N_gs = [Set(i) for i in N_gs]

models, shared_vars_dict = build_subgraph_model(data, N_gs, ACRPowerModel, build_acopf_with_free_lines)

algo = DD.LagrangeDual(BM.TrustRegionMethod)
# algo = DD.LagrangeDual()
for i in eachindex(N_gs)
    DD.add_block_model!(algo, i, models[i].model)
end

coupling_vars = Vector{DD.CouplingVariableRef}()
for i in eachindex(N_gs)
    vars_dict = shared_vars_dict[i]
    for (varname, dict) in vars_dict
        for (idx, vref) in dict
            id = varname * "_" * string(idx)
            push!(coupling_vars, DD.CouplingVariableRef(i, id, vref))
        end
    end
end

# set subnetwork optimizer
for i in eachindex(N_gs)
    JuMP.set_optimizer(models[i].model, nl_optimizer)
end

DD.set_coupling_variables!(algo, coupling_vars)
DD.run!(algo, optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))
