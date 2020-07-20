using PowerModels, JuMP, Ipopt, Gurobi, Plasmo, LinearAlgebra, MathOptInterface, DataStructures
using HDF5, JLD
const MOI = MathOptInterface

function find_generators(data::Dict{String, Any}, nodes::Vector{Int})
    return [i for i in keys(data["gen"]) if data["gen"][i]["gen_bus"] in nodes]
end

function find_cut_lines(data::Dict{String, Any}, nodes::Vector{Int})
    return [i for i in keys(data["branch"]) if (data["branch"][i]["f_bus"] in nodes) + (data["branch"][i]["t_bus"] in nodes) == 1]
end

function parse_pq_name(var::VariableRef)::Tuple
    str = JuMP.name(var)
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

function parse_v_name(v::VariableRef)
    name = JuMP.name(v)
    type = name[4]
    idx = parse(Int, name[6:end-1])
    return (type, idx)
end

# function collect_gen_bounds(data::Dict{String, Any})
#     return Dict(i => data["gen"][i]["pmax"] for i in keys(data["gen"])),
#            Dict(i => data["gen"][i]["pmin"] for i in keys(data["gen"])),
#            Dict(i => data["gen"][i]["qmax"] for i in keys(data["gen"])),
#            Dict(i => data["gen"][i]["qmin"] for i in keys(data["gen"]))
# end

function build_subgraph_model_from_file(
    data::Dict,
    pm::AbstractPowerModel,
    N_gs::Vector{Vector{Int64}},
    λs::Vector{Vector{Float64}}
    )

    # Pmax, Pmin, Qmax, Qmin = collect_gen_bounds(data)

    dm = ModelGraph()
    N_partitions = length(N_gs)
    @node(dm, nodes[1:N_partitions])
    shared_vars_dict = Dict()

    # collect shunt data
    N_shunt = length(data["shunt"])
    shunt_node = Int64[data["shunt"]["$(i)"]["shunt_bus"] for i in 1:N_shunt]
    gs = Dict()
    bs = Dict()
    for i in 1:N_shunt
        gs[shunt_node[i]] = data["shunt"]["$(i)"]["gs"]
        bs[shunt_node[i]] = data["shunt"]["$(i)"]["bs"]
    end

    # collect constraints
    cref_type_list = list_of_constraint_types(pm.model)
    crefs = Dict()
    for (i,j) in cref_type_list
        crefs[(i,j)] = all_constraints(pm.model, i, j)
    end

    lines = Dict(string(i) => (data["branch"]["$(i)"]["f_bus"], data["branch"]["$(i)"]["t_bus"]) for i in 1:length(data["branch"]))
    L_gs = [[string(i) for i in eachindex(lines) if lines[i][1] in N_g || lines[i][2] in N_g] for N_g in N_gs]

    for k in 1:N_partitions
        gen_idx_k = find_generators(data, N_gs[k])
        lines_in_cut = find_cut_lines(data, N_gs[k])
        buses = union(N_gs[k], vcat([[data["branch"][i]["f_bus"], data["branch"][i]["t_bus"]] for i in L_gs[k]]...) )
        buses = sort(buses) # this takes all buses, including "free" buses not in partition k
        buses_to_idx = Dict(buses[i] => i for i in eachindex(buses)) # buses index mapped to 1:length(buses)

        @variable(nodes[k], W[1:length(buses)*2, 1:length(buses)*2])
        for i in 1:length(buses), j in 1:length(buses)
            JuMP.set_name(W[i,j], "Wrr[$(buses[i]),$(buses[j])]")
            JuMP.set_name(W[i,j+length(buses)], "Wri[$(buses[i]),$(buses[j])]")
            JuMP.set_name(W[i+length(buses),j], "Wir[$(buses[i]),$(buses[j])]")
            JuMP.set_name(W[i+length(buses),j+length(buses)], "Wii[$(buses[i]),$(buses[j])]")
        end
        @variable(nodes[k], v[1:length(buses)*2])
        @variable(nodes[k], plf[L_gs[k]])
        @variable(nodes[k], plt[L_gs[k]])
        @variable(nodes[k], qlf[L_gs[k]])
        @variable(nodes[k], qlt[L_gs[k]])
        @variable(nodes[k], pg[gen_idx_k])
        @variable(nodes[k], qg[gen_idx_k])

        for i in 1:length(buses)*2, j in 1:length(buses)*2
            @constraint(nodes[k], W[i,j] == v[i] * v[j])
        end

        # balance constraints: more conenient to write our own
        for i in N_gs[k]
            if i in shunt_node
                @constraint(nodes[k], sum(plf[j] for j in L_gs[k] if lines[j][1] == i)
                             + sum(plt[j] for j in L_gs[k] if lines[j][2] == i)
                             - sum(pg[j] for j in gen_idx_k if data["gen"][j]["gen_bus"] == i)
                             + sum(j.second["pd"] for j in data["load"] if j.second["load_bus"] == i)
                             - gs[i] * (W[buses_to_idx[i],buses_to_idx[i]] + W[buses_to_idx[i]+length(buses),buses_to_idx[i]+length(buses)]) == 0)
                @constraint(nodes[k], sum(qlf[j] for j in L_gs[k] if lines[j][1] == i)
                             + sum(qlt[j] for j in L_gs[k] if lines[j][2] == i)
                             - sum(qg[j] for j in gen_idx_k if data["gen"][j]["gen_bus"] == i)
                             + sum(j.second["qd"] for j in data["load"] if j.second["load_bus"] == i)
                             - bs[i] * (W[buses_to_idx[i],buses_to_idx[i]] + W[buses_to_idx[i]+length(buses),buses_to_idx[i]+length(buses)]) == 0)
            else
                @constraint(nodes[k], sum(plf[j] for j in L_gs[k] if lines[j][1] == i)
                             + sum(plt[j] for j in L_gs[k] if lines[j][2] == i)
                             - sum(pg[j] for j in gen_idx_k if data["gen"][j]["gen_bus"] == i)
                             + sum(j.second["pd"] for j in data["load"] if j.second["load_bus"] == i) == 0)

                @constraint(nodes[k], sum(qlf[j] for j in L_gs[k] if lines[j][1] == i)
                             + sum(qlt[j] for j in L_gs[k] if lines[j][2] == i)
                             - sum(qg[j] for j in gen_idx_k if data["gen"][j]["gen_bus"] == i)
                             + sum(j.second["qd"] for j in data["load"] if j.second["load_bus"] == i) == 0)
            end
        end

        pm_p_vars = var(pm, :p)
        pm_q_vars = var(pm, :q)
        pm_vr_vars = var(pm, :vr)
        pm_vi_vars = var(pm, :vi)
        pm_pg_vars = var(pm, :pg)
        pm_qg_vars = var(pm, :qg)

        # constraints that define flow using W variables
        for cref in crefs[GenericQuadExpr{Float64,VariableRef}, MOI.EqualTo{Float64}]
            # find out which line variable (line_vref) is this for def
            fn = jump_function(constraint_object(cref))
            pq_var = first(keys(fn.aff.terms))
            line_idx, f_bus, t_bus = parse_pq_name(pq_var)
            if string(line_idx) in L_gs[k]
                is_f_var = false
                if lines[string(line_idx)] == (f_bus, t_bus)
                    is_f_var = true
                end
                if pq_var in pm_p_vars
                    is_f_var ? line_vref = plf[string(line_idx)] : line_vref = plt[string(line_idx)]
                else
                    is_f_var ? line_vref = qlf[string(line_idx)] : line_vref = qlt[string(line_idx)]
                end
                expr = 0
                # process quadratic terms
                for vpair in keys(fn.terms)
                    coeff = fn.terms[vpair]
                    type1, idx1 = parse_v_name(vpair.a)
                    type2, idx2 = parse_v_name(vpair.b)
                    expr += coeff * W[buses_to_idx[idx1] + (type1 == 'i') * length(buses), buses_to_idx[idx2] + (type2 == 'i') * length(buses)]
                end
                @constraint(nodes[k], expr + line_vref == 0)
            end
        end

        # quadratic inequality constraints (thermal line limits + voltage limits + angle difference limits)
        for cref in vcat(crefs[GenericQuadExpr{Float64,VariableRef}, MOI.GreaterThan{Float64}],
                         crefs[GenericQuadExpr{Float64,VariableRef}, MOI.LessThan{Float64}])
            fn = jump_function(constraint_object(cref))
            expr = 0
            for pair in keys(fn.terms)
                if pair.a in pm_vr_vars || pair.a in pm_vi_vars # this is a v pair
                    coeff = fn.terms[pair]
                    type1, idx1 = parse_v_name(pair.a)
                    type2, idx2 = parse_v_name(pair.b)
                    # if idx1 in N_gs[k] || idx2 in N_gs[k]
                    if idx1 in buses && idx2 in buses
                        expr += coeff * W[buses_to_idx[idx1] + (type1 == 'i') * length(buses), buses_to_idx[idx2] + (type2 == 'i') * length(buses)]
                    else
                        break
                    end
                else
                    coeff = fn.terms[pair] # this is a p/q pair
                    line_idx, f_bus, t_bus = parse_pq_name(pair.a)
                    if string(line_idx) in L_gs[k]
                        is_f_var = false
                        if lines[string(line_idx)] == (f_bus, t_bus)
                            is_f_var = true
                        end
                        if pair.a in pm_p_vars
                            is_f_var ? line_vref = plf[string(line_idx)] : line_vref = plt[string(line_idx)]
                        else
                            is_f_var ? line_vref = qlf[string(line_idx)] : line_vref = qlt[string(line_idx)]
                        end
                        expr += line_vref^2 * coeff
                    end
                end
            end
            if expr != 0
                set = constraint_object(cref).set
                @constraint(nodes[k], expr in set)
            end
        end

        # variable bounds
        for cref in vcat(crefs[VariableRef, MOI.GreaterThan{Float64}], crefs[VariableRef, MOI.LessThan{Float64}])
            vref = jump_function(constraint_object(cref))
            expr = 0
            if vref in pm_vr_vars
                _, idx = parse_v_name(vref)
                if idx in N_gs[k]
                # if idx in buses
                    expr = v[buses_to_idx[idx]]
                end
            elseif vref in pm_vi_vars
                _, idx = parse_v_name(vref)
                if idx in N_gs[k]
                # if idx in buses
                    expr = v[buses_to_idx[idx] + length(buses)]
                end
            elseif vref in pm_pg_vars
                _, idx = parse_v_name(vref)
                if "$(idx)" in gen_idx_k
                    expr = pg["$(idx)"]
                end
            elseif vref in pm_qg_vars
                _, idx = parse_v_name(vref)
                if "$(idx)" in gen_idx_k
                    expr = qg["$(idx)"]
                end
            elseif vref in pm_p_vars
                line_idx, f_bus, t_bus = parse_pq_name(vref)
                if "$(line_idx)" in L_gs[k]
                    is_f_var = false
                    if lines[string(line_idx)] == (f_bus, t_bus)
                        is_f_var = true
                    end
                    is_f_var ? expr = plf[string(line_idx)] : expr = plt[string(line_idx)]
                end
            else
                line_idx, f_bus, t_bus = parse_pq_name(vref)
                if "$(line_idx)" in L_gs[k]
                    is_f_var = false
                    if lines[string(line_idx)] == (f_bus, t_bus)
                        is_f_var = true
                    end
                    is_f_var ? expr = qlf[string(line_idx)] : expr = qlt[string(line_idx)]
                end
            end
            if expr != 0
                set = constraint_object(cref).set
                @constraint(nodes[k], expr in set)
            end
        end

        # build objective function
        original_obj_fn = objective_function(pm.model)
        obj_fn = 0
        if original_obj_fn isa GenericAffExpr
            for i in gen_idx_k
                if var(pm, :pg, parse(Int, i)) in keys(original_obj_fn.terms)
                    coeff = original_obj_fn.terms[var(pm, :pg, parse(Int, i))]
                    obj_fn += coeff * pg[i] + data["gen"][i]["cost"][end]
                end
            end
        elseif original_obj_fn isa GenericQuadExpr
            for i in gen_idx_k
                temp_var = var(pm, :pg, parse(Int, i))
                temp_var in keys(original_obj_fn.aff.terms) ? aff_coeff = original_obj_fn.aff.terms[temp_var] : aff_coeff = 0
                temp_pair = UnorderedPair(temp_var, temp_var)
                temp_pair in keys(original_obj_fn.terms) ? quad_coeff = original_obj_fn.terms[temp_pair] : quad_coeff = 0
                obj_fn += aff_coeff * pg[i] + quad_coeff * pg[i]^2
                obj_fn += data["gen"][i]["cost"][end]
            end
        end

        # collect splitted variables of the node
        # shared_vars = Vector{VariableRef}(undef, 8 * length(lines_in_cut))
        shared_vars = Vector{VariableRef}(undef, 4 * length(lines_in_cut))
        for i in eachindex(lines_in_cut)
            line = lines_in_cut[i]
            f_idx = buses_to_idx[lines[line][1]]
            t_idx = buses_to_idx[lines[line][2]]
            shared_vars[8*(i-1)+1] = W[f_idx, t_idx]
            shared_vars[8*(i-1)+2] = W[f_idx, t_idx+length(buses)]
            shared_vars[8*(i-1)+3] = W[f_idx+length(buses), t_idx]
            shared_vars[8*(i-1)+4] = W[f_idx+length(buses), t_idx+length(buses)]
            shared_vars[8*(i-1)+5] = plf[line]
            shared_vars[8*(i-1)+6] = plt[line]
            shared_vars[8*(i-1)+7] = qlf[line]
            shared_vars[8*(i-1)+8] = qlt[line]
            # shared_vars[4*(i-1)+1] = W[f_idx, t_idx]
            # shared_vars[4*(i-1)+2] = W[f_idx, t_idx+length(buses)]
            # shared_vars[4*(i-1)+3] = W[f_idx+length(buses), t_idx]
            # shared_vars[4*(i-1)+4] = W[f_idx+length(buses), t_idx+length(buses)]
        end

        shared_vars_dict[k] = shared_vars

        # objective function
        @objective(nodes[k], Min, obj_fn - sum(λs[k] .* shared_vars))
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
