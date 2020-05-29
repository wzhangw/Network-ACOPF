using PowerModels, JuMP, Ipopt, MathOptInterface
#using Gurobi, COSMO, Mosek, MosekTools
const MOI = MathOptInterface

file = "case5.m"
data = parse_matpower(file)
pm = instantiate_model(file, ACRPowerModel, PowerModels.build_opf)

# print all variables
vars = all_variables(pm.model)

# collect variables by name
N = length(data["bus"])
L = length(data["branch"])
vr = [variable_by_name(pm.model, "0_vr[$(i)]") for i in 1:N]
vi = [variable_by_name(pm.model, "0_vi[$(i)]") for i in 1:N]
pg = [variable_by_name(pm.model, "0_pg[$(i)]") for i in 1:N]
qg = [variable_by_name(pm.model, "0_qg[$(i)]") for i in 1:N]
plf = [variable_by_name(pm.model, "0_p[($(i), $(data["branch"]["$(i)"]["f_bus"]), $(data["branch"]["$(i)"]["t_bus"]))]") for i in 1:L]
qlf = [variable_by_name(pm.model, "0_q[($(i), $(data["branch"]["$(i)"]["f_bus"]), $(data["branch"]["$(i)"]["t_bus"]))]") for i in 1:L]
plt = [variable_by_name(pm.model, "0_p[($(i), $(data["branch"]["$(i)"]["t_bus"]), $(data["branch"]["$(i)"]["f_bus"]))]") for i in 1:L]
qlt = [variable_by_name(pm.model, "0_q[($(i), $(data["branch"]["$(i)"]["t_bus"]), $(data["branch"]["$(i)"]["f_bus"]))]") for i in 1:L]

cref_type_list = list_of_constraint_types(pm.model)
crefs = Dict()
for (i,j) in cref_type_list
    crefs[(i,j)] = all_constraints(pm.model, i, j)
end

# collect nodal balance constraints
# TODO: figure out a more systematic way of identifying balanc constraints...
nb_crefs_active = Dict(1 => crefs[cref_type_list[1]][10],
                         2 => crefs[cref_type_list[1]][4],
                         3 => crefs[cref_type_list[1]][6],
                         4 => crefs[cref_type_list[1]][2],
                         5 => crefs[cref_type_list[1]][8])
nb_crefs_reactive = Dict(1 => crefs[cref_type_list[1]][11],
                           2 => crefs[cref_type_list[1]][5],
                           3 => crefs[cref_type_list[1]][7],
                           4 => crefs[cref_type_list[1]][3],
                           5 => crefs[cref_type_list[1]][9])

# collect constraints defining p,q in terms of v (a.k.a W)
# TODO: improve this for large-scale problem (will be very slow)
plf_equality = Dict{Int64, ConstraintRef}()
plt_equality = Dict{Int64, ConstraintRef}()
qlf_equality = Dict{Int64, ConstraintRef}()
qlt_equality = Dict{Int64, ConstraintRef}()

for cref in crefs[GenericQuadExpr{Float64, VariableRef}, MOI.EqualTo{Float64}]
    for i in 1:L
        if normalized_coefficient(cref, plf[i]) != 0
            plf_equality[i] = cref
            break
        elseif normalized_coefficient(cref, qlf[i]) != 0
            qlf_equality[i] = cref
            break
        elseif normalized_coefficient(cref, plt[i]) != 0
            plt_equality[i] = cref
            break
        elseif normalized_coefficient(cref, qlt[i]) != 0
            qlt_equality[i] = cref
            break
        end
    end
end

# nb_crefs_active = Dict()
# nb_crefs_reactive = Dict()
# for constr in crefs[GenericAffExpr{Float64, VariableRef}, MOI.EqualTo{Float64}] # loop through nodal balance constraints
#     for i in 1:N
#         if normalized_coefficient(constr, pg[i]) != 0
#             nb_crefs_active[i] = constr
#         end
#         if normalized_coefficient(constr, qg[i]) != 0
#             nb_crefs_reactive[i] = constr
#         end
#     end
# end


# Define partitioned decision variables
# This part can be replaced with generalized code later...
partitions = [Set([1,2,3]), Set([4,5])]
K = length(partitions)
cut_lines_idx = [2, 3, 5]
cut_lines = [(1,4), (1,5), (3,4)]
@variable(pm.model, w[l in cut_lines])
plfk = Dict{NTuple{4, Int64}, VariableRef}() # plfk[l,k] == pf[l] for every partitio k, for every l in the cut
qlfk = Dict{NTuple{4, Int64}, VariableRef}()
pltk = Dict{NTuple{4, Int64}, VariableRef}()
qltk = Dict{NTuple{4, Int64}, VariableRef}()
Wrr = Dict{NTuple{3, Int64}, VariableRef}() # Wrr[i,j,k] == vr[i] * vr[j] for every partition k, for every l = (i,j) in the cut
Wri = Dict{NTuple{3, Int64}, VariableRef}() # Wri[i,j,k] == vr[i] * vi[j] for every partition k, for every l = (i,j) in the cut
Wir = Dict{NTuple{3, Int64}, VariableRef}() # Wir[i,j,k] == vi[i] * vr[j] for every partition k, for every l = (i,j) in the cut
Wii = Dict{NTuple{3, Int64}, VariableRef}() # Wii[i,j,k] == vi[i] * vi[j] for every partition k, for every l = (i,j) in the cut

for i in eachindex(cut_lines_idx), j in 1:K
    idx = (cut_lines[i][1], cut_lines[i][2], j)
    Wrr[idx] = @variable(pm.model, base_name = "Wrr[$(idx)]")
    Wri[idx] = @variable(pm.model, base_name = "Wri[$(idx)]")
    Wir[idx] = @variable(pm.model, base_name = "Wir[$(idx)]")
    Wii[idx] = @variable(pm.model, base_name = "Wii[$(idx)]")

    idx = (cut_lines_idx[i], cut_lines[i][1], cut_lines[i][2], j)
    plfk[idx] = @variable(pm.model, base_name = "plfk[$(idx)]")
    qlfk[idx] = @variable(pm.model, base_name = "qlfk[$(idx)]")

    idx = (cut_lines_idx[i], cut_lines[i][2], cut_lines[i][1], j)
    pltk[idx] = @variable(pm.model, base_name = "pltk[$(idx)]")
    qltk[idx] = @variable(pm.model, base_name = "qltk[$(idx)]")
end

for i in 1:N, j in 1:K
    idx = (i, i, j)
    Wrr[idx] = @variable(pm.model, base_name = "Wrr[$(idx)]")
    Wri[idx] = @variable(pm.model, base_name = "Wri[$(idx)]")
    Wir[idx] = @variable(pm.model, base_name = "Wir[$(idx)]")
    Wii[idx] = @variable(pm.model, base_name = "Wii[$(idx)]")
end

# Define partitioned variables
for i in eachindex(cut_lines_idx), j in 1:K
    idx = (cut_lines[i][1], cut_lines[i][2], j)
    @constraint(pm.model, Wrr[idx] == vr[cut_lines[i][1]] * vr[cut_lines[i][2]])
    @constraint(pm.model, Wri[idx] == vr[cut_lines[i][1]] * vi[cut_lines[i][2]])
    @constraint(pm.model, Wir[idx] == vi[cut_lines[i][1]] * vr[cut_lines[i][2]])
    @constraint(pm.model, Wii[idx] == vi[cut_lines[i][1]] * vi[cut_lines[i][2]])

    idx = (cut_lines_idx[i], cut_lines[i][1], cut_lines[i][2], j)
    @constraint(pm.model, plf[cut_lines_idx[i]] == plfk[idx])
    @constraint(pm.model, qlf[cut_lines_idx[i]] == qlfk[idx])
    idx = (cut_lines_idx[i], cut_lines[i][2], cut_lines[i][1], j)
    @constraint(pm.model, plt[cut_lines_idx[i]] == pltk[idx])
    @constraint(pm.model, qlt[cut_lines_idx[i]] == qltk[idx])
end

for i in 1:N, j in 1:K
    idx = (i, i, j)
    @constraint(pm.model, Wrr[idx] == vr[i] * vr[i])
    @constraint(pm.model, Wri[idx] == vr[i] * vi[i])
    @constraint(pm.model, Wir[idx] == vi[i] * vr[i])
    @constraint(pm.model, Wii[idx] == vi[i] * vi[i])
end

# Replace v * v pairs in constraints (1b), (1c) which relate p, q with v (a.k.a W)
# TODO: this part seems to relax the original problem...
arr = []
for i in eachindex(cut_lines_idx)
    line_idx = cut_lines_idx[i]
    f_bus, t_bus = cut_lines[i]
    for (cref, pqref) in [(plf_equality[line_idx], plfk),
                          (qlf_equality[line_idx], qlfk)]
        constr = constraint_object(cref)
        func = constr.func
        set = constr.set
        terms = func.terms
        # add constraints for each partition
        for j in 1:K
            idx = (line_idx, f_bus, t_bus, j)
            new_expr = zero(QuadExpr) + pqref[idx]
            add_to_expression!(new_expr, Wrr[(f_bus, f_bus, j)] * terms[UnorderedPair(vr[f_bus], vr[f_bus])])
            add_to_expression!(new_expr, Wii[(f_bus, f_bus, j)] * terms[UnorderedPair(vi[f_bus], vi[f_bus])])
            add_to_expression!(new_expr, Wri[(f_bus, t_bus, j)] * terms[UnorderedPair(vr[f_bus], vi[t_bus])])
            add_to_expression!(new_expr, Wir[(f_bus, t_bus, j)] * terms[UnorderedPair(vi[f_bus], vr[t_bus])])
            add_to_expression!(new_expr, Wrr[(f_bus, t_bus, j)] * terms[UnorderedPair(vr[f_bus], vr[t_bus])])
            add_to_expression!(new_expr, Wii[(f_bus, t_bus, j)] * terms[UnorderedPair(vi[f_bus], vi[t_bus])])
            new_con = ScalarConstraint(new_expr, set)
            push!(arr, add_constraint(pm.model, new_con))
        end
        delete(pm.model, cref)
    end
    for (cref, pqref) in [(plt_equality[line_idx], pltk),
                          (qlt_equality[line_idx], qltk)]
        constr = constraint_object(cref)
        func = constr.func
        set = constr.set
        terms = func.terms
        # add constraints for each partition
        for j in 1:K
            idx = (line_idx, t_bus, f_bus, j)
            new_expr = zero(QuadExpr) + pqref[idx]
            add_to_expression!(new_expr, Wrr[(t_bus, t_bus, j)] * terms[UnorderedPair(vr[t_bus], vr[t_bus])])
            add_to_expression!(new_expr, Wii[(t_bus, t_bus, j)] * terms[UnorderedPair(vi[t_bus], vi[t_bus])])
            add_to_expression!(new_expr, Wri[(f_bus, t_bus, j)] * terms[UnorderedPair(vr[t_bus], vi[f_bus])])
            add_to_expression!(new_expr, Wir[(f_bus, t_bus, j)] * terms[UnorderedPair(vi[t_bus], vr[f_bus])])
            add_to_expression!(new_expr, Wrr[(f_bus, t_bus, j)] * terms[UnorderedPair(vr[t_bus], vr[f_bus])])
            add_to_expression!(new_expr, Wii[(f_bus, t_bus, j)] * terms[UnorderedPair(vi[t_bus], vi[f_bus])])
            new_con = ScalarConstraint(new_expr, set)
            push!(arr, add_constraint(pm.model, new_con))
        end
        delete(pm.model, cref)
    end
end

# Rewrite nodal balance in terms of partitioned decision variables
for i in 1:N
    cref_active = nb_crefs_active[i]
    cref_reactive = nb_crefs_reactive[i]
    partition = Int(sum([ j * (i in partitions[j]) for j in eachindex(partitions) ]))
    for j in eachindex(cut_lines_idx)

        line_idx = cut_lines_idx[j]
        idx = (line_idx, cut_lines[j][1], cut_lines[j][2], partition)
        plf_idx = normalized_coefficient(cref_active, plf[line_idx])
        if plf_idx != 0
            set_normalized_coefficient(cref_active, plfk[idx], plf_idx)
            set_normalized_coefficient(cref_active, plf[line_idx], 0)
        end
        qlf_idx = normalized_coefficient(cref_reactive, qlf[line_idx])
        if normalized_coefficient(cref_reactive, qlf[line_idx]) != 0
            set_normalized_coefficient(cref_reactive, qlfk[idx], qlf_idx)
            set_normalized_coefficient(cref_reactive, qlf[line_idx], 0)
        end

        idx = (line_idx, cut_lines[j][2], cut_lines[j][1], partition)
        plt_idx = normalized_coefficient(cref_active, plt[line_idx])
        if plt_idx != 0
            set_normalized_coefficient(cref_active, pltk[idx], plt_idx)
            set_normalized_coefficient(cref_active, plt[line_idx], 0)
        end
        qlt_idx = normalized_coefficient(cref_reactive, qlt[line_idx])
        if normalized_coefficient(cref_reactive, qlt[line_idx]) != 0
            set_normalized_coefficient(cref_reactive, qltk[idx], qlt_idx)
            set_normalized_coefficient(cref_reactive, qlt[line_idx], 0)
        end
    end
end

set_optimizer(pm.model, Ipopt.Optimizer)
optimize!(pm.model)
