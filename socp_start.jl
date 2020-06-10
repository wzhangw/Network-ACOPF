using PowerModels, JuMP, Ipopt, Gurobi, MathOptInterface, Mosek, MosekTools, COSMO
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

# collect lines
lines = [(data["branch"]["$(i)"]["f_bus"], data["branch"]["$(i)"]["t_bus"]) for i in 1:L]
smax = Dict()
for i in 1:L
    if "rate_a" in keys(data["branch"]["$(i)"])
        smax[(data["branch"]["$(i)"]["f_bus"], data["branch"]["$(i)"]["t_bus"])] = data["branch"]["$(i)"]["rate_a"]
    end
end

# collect shunt data
N_shunt = length(data["shunt"])
shunt_node = [data["shunt"]["$(i)"]["shunt_bus"] for i in 1:N_shunt]
gs = Dict()
bs = Dict()
for i in 1:N_shunt
    gs[shunt_node[i]] = data["shunt"]["$(i)"]["gs"]
    bs[shunt_node[i]] = data["shunt"]["$(i)"]["bs"]
end

# build SOCP relaxation
socp = Model(Gurobi.Optimizer)
@variable(socp, W[1:2*N, 1:2*N], Symmetric)
ref_bus = 4;
@constraint(socp, W[ref_bus+N, ref_bus+N] == 0)
for i in 1:2*N
    if i != ref_bus+N
        @constraint(socp, W[ref_bus+N, i] == 0)
        @constraint(socp, W[i, ref_bus+N] == 0)
    end
end
# SOCP relaxation
for line in lines
    i = line[1]
    j = line[2]
    @constraint(socp, [W[i,i] + W[i+N,i+N] + W[j,j] + W[j+N, j+N],
                     W[i,i] + W[i+N,i+N] - W[j,j] - W[j+N, j+N],
                     2*(W[i,j] + W[i+N,j+N]),
                     2*(W[j,i+N] - W[i,j+N])] in SecondOrderCone())
end

# Other decision variables
@variable(socp, plf[lines])
@variable(socp, plt[lines])
@variable(socp, qlf[lines])
@variable(socp, qlt[lines])
@variable(socp, Pmin[i] <= pg[i=1:N_gen] <= Pmax[i])
@variable(socp, Qmin[i] <= qg[i=1:N_gen] <= Qmax[i])

# constraints 1d, 1e for each node
for i in 1:N
    if i in shunt_node
        @constraint(socp, sum(plf[j] for j in lines if j[1] == i)
                     + sum(plt[j] for j in lines if j[2] == i)
                     - sum(pg[j] for j in 1:N_gen if gen_bus[j] == i)
                     + sum(Pd[j] for j in 1:N_load if load_bus[j] == i)
                     - gs[i] * (W[i,i] + W[i+N,i+N]) == 0)
        @constraint(socp, sum(qlf[j] for j in lines if j[1] == i)
                     + sum(qlt[j] for j in lines if j[2] == i)
                     - sum(qg[j] for j in 1:N_gen if gen_bus[j] == i)
                     + sum(Qd[j] for j in 1:N_load if load_bus[j] == i)
                     - bs[i] * (W[i,i] + W[i+N,i+N]) == 0)
    else
        @constraint(socp, sum(plf[j] for j in lines if j[1] == i)
                     + sum(plt[j] for j in lines if j[2] == i)
                     - sum(pg[j] for j in 1:N_gen if gen_bus[j] == i)
                     + sum(Pd[j] for j in 1:N_load if load_bus[j] == i) == 0)

        @constraint(socp, sum(qlf[j] for j in lines if j[1] == i)
                     + sum(qlt[j] for j in lines if j[2] == i)
                     - sum(qg[j] for j in 1:N_gen if gen_bus[j] == i)
                     + sum(Qd[j] for j in 1:N_load if load_bus[j] == i) == 0)
    end
end

# constraints 1b, 1c
cref_type_list = list_of_constraint_types(pm.model)
crefs = Dict()
for (i,j) in cref_type_list
    crefs[(i,j)] = all_constraints(pm.model, i, j)
end

# extract coefficients for constraints 1b, 1c
for cref in crefs[(GenericQuadExpr{Float64,VariableRef}, MathOptInterface.EqualTo{Float64})]
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
        pairs = [UnorderedPair(variable_by_name(pm.model, "0_vr[$(f_bus)]"), variable_by_name(pm.model, "0_vr[$(f_bus)]")),
                 UnorderedPair(variable_by_name(pm.model, "0_vi[$(f_bus)]"), variable_by_name(pm.model, "0_vi[$(f_bus)]")),
                 UnorderedPair(variable_by_name(pm.model, "0_vr[$(f_bus)]"), variable_by_name(pm.model, "0_vr[$(t_bus)]")),
                 UnorderedPair(variable_by_name(pm.model, "0_vr[$(f_bus)]"), variable_by_name(pm.model, "0_vi[$(t_bus)]")),
                 UnorderedPair(variable_by_name(pm.model, "0_vi[$(f_bus)]"), variable_by_name(pm.model, "0_vr[$(t_bus)]")),
                 UnorderedPair(variable_by_name(pm.model, "0_vi[$(f_bus)]"), variable_by_name(pm.model, "0_vi[$(t_bus)]"))]
        if p_or_q == 'p'
            if is_f
                new_expr = zero(QuadExpr) + plf[line]
            else
                new_expr = zero(QuadExpr) + plt[line]
            end
            vrefs = [W[f_bus, f_bus], W[f_bus+N, f_bus+N], W[f_bus, t_bus],
                     W[f_bus, t_bus+N], W[f_bus+N, t_bus], W[f_bus+N, t_bus+N]]
            @constraint(socp, new_expr + sum(quad_terms[pairs[i]] * vrefs[i] for i in eachindex(pairs) if pairs[i] in keys(quad_terms)) == 0)
        else # p_or_q == 'q'
            if is_f
                new_expr = zero(QuadExpr) + qlf[line]
            else
            new_expr = zero(QuadExpr) + qlt[line]
            end
            vrefs = [W[f_bus, f_bus], W[f_bus+N, f_bus+N], W[f_bus, t_bus],
                     W[f_bus, t_bus+N], W[f_bus+N, t_bus], W[f_bus+N, t_bus+N]]
            @constraint(socp, new_expr + sum(quad_terms[pairs[i]] * vrefs[i] for i in eachindex(pairs) if pairs[i] in keys(quad_terms)) == 0)
        end
    end
end

# constraints 1g
for i in 1:N
    @constraint(socp, Vmin[i]^2 <= W[i,i] + W[i+N, i+N] <= Vmax[i]^2)
end

# constraints 1h
for i in lines
    if i in keys(smax)
        @constraint(socp, plf[i]^2 + qlf[i]^2 <= smax[i]^2)
        @constraint(socp, plt[i]^2 + qlt[i]^2 <= smax[i]^2)
    end
end

# objective
if gen_cost_type == 2
    @objective(socp, Min, sum( sum(costs[i][j] * pg[i]^(length(costs[i])-j) for j in 1:length(costs[i])) for i in 1:N_gen))
end

optimize!(socp)
W_start = value.(W)
plf_start = value.(plf)
plt_start = value.(plt)
qlf_start = value.(qlf)
qlt_start = value.(qlt)
pg_start = value.(pg)
qg_start = value.(qg)

# build model
dm = Model()
# W variables
@variable(dm, W[1:2*N, 1:2*N])

# Original: Exact rank-1 feasible set for W
@variable(dm, v[1:2*N])
for i in 1:2*N, j in 1:2*N
    @constraint(dm, W[i,j] == v[i] * v[j])
end

# reference bus (from PowerModels.jl model)
ref_bus = 4;
@constraint(dm, v[ref_bus+N] == 0)

# Other decision variables
@variable(dm, plf[lines])
@variable(dm, plt[lines])
@variable(dm, qlf[lines])
@variable(dm, qlt[lines])
@variable(dm, Pmin[i] <= pg[i=1:N_gen] <= Pmax[i])
@variable(dm, Qmin[i] <= qg[i=1:N_gen] <= Qmax[i])

# constraints 1d, 1e for each node
for i in 1:N
    if i in shunt_node
        @constraint(dm, sum(plf[j] for j in lines if j[1] == i)
                     + sum(plt[j] for j in lines if j[2] == i)
                     - sum(pg[j] for j in 1:N_gen if gen_bus[j] == i)
                     + sum(Pd[j] for j in 1:N_load if load_bus[j] == i)
                     - gs[i] * (W[i,i] + W[i+N,i+N]) == 0)
        @constraint(dm, sum(qlf[j] for j in lines if j[1] == i)
                     + sum(qlt[j] for j in lines if j[2] == i)
                     - sum(qg[j] for j in 1:N_gen if gen_bus[j] == i)
                     + sum(Qd[j] for j in 1:N_load if load_bus[j] == i)
                     - bs[i] * (W[i,i] + W[i+N,i+N]) == 0)
    else
        @constraint(dm, sum(plf[j] for j in lines if j[1] == i)
                     + sum(plt[j] for j in lines if j[2] == i)
                     - sum(pg[j] for j in 1:N_gen if gen_bus[j] == i)
                     + sum(Pd[j] for j in 1:N_load if load_bus[j] == i) == 0)

        @constraint(dm, sum(qlf[j] for j in lines if j[1] == i)
                     + sum(qlt[j] for j in lines if j[2] == i)
                     - sum(qg[j] for j in 1:N_gen if gen_bus[j] == i)
                     + sum(Qd[j] for j in 1:N_load if load_bus[j] == i) == 0)
    end
end

# constraints 1b, 1c
cref_type_list = list_of_constraint_types(pm.model)
crefs = Dict()
for (i,j) in cref_type_list
    crefs[(i,j)] = all_constraints(pm.model, i, j)
end

# extract coefficients for constraints 1b, 1c
for cref in crefs[(GenericQuadExpr{Float64,VariableRef}, MathOptInterface.EqualTo{Float64})]
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
        pairs = [UnorderedPair(variable_by_name(pm.model, "0_vr[$(f_bus)]"), variable_by_name(pm.model, "0_vr[$(f_bus)]")),
                 UnorderedPair(variable_by_name(pm.model, "0_vi[$(f_bus)]"), variable_by_name(pm.model, "0_vi[$(f_bus)]")),
                 UnorderedPair(variable_by_name(pm.model, "0_vr[$(f_bus)]"), variable_by_name(pm.model, "0_vr[$(t_bus)]")),
                 UnorderedPair(variable_by_name(pm.model, "0_vr[$(f_bus)]"), variable_by_name(pm.model, "0_vi[$(t_bus)]")),
                 UnorderedPair(variable_by_name(pm.model, "0_vi[$(f_bus)]"), variable_by_name(pm.model, "0_vr[$(t_bus)]")),
                 UnorderedPair(variable_by_name(pm.model, "0_vi[$(f_bus)]"), variable_by_name(pm.model, "0_vi[$(t_bus)]"))]
        if p_or_q == 'p'
            if is_f
                new_expr = zero(QuadExpr) + plf[line]
            else
                new_expr = zero(QuadExpr) + plt[line]
            end
            vrefs = [W[f_bus, f_bus], W[f_bus+N, f_bus+N], W[f_bus, t_bus],
                     W[f_bus, t_bus+N], W[f_bus+N, t_bus], W[f_bus+N, t_bus+N]]
            @constraint(dm, new_expr + sum(quad_terms[pairs[i]] * vrefs[i] for i in eachindex(pairs) if pairs[i] in keys(quad_terms)) == 0)
        else # p_or_q == 'q'
            if is_f
                new_expr = zero(QuadExpr) + qlf[line]
            else
            new_expr = zero(QuadExpr) + qlt[line]
            end
            vrefs = [W[f_bus, f_bus], W[f_bus+N, f_bus+N], W[f_bus, t_bus],
                     W[f_bus, t_bus+N], W[f_bus+N, t_bus], W[f_bus+N, t_bus+N]]
            @constraint(dm, new_expr + sum(quad_terms[pairs[i]] * vrefs[i] for i in eachindex(pairs) if pairs[i] in keys(quad_terms)) == 0)
        end
    end
end

# constraints 1g
for i in 1:N
    @constraint(dm, Vmin[i]^2 <= W[i,i] + W[i+N, i+N] <= Vmax[i]^2)
end

# constraints 1h
for i in lines
    if i in keys(smax)
        @constraint(dm, plf[i]^2 + qlf[i]^2 <= smax[i]^2)
        @constraint(dm, plt[i]^2 + qlt[i]^2 <= smax[i]^2)
    end
end

# objective
if gen_cost_type == 2
    @objective(dm, Min, sum( sum(costs[i][j] * pg[i]^(length(costs[i])-j) for j in 1:length(costs[i])) for i in 1:N_gen))
end

set_start_value.(W, W_start)
for i in lines
    set_start_value(plf[i], plf_start[i])
    set_start_value(plt[i], plt_start[i])
    set_start_value(qlf[i], qlf_start[i])
    set_start_value(qlt[i], qlt_start[i])
end
set_start_value.(pg, pg_start)
set_start_value.(qg, qg_start)
set_optimizer(dm, Gurobi.Optimizer)
set_optimizer_attribute(dm, "NonConvex", 2)
optimize!(dm)
t = solve_time(dm)

println("Solution time: ", t)
println("Objective value: ", objective_value(dm))
