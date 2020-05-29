using PowerModels, JuMP, Ipopt, Gurobi, MathOptInterface
const MOI = MathOptInterface

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
c = [data["gen"]["$(i)"]["cost"][1] for i in 1:N_gen] # assume linear cost

# Partition data
N_g1 = [1, 2, 3]
N_g2 = [4, 5]
L_g1 = [(1,2), (2,3), (1,4), (1,5), (3,4)]
L_g2 = [(4,5), (1,4), (1,5), (3,4)]
cut_lines = [(1,4), (1,5), (3,4)]
smax = Dict((1, 2) => 4.0, (4, 5) => 2.4)

# build model
dm = Model()
# Wk (and underlying vk) variables
@variable(dm, W1[1:2*N, 1:2*N])
@variable(dm, v1[1:2*N])
@variable(dm, W2[1:2*N, 1:2*N])
@variable(dm, v2[1:2*N])

for i in 1:2*N, j in 1:2*N
    @constraint(dm, W1[i,j] == v1[i] * v1[j])
    @constraint(dm, W2[i,j] == v2[i] * v2[j])
end

# xk, yk variables
@variable(dm, plf1[L_g1])
@variable(dm, plt1[L_g1])
@variable(dm, qlf1[L_g1])
@variable(dm, qlt1[L_g1])
@variable(dm, plf2[L_g2])
@variable(dm, plt2[L_g2])
@variable(dm, qlf2[L_g2])
@variable(dm, qlt2[L_g2])
@variable(dm, Pmin[i] <= pg[i=1:N_gen] <= Pmax[i])
@variable(dm, Qmin[i] <= qg[i=1:N_gen] <= Qmax[i])
# y variables
@variable(dm, wrr[cut_lines])
@variable(dm, wri[cut_lines])
@variable(dm, wir[cut_lines])
@variable(dm, wii[cut_lines])
@variable(dm, plf[cut_lines])
@variable(dm, qlf[cut_lines])
@variable(dm, plt[cut_lines])
@variable(dm, qlt[cut_lines])

# constraints 2d, 2e, 2f
for i in cut_lines
    @constraint(dm, wrr[i] == W1[i[1], i[2]])
    @constraint(dm, wrr[i] == W2[i[1], i[2]])
    @constraint(dm, wri[i] == W1[i[1], i[2]+N])
    @constraint(dm, wri[i] == W2[i[1], i[2]+N])
    @constraint(dm, wir[i] == W1[i[1]+N, i[2]])
    @constraint(dm, wir[i] == W2[i[1]+N, i[2]])
    @constraint(dm, wii[i] == W1[i[1]+N, i[2]+N])
    @constraint(dm, wii[i] == W2[i[1]+N, i[2]+N])

    @constraint(dm, plf[i] == plf1[i])
    @constraint(dm, plf[i] == plf2[i])
    @constraint(dm, qlf[i] == qlf1[i])
    @constraint(dm, qlf[i] == qlf2[i])
    @constraint(dm, plt[i] == plt1[i])
    @constraint(dm, plt[i] == plt2[i])
    @constraint(dm, qlt[i] == qlt1[i])
    @constraint(dm, qlt[i] == qlt2[i])
end

# constraints 1d, 1e for each node
for i in N_g1
    @constraint(dm, sum(plf1[j] for j in L_g1 if j[1] == i)
                 + sum(plt1[j] for j in L_g1 if j[2] == i)
                 - sum(pg[j] for j in 1:N_gen if gen_bus[j] == i)
                 + sum(Pd[j] for j in 1:N_load if load_bus[j] == i) == 0)

    @constraint(dm, sum(qlf1[j] for j in L_g1 if j[1] == i)
                 + sum(qlt1[j] for j in L_g1 if j[2] == i)
                 - sum(qg[j] for j in 1:N_gen if gen_bus[j] == i)
                 + sum(Qd[j] for j in 1:N_load if load_bus[j] == i) == 0)
end

for i in N_g2
    @constraint(dm, sum(plf2[j] for j in L_g2 if j[1] == i)
                 + sum(plt2[j] for j in L_g2 if j[2] == i)
                 - sum(pg[j] for j in 1:N_gen if gen_bus[j] == i)
                 + sum(Pd[j] for j in 1:N_load if load_bus[j] == i) == 0)

    @constraint(dm, sum(qlf2[j] for j in L_g2 if j[1] == i)
                 + sum(qlt2[j] for j in L_g2 if j[2] == i)
                 - sum(qg[j] for j in 1:N_gen if gen_bus[j] == i)
                 + sum(Qd[j] for j in 1:N_load if load_bus[j] == i) == 0)
end

# constraints 1b, 1c
cref_type_list = list_of_constraint_types(pm.model)
crefs = Dict()
for (i,j) in cref_type_list
    crefs[(i,j)] = all_constraints(pm.model, i, j)
end

# extract coefficients for constraints 1b, 1c
for cref in crefs[cref_type_list[2]]
    aff_expr = constraint_object(cref).func.aff
    var = first(keys(aff_expr.terms))
    quad_terms = constraint_object(cref).func.terms
    p_or_q = name(var)[3]
    f_bus = name(var)[9] - '0'
    t_bus = name(var)[12] - '0'
    if f_bus < t_bus
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
        if line in L_g1
            if is_f
                new_expr = zero(QuadExpr) + plf1[line]
            else
                new_expr = zero(QuadExpr) + plt1[line]
            end
            vrefs = [W1[f_bus, f_bus], W1[f_bus+N, f_bus+N], W1[f_bus, t_bus],
                     W1[f_bus, t_bus+N], W1[f_bus+N, t_bus], W1[f_bus+N, t_bus+N]]
            @constraint(dm, new_expr + sum(quad_terms[pairs[i]] * vrefs[i] for i in eachindex(pairs)) == 0)
        end
        if line in L_g2
            if is_f
                new_expr = zero(QuadExpr) + plf2[line]
            else
                new_expr = zero(QuadExpr) + plt2[line]
            end
            vrefs = [W2[f_bus, f_bus], W2[f_bus+N, f_bus+N], W2[f_bus, t_bus],
                     W2[f_bus, t_bus+N], W2[f_bus+N, t_bus], W2[f_bus+N, t_bus+N]]
            @constraint(dm, new_expr + sum(quad_terms[pairs[i]] * vrefs[i] for i in eachindex(pairs)) == 0)
        end
    else # p_or_q == 'q'
        if line in L_g1
            if is_f
                new_expr = zero(QuadExpr) + qlf1[line]
            else
                new_expr = zero(QuadExpr) + qlt1[line]
            end
            vrefs = [W1[f_bus, f_bus], W1[f_bus+N, f_bus+N], W1[f_bus, t_bus],
                     W1[f_bus, t_bus+N], W1[f_bus+N, t_bus], W1[f_bus+N, t_bus+N]]
            @constraint(dm, new_expr + sum(quad_terms[pairs[i]] * vrefs[i] for i in eachindex(pairs)) == 0)
        end
        if line in L_g2
            if is_f
                new_expr = zero(QuadExpr) + qlf2[line]
            else
                new_expr = zero(QuadExpr) + qlt2[line]
            end
            vrefs = [W2[f_bus, f_bus], W2[f_bus+N, f_bus+N], W2[f_bus, t_bus],
                     W2[f_bus, t_bus+N], W2[f_bus+N, t_bus], W2[f_bus+N, t_bus+N]]
            @constraint(dm, new_expr + sum(quad_terms[pairs[i]] * vrefs[i] for i in eachindex(pairs)) == 0)
        end
    end
end

# constraints 1g
for i in 1:N
    @constraint(dm, Vmin[i]^2 <= W1[i,i] + W1[i+N, i+N] <= Vmax[i]^2)
    @constraint(dm, Vmin[i]^2 <= W2[i,i] + W2[i+N, i+N] <= Vmax[i]^2)
end

# constraints 1h
for i in L_g1
    if i in keys(smax)
        @constraint(dm, plf1[i]^2 + qlf1[i]^2 <= smax[i])
        @constraint(dm, plt1[i]^2 + qlt1[i]^2 <= smax[i])
    end
end
for i in L_g2
    if i in keys(smax)
        @constraint(dm, plf2[i]^2 + qlf2[i]^2 <= smax[i])
        @constraint(dm, plt2[i]^2 + qlt2[i]^2 <= smax[i])
    end
end

# objective
@objective(dm, Min, sum(c .* pg))

# optimize
set_optimizer(dm, Gurobi.Optimizer)
set_optimizer_attribute(dm, "NonConvex", 2)


#set_optimizer(dm, Ipopt.Optimizer)
optimize!(dm)
println("Objective value: ", objective_value(dm))
