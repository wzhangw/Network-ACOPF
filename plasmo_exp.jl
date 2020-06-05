using JuMP, Plasmo, Gurobi, Ipopt

graph = ModelGraph()

@node(graph, n1)
@node(graph, n2)

@variable(n1, x)
@variable(n1, -0.5 <= y <= 0.5)
@constraint(n1, x + y >= 1)
@objective(n1, Min, y)

@variable(n2, x)
@variable(n2, z)
@constraint(n2, x + z <= 1)
@objective(n2, Max, z)

@linkconstraint(graph, n1[:x] == n2[:x])

JuMP.set_optimizer(n2, Ipopt.Optimizer)
optimize!(n2)
