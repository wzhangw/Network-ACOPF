# second version of model code: aims to incorporate multiple lines between nodes,
# and arbitrary indexing of buses

using PowerModels, JuMP, Ipopt, Gurobi, Plasmo, LinearAlgebra, MathOptInterface, DataStructures
using HDF5, JLD
const MOI = MathOptInterface

using Random
Random.seed!(0)

include("../partition/spec_cluster.jl")
include("../util_fns.jl")


# Main code
# file = "../../matpower/data/case30.m"
file = "../../pglib-opf/api/pglib_opf_case14_ieee__api.m"
# file = "../../pglib-opf/api/pglib_opf_case30_as__api.m"
# file = "case14.m"
# file = "../../pglib-opf/api/pglib_opf_case5_pjm__api.m"
data = parse_file(file)
pm = instantiate_model(file, ACRPowerModel, PowerModels.build_opf)

# Partition data
# ieee case 9
# N_gs = [[1, 2, 4, 8, 9], [3, 5, 6, 7]]
# N_gs = [[2, 3, 4], [1, 5]]
# N_gs = [[2, 3], [1, 4, 5]] # bad iteration
# N_gs = [[1,4,9],[3,5,6],[2,7,8]]
# N_gs = [[1,4,100],[3,5,6],[2,7,8]]
# N_gs = [[1,4,9],[3,5,6,7],[2,8]]
N_gs = [[1,2,3,4,5],[7,8,9,10,14],[6,11,12,13]]
# N_gs = [[1,2,3,4,5,6,7],[9,10,11,21,22],[12,13,16,17],[14,15,18,19,20],[23,24,25,26],[27,29,30,8,28]]
# N_gs = compute_cluster(file, 5)

# run the bundle-trust-region algorithm
λ_dims = [length(find_cut_lines(data, i))*8 for i in N_gs]
# λ_dims = [length(find_cut_lines(data, i))*4 for i in N_gs]
cut_lines = unique(vcat([find_cut_lines(data, i) for i in N_gs]...))
lambdas = [zeros(Float64, i) for i in λ_dims]
tr_center = [zeros(Float64, i) for i in λ_dims]
itr_count = 1
max_itr = 1000
mg_and_dict = build_subgraph_model_from_file(data, pm, N_gs, lambdas)
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

optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi_optimizer = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0, "NonConvex" => 2)

curr_obj_vals = []
mg, shared_vars_dict = mg_and_dict
for node in mg.modelnodes
    set_start_value.(all_variables(node.model), 1)
    JuMP.set_optimizer(node.model, optimizer)
    optimize!(node.model)

    # start_vals = value.(all_variables(node.model))
    # JuMP.set_optimizer(node.model, gurobi_optimizer)
    # set_start_value.(all_variables(node.model), start_vals)
    # optimize!(node.model)

    append!(curr_obj_vals, objective_value(node.model))
end
major_obj_val = sum(curr_obj_vals)

for line in cut_lines
    f_bus = data["branch"][line]["f_bus"]
    t_bus = data["branch"][line]["t_bus"]
    k_f = 0
    k_t = 0
    f_idx_in_shared_var_list = 0
    t_idx_in_shared_var_list = 0
    for i in eachindex(N_gs)
        if f_bus in N_gs[i]
            k_f = i
            lines_in_cut = find_cut_lines(data, N_gs[i])
            f_idx_in_shared_var_list = first(findall(x->x==line, lines_in_cut))
        end
        if t_bus in N_gs[i]
            k_t = i
            lines_in_cut = find_cut_lines(data, N_gs[i])
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
    global mg, shared_vars_dict = build_subgraph_model_from_file(data, pm, N_gs, lambdas)
    curr_obj_vals = []
    for node in mg.modelnodes
        set_start_value.(all_variables(node.model), 1)
        JuMP.set_optimizer(node.model, optimizer)
        optimize!(node.model)
        while termination_status(node.model) != LOCALLY_SOLVED
            # set_start_value.(all_variables(node.model), rand(Int) % 100)
            println("Subgraph not solved to local optimality. Restart with different initial values.")
            set_start_value.(all_variables(node.model), rand())
            JuMP.set_optimizer(node.model, optimizer)
            optimize!(node.model)
        end
        itr_time += solve_time(node.model)
        push!(term_statuses, raw_status(node.model))

        # start_vals = value.(all_variables(node.model))
        # JuMP.set_optimizer(node.model, gurobi_optimizer)
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
