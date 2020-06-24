using HDF5, JLD, Plots

case_no = 9
pm_obj = 0
if case_no == 5
    pm_obj = 17551.9
elseif case_no == 9
    pm_obj = 5296.68
elseif case_no == 14
    pm_obj = 8081.52
elseif case_no == 30
    pm_obj = 576.892
end

#file = "./case$(case_no)_ipopt_only/history.jld"
file = "./case$(case_no)_gurobi/history.jld"
#file = "./history.jld"
history = load(file)["history"]
m_kl = history["m_kl"]
D_kl = history["D_kl"]
major_obj_val = history["major_obj_val"]
time = history["time"]
steps = history["step"]
term_statuses = history["termination_status"]

final_itr = findfirst(x->x=="", steps) - 1
avg_time = sum(time[1:final_itr]) / final_itr
plt = plot(m_kl[1:final_itr], label = "TR MP Obj. Val.", legend = (0.4,0.3), xlabel = "No. Iteration", ylabel = "Objective Value")
plot!(D_kl[1:final_itr-1], label = "D_kl")
plot!(major_obj_val[1:final_itr], label = "D_k")
plot!(ones(final_itr) * pm_obj, label = "Global Solution", linestyle = :dash)

serious_steps = findall(x->x=="serious", steps)
if !isempty(serious_steps)
#    vline!(serious_steps, linestyle = :dash, color = :black, label = "serious step")
end
not_solved = findall(x->"Maximum_Iterations_Exceeded" in x, term_statuses)
if !isempty(not_solved)
#    vline!(not_solved, linestyle = :solid, color = :red, label = "subproblem not solved")
end

display(plt)
