using PowerModels
using JuMP
using Gurobi
using Ipopt

# Gurobi considers nonconvex quadratic as bilinear constraints
# and use spatial branch-and-bound
function optimize_with_gurobi!(model::JuMP.Model)
    set_optimizer(model, optimizer_with_attributes(Gurobi.Optimizer,
        "NonConvex" => 2,
        "Threads" => 4,
        "TimeLimit" => 60,
        "FeasibilityTol" => 1e-6))
    optimize!(model)
end

function run_acr(data_file::String)
    pm = instantiate_model(data_file, ACRPowerModel, build_opf);
    set_optimizer(pm.model, Ipopt.Optimizer)
    set_optimizer_attribute(pm.model, "constr_viol_tol", 1e-6)
    optimize!(pm.model)
    return pm
end

# Solve ACR with ipopt to set start values
# and use Gurobi
# But, Gurobi cannot translate the start to a feasible solution.
function run_ipopt_and_gurobi(data_file::String)
    pm = run_acr(data_file)
    optimize_with_gurobi!(pm.model)
    return pm
end

function set_pm_start(pm_soc::SOCWRPowerModel, pm_acr::ACRPowerModel)
    for i in ids(pm_soc, :bus)
        vr = JuMP.value(var(pm_acr, :vr)[i])
        vi = JuMP.value(var(pm_acr, :vi)[i])
        JuMP.set_start_value(var(pm_soc, :vr)[i], vr)
        JuMP.set_start_value(var(pm_soc, :vi)[i], vi)
        JuMP.set_start_value(var(pm_soc, :w)[i], vr^2 + vi^2)
    end
    for (i,j) in ids(pm_soc, :buspairs)
        vri = JuMP.value(var(pm_acr, :vr)[i])
        vrj = JuMP.value(var(pm_acr, :vr)[j])
        vii = JuMP.value(var(pm_acr, :vi)[i])
        vij = JuMP.value(var(pm_acr, :vi)[j])
        JuMP.set_start_value(var(pm_soc, :wr)[(i,j)], vri * vrj + vii * vij)
        JuMP.set_start_value(var(pm_soc, :wi)[(i,j)], vii * vrj - vri * vij)
    end
    for (i,j) in ids(pm_soc, :buspairs)
        vri = JuMP.value(var(pm_acr, :vr)[i])
        vrj = JuMP.value(var(pm_acr, :vr)[j])
        vii = JuMP.value(var(pm_acr, :vi)[i])
        vij = JuMP.value(var(pm_acr, :vi)[j])
        JuMP.set_start_value(var(pm_soc, :wr)[(i,j)], vri * vrj + vii * vij)
        JuMP.set_start_value(var(pm_soc, :wi)[(i,j)], vii * vrj - vri * vij)
    end
    for i in ids(pm_soc, :gen)
        JuMP.set_start_value(var(pm_soc, :pg)[i], JuMP.value(var(pm_acr, :pg)[i]))
        JuMP.set_start_value(var(pm_soc, :qg)[i], JuMP.value(var(pm_acr, :qg)[i]))
    end
    for i in ref(pm_soc, :arcs)
        JuMP.set_start_value(var(pm_soc, :p)[i], JuMP.value(var(pm_acr, :p)[i]))
        JuMP.set_start_value(var(pm_soc, :q)[i], JuMP.value(var(pm_acr, :q)[i]))
    end
end

function set_pm_start(pm_soc::SOCWRPowerModel, data_file::String)
    pm_acr = run_acr(data_file)
    set_pm_start(pm_soc, pm_acr)
end

# Build ACR with SOC constraints with start from ipopt
function build_acr_with_socp_cons(data_file::String)
    function build_acopf_with_soc(pm::AbstractPowerModel)
        build_opf(pm)

        variable_bus_voltage_real(pm)
        variable_bus_voltage_imaginary(pm)

        w  = var(pm,  :w)
        wr = var(pm, :wr)
        wi = var(pm, :wi)

        vr = var(pm, :vr)
        vi = var(pm, :vi)

        for (i, bus) in ref(pm, :bus)
            JuMP.@constraint(pm.model, w[i] == vr[i]^2 + vi[i]^2)
        end

        for (i,j) in ids(pm, :buspairs)
            JuMP.@constraint(pm.model, wr[(i,j)] == vr[i] * vr[j] + vi[i] * vi[j])
            JuMP.@constraint(pm.model, wi[(i,j)] == vi[i] * vr[j] - vr[i] * vi[j])
        end
    end

    pm = instantiate_model(data_file, SOCWRPowerModel, build_acopf_with_soc);

    set_pm_start(pm, data_file)

    return pm
end

function add_rlt_lazy_constraints(cb_data, pm::AbstractPowerModel)
    # variable references
    w  = var(pm,  :w)
    wr = var(pm, :wr)
    wi = var(pm, :wi)
    vr = var(pm, :vr)
    vi = var(pm, :vi)

    for (i, bus) in ref(pm, :bus)
        w_val = callback_value(cb_data, w[i])
        vr_val = callback_value(cb_data, vr[i])
        vmax = bus["vmax"]
        vmin = -bus["vmax"]

        if w_val - vmin*vr_val - vmin*vr_val >= -vmin*vmin + 1e-6
            con = @build_constraint(w[i] - vmin*vr[i] - vmin*vr[i] >= -vmin*vmin)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end

        if w_val - vmax*vr_val - vmax*vr_val >= -vmax*vmax + 1e-6
            con = @build_constraint(w[i] - vmax*vr[i] - vmax*vr[i] >= -vmax*vmax)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end

        if w_val - vmin*vr_val - vmax*vr_val <= -vmin*vmax - 1e-6
            con = @build_constraint(w[i] - vmin*vr[i] - vmax*vr[i] <= -vmin*vmax)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end

        if w_val - vmax*vr_val - vmin*vr_val <= -vmax*vmin - 1e-6
            con = @build_constraint(w[i] - vmax*vr[i] - vmin*vr[i] <= -vmax*vmin)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end
    end
    for (i,j) in ids(pm, :buspairs)
        wr_val = callback_value(cb_data, wr[(i,j)])
        wi_val = callback_value(cb_data, wi[(i,j)])
        vri_val = callback_value(cb_data, vr[i])
        vrj_val = callback_value(cb_data, vr[j])
        vii_val = callback_value(cb_data, vi[i])
        vij_val = callback_value(cb_data, vi[j])

        bus = ref(pm, :bus)
        vmaxi = bus[i]["vmax"]
        vmaxj = bus[j]["vmax"]
        vmini = -bus[i]["vmax"]
        vminj = -bus[j]["vmax"]

        if wr_val - vmini*vrj_val - vminj*vri_val >= -vmini*vminj + 1e-6
            con = @build_constraint(wr[(i,j)] - vmini*vr[j] - vminj*vr[i] >= -vmini*vminj)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end

        if wi_val - vmini*vij_val - vminj*vii_val >= -vmini*vminj + 1e-6
            con = @build_constraint(wi[(i,j)] - vmini*vi[j] - vminj*vi[i] >= -vmini*vminj)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end

        if wr_val - vmaxi*vrj_val - vmaxj*vri_val >= -vmaxi*vmaxj + 1e-6
            con = @build_constraint(wr[(i,j)] - vmaxi*vr[j] - vmaxj*vr[i] >= -vmaxi*vmaxj)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end

        if wi_val - vmaxi*vij_val - vmaxj*vii_val >= -vmaxi*vmaxj + 1e-6
            con = @build_constraint(wi[(i,j)] - vmaxi*vi[j] - vmaxj*vi[i] >= -vmaxi*vmaxj)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end

        if wr_val - vmini*vrj_val - vmaxj*vri_val <= -vmini*vmaxj - 1e-6
            con = @build_constraint(wr[(i,j)] - vmini*vr[j] - vmaxj*vr[i] <= -vmini*vmaxj)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end

        if wi_val - vmini*vij_val - vmaxj*vii_val <= -vmini*vmaxj - 1e-6
            con = @build_constraint(wi[(i,j)] - vmini*vi[j] - vmaxj*vi[i] <= -vmini*vmaxj)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end

        if wr_val - vmaxi*vrj_val - vminj*vri_val <= -vmaxi*vminj - 1e-6
            con = @build_constraint(wr[(i,j)] - vmaxi*vr[j] - vminj*vr[i] <= -vmaxi*vminj)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end

        if wi_val - vmaxi*vij_val - vminj*vii_val <= -vmaxi*vminj - 1e-6
            con = @build_constraint(wi[(i,j)] - vmaxi*vi[j] - vminj*vi[i] <= -vmaxi*vminj)
            MOI.submit(pm.model, MOI.LazyConstraint(cb_data), con)
        end
    end
end

# Solve ACR with SOC constraints with start from ipopt
function run_acr_with_socp_cons(data_file::String)
    pm = build_acr_with_socp_cons(data_file);


    function my_lazy_callback_function(cb_data)
        add_rlt_lazy_constraints(cb_data, pm)
    end
    MOI.set(pm.model, MOI.LazyConstraintCallback(), my_lazy_callback_function)

    optimize_with_gurobi!(pm.model)
    return pm
end

# data_file = "../pglib-opf/pglib_opf_case5_pjm.m"
data_file = "../pglib-opf/pglib_opf_case30_ieee.m"

# pm_acr = run_acr(data_file)
# pm = run_ipopt_and_gurobi(data_file);
pm = run_acr_with_socp_cons(data_file);
