using PowerModels, DelimitedFiles

file = "../../pglib-opf/api/pglib_opf_case24_ieee_rts__api.m"
data = parse_file(file)
L = length(data["branch"])
table = zeros(Int, L, 2)
for i in 1:L
    table[i, 1] = data["branch"]["$(i)"]["f_bus"]
    table[i, 2] = data["branch"]["$(i)"]["t_bus"]
end
writedlm("file.csv", table)
