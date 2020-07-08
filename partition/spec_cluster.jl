using LinearAlgebra, PowerModels
using LightGraphs, SimpleWeightedGraphs
using Clustering

function compute_cluster(file, N_partition = -1)
    data = parse_matpower(file)
    N = length(data["bus"])
    buses = collect(keys(data["bus"]))
    buses = sort([parse(Int, i) for i in buses])
    bus_idx_dict = Dict(buses[i] => i for i in eachindex(buses))
    L = length(data["branch"])
    lines = [(data["branch"]["$(i)"]["f_bus"], data["branch"]["$(i)"]["t_bus"]) for i in 1:L]


    # compute weight and degree
    ## based on topology
    # W = zeros(N, N)
    # for (i,j) in lines
    #     W[i,j] = 1
    #     W[j,i] = 1
    # end
    # for i in 1:N
    #     W[i,i] = sum(W[i,:])
    # end
    ## based on admittance
    W = zeros(N, N)
    for k in 1:L
        (i,j) = lines[k]
        r = data["branch"]["$(k)"]["br_r"]
        x = data["branch"]["$(k)"]["br_x"]
        temp = 1 / norm(r + im * x)
        W[bus_idx_dict[i],bus_idx_dict[j]] = temp
        W[bus_idx_dict[j],bus_idx_dict[i]] = temp
    end
    for i in 1:N
        W[i,i] = sum(W[i,:])
    end

    # compute normalized laplacian
    Ln = zeros(N, N)
    for i in 1:N, j in 1:N
        if i == j
            Ln[i,j] = 1
        elseif (buses[i],buses[j]) in lines
            Ln[i,j] = -W[i,j] / (sqrt(W[i,i]) * sqrt(W[j,j]))
            Ln[j,i] = -W[j,i] / (sqrt(W[i,i]) * sqrt(W[j,j]))
        end
    end

    # compute the eigenvectors and eigenvalues of normalized laplacian
    eigen_res = eigen(Ln)
    eigen_vecs = eigen_res.vectors
    eigen_vals = eigen_res.values

    # determine dimension (manually or systematically)
    if N_partition == -1
        gamma = [(eigen_vals[i+1] - eigen_vals[i])/eigen_vals[i] for i in 2:N-1]
        N_partition = findfirst(x->x==maximum(gamma), gamma) + 1
    end

    # normalization
    X = copy(eigen_vecs[:, 1:N_partition])
    for i in 1:N
        X[i,:] = X[i,:] ./ norm(X[i,:])
    end

    # compute distance matrix
    dist = zeros(N, N)
    g = SimpleWeightedGraph(N)
    for (i,j) in lines
        temp_dist = norm(X[bus_idx_dict[i],:] - X[bus_idx_dict[j],:])
        LightGraphs.add_edge!(g, bus_idx_dict[i], bus_idx_dict[j], temp_dist)
    end
    for i in 1:N
        dist[i,:] = dijkstra_shortest_paths(g, i).dists
    end
    dist = (dist + dist') / 2

    # compute hierarchical clustering
    hclu = hclust(dist, linkage=:single)
    pt = cutree(hclu, k = N_partition)
    clusters = [findall(x->x==i, pt) for i in 1:N_partition]

    clusters_orig = [buses[clus] for clus in clusters]
    return clusters_orig
end
