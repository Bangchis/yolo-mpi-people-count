// MPI scheduling and communication.
// Static mode uses block-cyclic mapping; dynamic mode uses a master-worker queue.

// Gather variable-length text payloads from all ranks into one string on root.
static std::string gather_string(const std::string& local, int root, MPI_Comm comm) {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int local_size = static_cast<int>(local.size());
    std::vector<int> sizes(size);
    MPI_Gather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, root, comm);
    std::vector<int> displs(size, 0);
    int total = 0;
    if (rank == root) {
        for (int i = 0; i < size; ++i) {
            displs[i] = total;
            total += sizes[i];
        }
    }
    std::string gathered;
    if (rank == root) gathered.resize(total);
    MPI_Gatherv(
        local.data(),
        local_size,
        MPI_CHAR,
        rank == root ? gathered.data() : nullptr,
        sizes.data(),
        displs.data(),
        MPI_CHAR,
        root,
        comm
    );
    return gathered;
}
