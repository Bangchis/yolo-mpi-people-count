// MPI scheduling and communication.
// Static mode uses block-cyclic mapping and then gathers text payloads at rank 0.

// Gather variable-length text payloads from all ranks into one string on root.
static std::string gather_string(const std::string& local, int root, MPI_Comm comm) {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int local_size = static_cast<int>(local.size());
    std::vector<int> sizes(size);

    // First gather message sizes because each rank has a different payload length.
    MPI_Gather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, root, comm);

    std::vector<int> displs(size, 0);
    int total = 0;

    if (rank == root) {
        for (int i = 0; i < size; ++i) {
            // displs[i] is the byte offset where rank i's payload begins.
            displs[i] = total;
            total += sizes[i];
        }
    }

    std::string gathered;

    if (rank == root) {
        // Root allocates exactly enough space for all serialized payloads.
        gathered.resize(total);
    }

    // Gatherv is needed because payload sizes are not equal across ranks.
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

// Non-blocking variable-length gather for static mode.
//
// The message size is not known by the root in advance, so the gather still has
// two logical phases:
//   1. every non-root rank sends its payload length with MPI_Isend;
//   2. root posts MPI_Irecv for all payload buffers and waits for them together.
//
// This does not overlap YOLO computation yet because all payloads are produced
// after local inference. It does avoid receiving rank results one by one and is
// the non-blocking communication variant used in the report experiment.
static std::string gather_string_nonblocking(
    const std::string& local,
    int root,
    MPI_Comm comm,
    int tag_base
) {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int local_size = static_cast<int>(local.size());

    if (rank != root) {
        std::vector<MPI_Request> send_requests;
        send_requests.reserve(local_size > 0 ? 2 : 1);

        MPI_Request size_request = MPI_REQUEST_NULL;
        MPI_Isend(&local_size, 1, MPI_INT, root, tag_base, comm, &size_request);
        send_requests.push_back(size_request);

        if (local_size > 0) {
            MPI_Request data_request = MPI_REQUEST_NULL;
            MPI_Isend(local.data(), local_size, MPI_CHAR, root, tag_base + 1, comm, &data_request);
            send_requests.push_back(data_request);
        }

        MPI_Waitall(static_cast<int>(send_requests.size()), send_requests.data(), MPI_STATUSES_IGNORE);
        return "";
    }

    std::vector<int> sizes(size, 0);
    sizes[root] = local_size;

    std::vector<MPI_Request> size_requests;
    size_requests.reserve(std::max(0, size - 1));

    for (int src = 0; src < size; ++src) {
        if (src == root) {
            continue;
        }

        MPI_Request request = MPI_REQUEST_NULL;
        MPI_Irecv(&sizes[src], 1, MPI_INT, src, tag_base, comm, &request);
        size_requests.push_back(request);
    }

    if (!size_requests.empty()) {
        MPI_Waitall(static_cast<int>(size_requests.size()), size_requests.data(), MPI_STATUSES_IGNORE);
    }

    std::vector<int> displs(size, 0);
    int total = 0;

    for (int i = 0; i < size; ++i) {
        displs[i] = total;
        total += sizes[i];
    }

    std::string gathered;
    gathered.resize(total);

    if (local_size > 0) {
        std::copy(local.begin(), local.end(), gathered.begin() + displs[root]);
    }

    std::vector<MPI_Request> data_requests;
    data_requests.reserve(std::max(0, size - 1));

    for (int src = 0; src < size; ++src) {
        if (src == root || sizes[src] == 0) {
            continue;
        }

        MPI_Request request = MPI_REQUEST_NULL;
        MPI_Irecv(gathered.data() + displs[src], sizes[src], MPI_CHAR, src, tag_base + 1, comm, &request);
        data_requests.push_back(request);
    }

    if (!data_requests.empty()) {
        MPI_Waitall(static_cast<int>(data_requests.size()), data_requests.data(), MPI_STATUSES_IGNORE);
    }

    return gathered;
}
