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

// One asynchronous send keeps its text buffer alive until MPI confirms delivery.
struct StreamSend {
    std::string payload;
    int payload_size = 0;
    MPI_Request size_request = MPI_REQUEST_NULL;
    MPI_Request data_request = MPI_REQUEST_NULL;
};

// Start sending one batch result. The caller may immediately continue computing.
static double stream_start_send(
    const std::string& payload,
    int root,
    MPI_Comm comm,
    int tag_base,
    std::deque<StreamSend>& pending
) {
    auto t0 = std::chrono::steady_clock::now();

    pending.emplace_back();
    auto& send = pending.back();
    send.payload = payload;
    send.payload_size = static_cast<int>(send.payload.size());

    MPI_Isend(&send.payload_size, 1, MPI_INT, root, tag_base, comm, &send.size_request);

    if (send.payload_size > 0) {
        MPI_Isend(
            send.payload.data(),
            send.payload_size,
            MPI_CHAR,
            root,
            tag_base + 1,
            comm,
            &send.data_request
        );
    }

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Wait for one pending batch send. This is where back-pressure becomes visible.
static double stream_wait_one_send(std::deque<StreamSend>& pending) {
    if (pending.empty()) {
        return 0.0;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto& send = pending.front();

    MPI_Wait(&send.size_request, MPI_STATUS_IGNORE);

    if (send.data_request != MPI_REQUEST_NULL) {
        MPI_Wait(&send.data_request, MPI_STATUS_IGNORE);
    }

    pending.pop_front();

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Flush all sends before the worker announces that it is done.
static double stream_wait_all_sends(std::deque<StreamSend>& pending) {
    double wait_ms = 0.0;

    while (!pending.empty()) {
        wait_ms += stream_wait_one_send(pending);
    }

    return wait_ms;
}

// A negative size is the end-of-stream marker for one rank.
static double stream_send_done(int root, MPI_Comm comm, int tag_base) {
    int done = -1;
    auto t0 = std::chrono::steady_clock::now();
    MPI_Send(&done, 1, MPI_INT, root, tag_base, comm);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Root receives batch payloads into stable buffers until each MPI_Irecv completes.
struct StreamReceive {
    int source = 0;
    std::string payload;
    MPI_Request request = MPI_REQUEST_NULL;
};

// Move finished receive buffers into the per-rank payload buckets.
static bool stream_collect_finished_receives(
    std::deque<StreamReceive>& pending,
    std::vector<std::string>& payload_by_rank
) {
    bool progress = false;

    for (auto it = pending.begin(); it != pending.end();) {
        int done = 0;
        MPI_Test(&it->request, &done, MPI_STATUS_IGNORE);

        if (done) {
            payload_by_rank[it->source] += it->payload;
            it = pending.erase(it);
            progress = true;
        } else {
            ++it;
        }
    }

    return progress;
}

// Poll for available size messages. Data receives are posted without blocking.
static bool stream_probe_size_messages(
    MPI_Comm comm,
    int tag_base,
    int& active_senders,
    std::deque<StreamReceive>& pending
) {
    bool progress = false;

    while (true) {
        int available = 0;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, tag_base, comm, &available, &status);

        if (!available) {
            break;
        }

        int payload_size = 0;
        MPI_Recv(&payload_size, 1, MPI_INT, status.MPI_SOURCE, tag_base, comm, MPI_STATUS_IGNORE);
        progress = true;

        if (payload_size < 0) {
            // This worker has sent all batches and its metrics.
            active_senders -= 1;
            continue;
        }

        pending.emplace_back();
        auto& recv = pending.back();
        recv.source = status.MPI_SOURCE;
        recv.payload.resize(payload_size);

        if (payload_size > 0) {
            MPI_Irecv(
                recv.payload.data(),
                payload_size,
                MPI_CHAR,
                status.MPI_SOURCE,
                tag_base + 1,
                comm,
                &recv.request
            );
        }
    }

    return progress;
}

// Lightweight receive progress used by root between its own compute batches.
static bool stream_poll_root(
    MPI_Comm comm,
    int tag_base,
    int& active_senders,
    std::deque<StreamReceive>& pending,
    std::vector<std::string>& payload_by_rank
) {
    bool progress = false;
    progress = stream_collect_finished_receives(pending, payload_by_rank) || progress;
    progress = stream_probe_size_messages(comm, tag_base, active_senders, pending) || progress;
    progress = stream_collect_finished_receives(pending, payload_by_rank) || progress;
    return progress;
}

// After root finishes its own work, drain the remaining worker messages.
static double stream_drain_root(
    MPI_Comm comm,
    int tag_base,
    int& active_senders,
    std::deque<StreamReceive>& pending,
    std::vector<std::string>& payload_by_rank
) {
    auto t0 = std::chrono::steady_clock::now();

    while (active_senders > 0 || !pending.empty()) {
        bool progress = stream_poll_root(comm, tag_base, active_senders, pending, payload_by_rank);

        if (progress) {
            continue;
        }

        if (!pending.empty()) {
            // There is already a posted receive, so wait for the oldest payload.
            MPI_Wait(&pending.front().request, MPI_STATUS_IGNORE);
            payload_by_rank[pending.front().source] += pending.front().payload;
            pending.pop_front();
            continue;
        }

        // No posted receive exists yet, so wait for the next size or done marker.
        MPI_Status status;
        int payload_size = 0;
        MPI_Recv(&payload_size, 1, MPI_INT, MPI_ANY_SOURCE, tag_base, comm, &status);

        if (payload_size < 0) {
            active_senders -= 1;
            continue;
        }

        std::string payload(payload_size, '\0');

        if (payload_size > 0) {
            MPI_Recv(
                payload.data(),
                payload_size,
                MPI_CHAR,
                status.MPI_SOURCE,
                tag_base + 1,
                comm,
                MPI_STATUS_IGNORE
            );
        }

        payload_by_rank[status.MPI_SOURCE] += payload;
    }

    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
