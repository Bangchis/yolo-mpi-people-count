// Small MPI protocol helpers used by live mode.
// Result and camera-tile messages are length-prefixed text.

// Send arbitrary text payload through MPI using a length prefix.
static void send_string(const std::string& payload, int dest, int tag) {
    int n = static_cast<int>(payload.size());

    // Send size first so the receiver can allocate the string buffer.
    MPI_Send(&n, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);

    if (n > 0) {
        MPI_Send(payload.data(), n, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
    }
}

// Receive a length-prefixed text payload and optionally expose MPI status.
static std::string recv_string(int source, int tag, MPI_Status* status_out = nullptr) {
    MPI_Status status;
    int n = 0;

    MPI_Recv(&n, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);

    std::string payload;
    payload.resize(n);

    if (n > 0) {
        // Receive from status.MPI_SOURCE because source can be MPI_ANY_SOURCE.
        MPI_Recv(payload.data(), n, MPI_CHAR, status.MPI_SOURCE, tag, MPI_COMM_WORLD, &status);
    }

    if (status_out) {
        *status_out = status;
    }

    return payload;
}

// Read one line from a child process pipe and trim newline characters.
static bool read_pipe_line(FILE* stream, std::string& line) {
    char* buffer = nullptr;
    size_t cap = 0;
    ssize_t n = getline(&buffer, &cap, stream);

    if (n < 0) {
        std::free(buffer);
        return false;
    }

    line.assign(buffer, static_cast<size_t>(n));
    std::free(buffer);

    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }

    return true;
}

// Tiny process wrapper for helpers that write lines to stdout, e.g. camera source.
class OutputPipeProcess {
public:
    // Start a helper process and capture its stdout as a readable line stream.
    explicit OutputPipeProcess(const std::vector<std::string>& args) {
        int from_child[2] = {-1, -1};
        if (pipe(from_child) != 0) {
            throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));
        }
        pid_ = fork();
        if (pid_ < 0) {
            throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
        }

        if (pid_ == 0) {
            // Child writes to stdout; parent reads the other end of this pipe.
            dup2(from_child[1], STDOUT_FILENO);
            close(from_child[0]);
            close(from_child[1]);

            std::vector<std::string> mutable_args = args;
            std::vector<char*> argv;
            argv.reserve(mutable_args.size() + 1);

            for (auto& arg : mutable_args) {
                argv.push_back(arg.data());
            }

            argv.push_back(nullptr);
            execvp(argv[0], argv.data());
            std::cerr << "failed to exec " << argv[0] << ": " << std::strerror(errno) << "\n";
            _exit(127);
        }

        close(from_child[1]);
        output_ = fdopen(from_child[0], "r");

        if (!output_) {
            throw std::runtime_error("fdopen failed for output pipe");
        }
    }

    OutputPipeProcess(const OutputPipeProcess&) = delete;
    OutputPipeProcess& operator=(const OutputPipeProcess&) = delete;

    // Close the pipe and wait for the helper process to exit.
    ~OutputPipeProcess() {
        if (output_) {
            std::fclose(output_);
            output_ = nullptr;
        }
        if (pid_ > 0) {
            int status = 0;
            waitpid(pid_, &status, 0);
            pid_ = -1;
        }
    }

    // Read one protocol line from the helper process.
    bool read_line(std::string& line) {
        if (!output_) {
            return false;
        }

        return read_pipe_line(output_, line);
    }

private:
    pid_t pid_ = -1;
    FILE* output_ = nullptr;
};

// Tiny process wrapper for helpers that read lines from stdin, e.g. live viewer.
class InputPipeProcess {
public:
    // Start a helper process and keep its stdin writable from C++.
    explicit InputPipeProcess(const std::vector<std::string>& args) {
        int to_child[2] = {-1, -1};
        if (pipe(to_child) != 0) {
            throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));
        }
        pid_ = fork();
        if (pid_ < 0) {
            throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
        }
        if (pid_ == 0) {
            // Child reads from stdin; parent writes the other end of this pipe.
            dup2(to_child[0], STDIN_FILENO);
            close(to_child[0]);
            close(to_child[1]);

            std::vector<std::string> mutable_args = args;
            std::vector<char*> argv;
            argv.reserve(mutable_args.size() + 1);

            for (auto& arg : mutable_args) {
                argv.push_back(arg.data());
            }

            argv.push_back(nullptr);
            execvp(argv[0], argv.data());
            std::cerr << "failed to exec " << argv[0] << ": " << std::strerror(errno) << "\n";
            _exit(127);
        }

        close(to_child[0]);
        input_ = fdopen(to_child[1], "w");

        if (!input_) {
            throw std::runtime_error("fdopen failed for input pipe");
        }

        setvbuf(input_, nullptr, _IOLBF, 0);
    }

    InputPipeProcess(const InputPipeProcess&) = delete;
    InputPipeProcess& operator=(const InputPipeProcess&) = delete;

    // Tell the helper to quit, then close and reap it.
    ~InputPipeProcess() {
        if (input_) {
            write_line("QUIT");
            std::fclose(input_);
            input_ = nullptr;
        }
        if (pid_ > 0) {
            int status = 0;
            waitpid(pid_, &status, 0);
            pid_ = -1;
        }
    }

    // Send one line to the helper process over stdin.
    void write_line(const std::string& line) {
        if (!input_) {
            return;
        }

        std::fwrite(line.data(), 1, line.size(), input_);
        std::fwrite("\n", 1, 1, input_);
        std::fflush(input_);
    }

private:
    pid_t pid_ = -1;
    FILE* input_ = nullptr;
};
