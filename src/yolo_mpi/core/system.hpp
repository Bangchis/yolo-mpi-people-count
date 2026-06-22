// Config parsing and task generation.
// This file answers: "what input did the user request, and what are the tasks?"

static std::string hostname() {
    std::array<char, 256> buf{};
    if (gethostname(buf.data(), buf.size() - 1) == 0) {
        return std::string(buf.data());
    }
    return "unknown";
}

static bool read_bool_arg(const std::string& value) {
    return value == "1" || value == "true" || value == "TRUE" || value == "yes";
}
