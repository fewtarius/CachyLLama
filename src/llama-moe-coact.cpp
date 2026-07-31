// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 fewtarius

#ifdef _WIN32
#    include <direct.h>
#endif

#include "llama-moe-coact.h"
#include "llama.h"
#include "llama-model.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <sys/stat.h>

namespace llama_moe_coact {

void init(matrix & m, const struct llama_model & model) {
    const int nl = model.hparams.n_layer();
    const int ne = model.hparams.n_expert;
    if (nl <= 0 || ne <= 0) {
        m.num_layers = 0;
        m.num_experts = 0;
        return;
    }
    m.num_layers = nl;
    m.num_experts = ne;
    m.layer_pair_counts.assign(nl, std::vector<uint32_t>((size_t) ne * (size_t) ne, 0));
    m.cross_counts.assign(nl, std::vector<std::vector<uint32_t>>(ne, std::vector<uint32_t>(ne, 0)));
    m.observation_counts.assign(nl, 0);
}

void record(matrix & m,
            int layer,
            const int32_t * selected,
            int n_selected) {
    if (layer < 0 || layer >= m.num_layers) return;
    if (!selected || n_selected <= 0) return;
    const int ne = m.num_experts;

    auto & pc = m.layer_pair_counts[layer];
    auto & cc = m.cross_counts[layer];
    auto & oc = m.observation_counts[layer];
    oc++;

    // Within-layer pair counts (symmetric, including diagonal).
    for (int i = 0; i < n_selected; ++i) {
        const int a = selected[i];
        if (a < 0 || a >= ne) continue;
        // Self-count.
        pc[(size_t) a * ne + a]++;
        for (int j = i + 1; j < n_selected; ++j) {
            const int b = selected[j];
            if (b < 0 || b >= ne) continue;
            pc[(size_t) a * ne + b]++;
            pc[(size_t) b * ne + a]++;
        }
    }

    // Cross-layer is recorded via record_cross_layer() so we can pair the
    // current selection with the previous one explicitly. This keeps the
    // public API simple.
    (void) cc;
}

void record_cross_layer(matrix & m,
                        int layer,
                        const int32_t * selected_n,
                        int n_n,
                        const int32_t * selected_n1,
                        int n_n1) {
    if (layer < 0 || layer >= m.num_layers) return;
    if (!selected_n || !selected_n1) return;
    const int ne = m.num_experts;
    auto & cc = m.cross_counts[layer];

    for (int i = 0; i < n_n; ++i) {
        const int a = selected_n[i];
        if (a < 0 || a >= ne) continue;
        for (int j = 0; j < n_n1; ++j) {
            const int b = selected_n1[j];
            if (b < 0 || b >= ne) continue;
            cc[(size_t) a][(size_t) b]++;
        }
    }
}

std::vector<int32_t> predict_same_layer(
        const matrix & m,
        int layer,
        const int32_t * observed,
        int n_observed,
        int top_k) {
    std::vector<int32_t> out;
    if (layer < 0 || layer >= m.num_layers) return out;
    if (!observed || n_observed <= 0 || top_k <= 0) return out;
    const int ne = m.num_experts;
    if (m.observation_counts[layer] == 0) return out;

    std::vector<uint32_t> scores(ne, 0);
    const auto & pc = m.layer_pair_counts[layer];
    for (int i = 0; i < n_observed; ++i) {
        const int e = observed[i];
        if (e < 0 || e >= ne) continue;
        for (int j = 0; j < ne; ++j) {
            scores[j] += pc[(size_t) e * ne + j];
        }
    }
    // Don't predict the already-observed experts.
    for (int i = 0; i < n_observed; ++i) {
        const int e = observed[i];
        if (e >= 0 && e < ne) scores[e] = 0;
    }

    // Sort by score descending, return top-k.
    std::vector<std::pair<int, uint32_t>> ranked;
    ranked.reserve(ne);
    for (int j = 0; j < ne; ++j) {
        if (scores[j] > 0) ranked.emplace_back(j, scores[j]);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });
    out.reserve(std::min(top_k, (int) ranked.size()));
    for (int i = 0; i < top_k && i < (int) ranked.size(); ++i) {
        out.push_back(ranked[i].first);
    }
    return out;
}

std::vector<int32_t> predict_next_layer(
        const matrix & m,
        int layer,
        const int32_t * observed,
        int n_observed,
        int top_k) {
    std::vector<int32_t> out;
    if (layer < 0 || layer >= m.num_layers) return out;
    if (!observed || n_observed <= 0 || top_k <= 0) return out;
    const int ne = m.num_experts;
    if (m.observation_counts[layer] == 0) return out;

    std::vector<uint32_t> scores(ne, 0);
    const auto & cc = m.cross_counts[layer];
    for (int i = 0; i < n_observed; ++i) {
        const int e = observed[i];
        if (e < 0 || e >= ne) continue;
        for (int j = 0; j < ne; ++j) {
            scores[j] += cc[(size_t) e][(size_t) j];
        }
    }

    std::vector<std::pair<int, uint32_t>> ranked;
    ranked.reserve(ne);
    for (int j = 0; j < ne; ++j) {
        if (scores[j] > 0) ranked.emplace_back(j, scores[j]);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });
    out.reserve(std::min(top_k, (int) ranked.size()));
    for (int i = 0; i < top_k && i < (int) ranked.size(); ++i) {
        out.push_back(ranked[i].first);
    }
    return out;
}

std::string persistence_path(const std::string & model_path) {
    // Extract base name without path or extension.
    std::string base  = model_path;
    auto        slash = base.find_last_of("/\\");
    if (slash != std::string::npos) {
        base = base.substr(slash + 1);
    }
    auto dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        base = base.substr(0, dot);
    }

    const char * home = std::getenv("HOME");
    if (!home || !*home) {
#ifdef _WIN32
        home = std::getenv("TEMP");
        if (!home || !*home) {
            home = "C:\\Temp";
        }
#else
        home = "/tmp";
#endif
    }

#ifdef _WIN32
    std::string dir = std::string(home) + "\\.cachylla\\coactivation";
    _mkdir(dir.c_str());
    return dir + "\\" + base + ".json";
#else
    std::string dir = std::string(home) + "/.cachylla/coactivation";
    mkdir(dir.c_str(), 0755);
    return dir + "/" + base + ".json";
#endif
}

bool save(const matrix & m, const std::string & path) {
    std::ofstream out(path);
    if (!out) return false;

    // Custom JSON serialization (compact, avoid pulling in a heavy dep).
    out << "{\"v\":1,\"nl\":" << m.num_layers << ",\"ne\":" << m.num_experts << ",\n";
    out << "\"oc\":[";
    for (int i = 0; i < m.num_layers; ++i) {
        if (i) out << ",";
        out << m.observation_counts[i];
    }
    out << "],\n";
    out << "\"lp\":[";
    for (int i = 0; i < m.num_layers; ++i) {
        if (i) out << ",";
        out << "[";
        const auto & pc = m.layer_pair_counts[i];
        for (int j = 0; j < (int) pc.size(); ++j) {
            if (j) out << ",";
            out << pc[j];
        }
        out << "]";
    }
    out << "],\n";
    out << "\"cc\":[";
    for (int i = 0; i < m.num_layers; ++i) {
        if (i) out << ",";
        out << "[";
        const auto & row = m.cross_counts[i];
        for (int j = 0; j < (int) row.size(); ++j) {
            if (j) out << ",";
            out << "[";
            for (int k = 0; k < (int) row[j].size(); ++k) {
                if (k) out << ",";
                out << row[j][k];
            }
            out << "]";
        }
        out << "]";
    }
    out << "]\n";
    out << "}\n";
    return (bool) out;
}

// Minimal JSON parser sufficient for our compact format.
// Reads from the start of `in`, returns true on success.
static bool parse_int_array(const std::string & in, size_t & pos, std::vector<uint32_t> & out) {
    if (in[pos] != '[') return false;
    pos++;
    while (pos < in.size() && in[pos] != ']') {
        if (in[pos] == ',') pos++;
        // Skip whitespace.
        while (pos < in.size() && std::isspace((unsigned char) in[pos])) pos++;
        // Read integer.
        int sign = 1;
        if (pos < in.size() && in[pos] == '-') { sign = -1; pos++; }
        uint64_t v = 0;
        bool any = false;
        while (pos < in.size() && std::isdigit((unsigned char) in[pos])) {
            v = v * 10 + (in[pos] - '0');
            pos++;
            any = true;
        }
        if (!any) return false;
        out.push_back((uint32_t)(sign * (int64_t) v));
    }
    if (pos < in.size() && in[pos] == ']') pos++;
    return true;
}

static bool skip_value(const std::string & in, size_t & pos) {
    // Skip a JSON value: number, string, array, object, bool, null.
    if (pos >= in.size()) return false;
    char c = in[pos];
    if (c == '{' || c == '[') {
        char open = c;
        char close = (c == '{') ? '}' : ']';
        int depth = 1;
        pos++;
        while (pos < in.size() && depth > 0) {
            if (in[pos] == open) depth++;
            else if (in[pos] == close) depth--;
            pos++;
        }
        return depth == 0;
    }
    if (c == '"') {
        pos++;
        while (pos < in.size() && in[pos] != '"') {
            if (in[pos] == '\\' && pos + 1 < in.size()) pos++;
            pos++;
        }
        if (pos < in.size()) pos++;
        return pos <= in.size();
    }
    while (pos < in.size() && in[pos] != ',' && in[pos] != '}' && in[pos] != ']' && !std::isspace((unsigned char) in[pos])) pos++;
    return true;
}

static bool find_key(const std::string & in, size_t & pos, const std::string & key) {
    // Skip whitespace and commas.
    while (pos < in.size() && (std::isspace((unsigned char) in[pos]) || in[pos] == ',')) pos++;
    if (pos >= in.size() || in[pos] != '"') return false;
    pos++;
    size_t start = pos;
    while (pos < in.size() && in[pos] != '"') pos++;
    if (pos >= in.size()) return false;
    std::string found(in.data() + start, pos - start);
    pos++;
    while (pos < in.size() && std::isspace((unsigned char) in[pos])) pos++;
    if (pos >= in.size() || in[pos] != ':') return false;
    pos++;
    while (pos < in.size() && std::isspace((unsigned char) in[pos])) pos++;
    return found == key;
}

bool load(matrix & m, const std::string & path) {
    std::ifstream in(path);
    if (!in) return false;
    std::stringstream ss; ss << in.rdbuf();
    std::string s = ss.str();
    size_t pos = 0;

    if (!find_key(s, pos, "nl")) return false;
    std::vector<uint32_t> tmp;
    if (!parse_int_array(s, pos, tmp)) return false;
    if (tmp.size() != 1) return false;
    int nl = (int) tmp[0];

    if (!find_key(s, pos, "ne")) return false;
    tmp.clear();
    if (!parse_int_array(s, pos, tmp)) return false;
    if (tmp.size() != 1) return false;
    int ne = (int) tmp[0];

    if (nl <= 0 || ne <= 0) return false;

    m.num_layers = nl;
    m.num_experts = ne;
    m.observation_counts.assign(nl, 0);
    m.layer_pair_counts.assign(nl, std::vector<uint32_t>((size_t) ne * (size_t) ne, 0));
    m.cross_counts.assign(nl, std::vector<std::vector<uint32_t>>(ne, std::vector<uint32_t>(ne, 0)));

    // Skip "oc" (recomputed from current session; we keep the loaded value
    // for reference).
    if (!find_key(s, pos, "oc")) return false;
    tmp.clear();
    if (!parse_int_array(s, pos, tmp)) return false;
    if ((int) tmp.size() == nl) {
        for (int i = 0; i < nl; ++i) m.observation_counts[i] = tmp[i];
    }

    if (!find_key(s, pos, "lp")) return false;
    // Expect outer array of nl inner arrays, each of size ne*ne.
    if (pos >= s.size() || s[pos] != '[') return false;
    pos++;
    for (int i = 0; i < nl; ++i) {
        if (!parse_int_array(s, pos, tmp)) return false;
        if ((int) tmp.size() != ne * ne) return false;
        m.layer_pair_counts[i] = std::move(tmp);
        tmp.clear();
    }
    if (pos < s.size() && s[pos] == ']') pos++;

    if (!find_key(s, pos, "cc")) return false;
    if (pos >= s.size() || s[pos] != '[') return false;
    pos++;
    for (int i = 0; i < nl; ++i) {
        if (pos >= s.size() || s[pos] != '[') return false;
        pos++;
        for (int j = 0; j < ne; ++j) {
            if (!parse_int_array(s, pos, tmp)) return false;
            if ((int) tmp.size() != ne) return false;
            m.cross_counts[i][j] = std::move(tmp);
            tmp.clear();
        }
        if (pos < s.size() && s[pos] == ']') pos++;
    }
    if (pos < s.size() && s[pos] == ']') pos++;

    return true;
}

} // namespace llama_moe_coact
