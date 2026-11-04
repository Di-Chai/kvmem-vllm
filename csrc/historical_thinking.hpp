// Token-coordinate classification for accepted historical KV rows.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vllm_kvmem {

// This parser never decides acceptance. Its caller feeds only the input-token
// prefix whose forward KV rows were actually retained, not sampled output IDs.
// Header encodings come from the model's real tokenizer. Unknown templates and
// incomplete/unknown headers cannot classify body tokens as thinking.
class HistoricalThinkingParser {
public:
    enum Label : uint8_t { Unknown = 0, NonThinking = 1, Thinking = 2 };
    // Header order: system, developer, user, assistant, tool. Each includes the
    // terminating newline and excludes the preceding message-start token.
    HistoricalThinkingParser(std::vector<uint32_t> markers,
                             std::vector<std::vector<uint32_t>> headers)
        : headers_(std::move(headers)) {
        if (markers.size() != 4 || headers_.size() != 5)
            throw std::invalid_argument("Thinking parser requires four markers and five role headers");
        std::copy(markers.begin(), markers.end(), markers_.begin());
        auto distinct = markers;
        std::sort(distinct.begin(), distinct.end());
        if (std::adjacent_find(distinct.begin(), distinct.end()) != distinct.end())
            throw std::invalid_argument("Thinking parser markers must be distinct");
        for (size_t i = 0; i < headers_.size(); ++i) {
            const auto &header = headers_[i];
            if (header.empty() || header.size() > 32)
                throw std::invalid_argument("Thinking role header must contain 1..32 tokens");
            for (auto token : header)
                if (std::find(markers_.begin(), markers_.end(), token) != markers_.end())
                    throw std::invalid_argument("Thinking role header contains a structural marker");
            for (size_t j = 0; j < i; ++j) {
                const auto length = std::min(header.size(), headers_[j].size());
                if (std::equal(header.begin(), header.begin() + length, headers_[j].begin()))
                    throw std::invalid_argument("Thinking role headers must not share a complete prefix");
            }
        }
    }

    void reset() noexcept { role_ = Role::Unknown; thinking_ = false; header_at_ = candidates_ = 0; }

    Label consume(uint32_t token) noexcept {
        if (token == markers_[0]) {
            role_ = Role::Header; thinking_ = false; header_at_ = 0; candidates_ = 31;
            return NonThinking;
        }
        if (token == markers_[1]) {
            reset();
            return NonThinking;
        }
        if (role_ == Role::Header) {
            uint32_t remaining = 0;
            for (size_t i = 0; i < headers_.size(); ++i) {
                if (!(candidates_ & (1u << i)) || header_at_ >= headers_[i].size() ||
                        headers_[i][header_at_] != token) continue;
                if (header_at_ + 1 == headers_[i].size()) {
                    role_ = i == 3 ? Role::Assistant : Role::Other;
                    header_at_ = candidates_ = 0;
                    return NonThinking;
                }
                remaining |= 1u << i;
            }
            candidates_ = remaining;
            ++header_at_;
            if (!remaining) { reset(); return Unknown; }
            return NonThinking;
        }
        if (role_ == Role::Unknown) return Unknown;
        if (role_ == Role::Other) return NonThinking;
        if (token == markers_[2]) { thinking_ = true; return NonThinking; }
        if (token == markers_[3]) { thinking_ = false; return NonThinking; }
        return thinking_ ? Thinking : NonThinking;
    }

    std::vector<uint8_t> feed(const std::vector<uint32_t> &tokens) {
        std::vector<uint8_t> labels;
        labels.reserve(tokens.size());
        for (auto token : tokens) labels.push_back(consume(token));
        return labels;
    }
    void seed(const std::vector<uint32_t> &tokens) noexcept {
        for (auto token : tokens) consume(token);
    }
    bool thinking() const noexcept { return role_ == Role::Assistant && thinking_; }

private:
    enum class Role : uint8_t { Unknown, Header, Other, Assistant };
    std::array<uint32_t, 4> markers_{};
    std::vector<std::vector<uint32_t>> headers_;
    Role role_ = Role::Unknown;
    bool thinking_ = false;
    uint32_t header_at_ = 0, candidates_ = 0;
};

} // namespace vllm_kvmem
