// Construction and configuration only; never called by a decode/policy step.
#pragma once

#include "historical_kv_policy.hpp"

#include <map>
#include <memory>
#include <string>
#include <variant>

namespace vllm_kvmem {

using HistoricalPolicyValue = std::variant<bool, int64_t, double, std::string>;
using HistoricalPolicyParameters = std::map<std::string, HistoricalPolicyValue>;

namespace historical_policy_detail {
inline bool lru_keep_mandatory(const HistoricalPolicyParameters &params, bool fallback) {
    for (const auto &[key, value] : params) {
        if (key != "keep_mandatory")
            throw std::invalid_argument("Unknown historical_gc.policy_params.lru parameter: " + key);
        const auto *keep = std::get_if<bool>(&value);
        if (!keep)
            throw std::invalid_argument("historical_gc.policy_params.keep_mandatory must be boolean");
        fallback = *keep;
    }
    return fallback;
}
[[noreturn]] inline void unknown(const std::string &name) {
    throw std::invalid_argument("Unknown historical_gc.policy: " + name + " (supported: lru)");
}
} // namespace historical_policy_detail

inline void validate_historical_kv_policy(const std::string &name,
                                          const HistoricalPolicyParameters &params) {
    if (name == "lru") {
        historical_policy_detail::lru_keep_mandatory(params, true);
        return;
    }
    historical_policy_detail::unknown(name);
}

// Add each built-in policy's validation/construction here. Policies only emit
// intents; all implementations share the manager's existing release protocol.
// The fallback preserves the earlier native Config.keep_mandatory API; an
// explicit policy parameter takes precedence.
inline std::unique_ptr<HistoricalKvPolicy> make_historical_kv_policy(
        const std::string &name, const HistoricalPolicyParameters &params,
        size_t capacity, bool keep_mandatory = true) {
    if (name == "lru")
        return std::make_unique<HistoricalLruPolicy>(capacity,
            historical_policy_detail::lru_keep_mandatory(params, keep_mandatory));
    historical_policy_detail::unknown(name);
}

} // namespace vllm_kvmem
