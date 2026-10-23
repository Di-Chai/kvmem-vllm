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
inline HistoricalStatisticalPolicy::Config statistical_config(const HistoricalPolicyParameters &params) {
    HistoricalStatisticalPolicy::Config config;
    for (const auto &[key, value] : params) {
        if (key == "min_observations" || key == "min_cold_observations") {
            const auto *count = std::get_if<int64_t>(&value);
            if (!count || *count < 0 || (key == "min_observations" && *count == 0))
                throw std::invalid_argument("historical_gc.policy_params." + key + " must be an integer (observations > 0, cold observations >= 0)");
            if (key == "min_observations") config.min_observations = static_cast<uint64_t>(*count);
            else config.min_cold_observations = static_cast<uint64_t>(*count);
        } else if (key == "proactive") {
            const auto *enabled = std::get_if<bool>(&value);
            if (!enabled) throw std::invalid_argument("historical_gc.policy_params.proactive must be boolean");
            config.proactive = *enabled;
        } else if (key == "max_importance" || key == "min_model_reclaim_score") {
            double number;
            if (const auto *fraction = std::get_if<double>(&value)) number = *fraction;
            else if (const auto *integer = std::get_if<int64_t>(&value)) number = static_cast<double>(*integer);
            else throw std::invalid_argument("historical_gc.policy_params." + key + " must be a finite number");
            const bool valid = std::isfinite(number) &&
                ((number >= 0 && number <= 1) || (key == "min_model_reclaim_score" && number == -1));
            if (!valid) throw std::invalid_argument("historical_gc.policy_params." + key + " is out of range");
            if (key == "max_importance") config.max_importance = number;
            else config.min_model_reclaim_score = number;
        } else throw std::invalid_argument("Unknown historical_gc.policy_params.statistical parameter: " + key);
    }
    return config;
}
[[noreturn]] inline void unknown(const std::string &name) {
    throw std::invalid_argument("Unknown historical_gc.policy: " + name + " (supported: lru, statistical)");
}
} // namespace historical_policy_detail

inline void validate_historical_kv_policy(const std::string &name,
                                          const HistoricalPolicyParameters &params) {
    if (name == "lru") {
        historical_policy_detail::lru_keep_mandatory(params, true);
        return;
    }
    if (name == "statistical") {
        historical_policy_detail::statistical_config(params);
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
    if (name == "statistical")
        return std::make_unique<HistoricalStatisticalPolicy>(capacity,
            historical_policy_detail::statistical_config(params));
    historical_policy_detail::unknown(name);
}

} // namespace vllm_kvmem
