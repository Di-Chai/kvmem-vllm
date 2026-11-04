#include "historical_thinking.hpp"
#include <iostream>
#include <stdexcept>

using vllm_kvmem::HistoricalThinkingParser;
static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
static HistoricalThinkingParser parser() {
    return {{1, 2, 3, 4}, {{10, 20}, {11, 20}, {12, 20}, {13, 20}, {14, 20}}};
}
int main() {
    auto p = parser();
    check(p.feed({90, 3, 91}) == std::vector<uint8_t>({0, 0, 0}), "unknown input became thinking");
    check(p.feed({1, 12, 20, 3, 91, 4, 2}) == std::vector<uint8_t>({1, 1, 1, 1, 1, 1, 1}),
          "user quoted markers changed thinking");
    p.seed({1, 13});
    check(p.feed({20, 3}) == std::vector<uint8_t>({1, 1}) && p.thinking(), "open template prefix lost");
    check(p.feed({90, 91}) == std::vector<uint8_t>({2, 2}), "accepted thinking body missing");
    // A verifier could have proposed </think> after these rows. It is not fed
    // until accepted and must not change the committed state in the meantime.
    check(p.feed({92}) == std::vector<uint8_t>({2}), "rejected marker changed state");
    check(p.feed({4, 93}) == std::vector<uint8_t>({1, 1}) && !p.thinking(), "answer remained thinking");
    p.seed({2, 1, 14, 20});
    check(p.feed({3, 94, 4}) == std::vector<uint8_t>({1, 1, 1}), "tool output became thinking");
    p.seed({2, 1, 13, 20, 3});
    check(p.feed({95}) == std::vector<uint8_t>({2}), "tool continuation lost fresh thinking prefix");
    p.reset();
    check(p.feed({95}) == std::vector<uint8_t>({0}), "reset retained future thinking state");
    p.seed({1, 13, 20, 3, 4});
    check(p.feed({95}) == std::vector<uint8_t>({1}), "empty disabled-thinking prefix misclassified");
    p.seed({1, 99, 20, 3});
    check(p.feed({95}) == std::vector<uint8_t>({0}), "unknown role became thinking");
    bool rejected = false;
    try { HistoricalThinkingParser bad({1, 2, 3, 3}, {{10}, {11}, {12}, {13}, {14}}); }
    catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "ambiguous markers accepted");
    std::cout << "historical thinking parser passed\n";
}
