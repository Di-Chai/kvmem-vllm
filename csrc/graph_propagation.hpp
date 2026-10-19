// Immutable logical CSR index and V2 one-hop propagation. Mask-dependent edge
// filtering and denominators are rebuilt for EVERY call, never cached.
#pragma once
#include "block_graph_accept.hpp"
#include <unordered_set>
#include <map>

namespace kvmem_propagation {
namespace py=pybind11;
using kvmem_graph::ExactSum;

class Index {
    std::vector<int64_t> sources,targets,samples,span_starts,span_ends,boundaries;
    std::vector<size_t> offsets{0};
    std::vector<double> weights;

    int classify(int64_t source,int64_t target,int64_t block) const {
        const auto low=std::min(source,target),high=std::max(source,target)+block;
        auto span=std::upper_bound(span_starts.begin(),span_starts.end(),low);
        if (span!=span_starts.begin() && high<=span_ends[span-span_starts.begin()-1]) return 1;
        auto edge=std::upper_bound(boundaries.begin(),boundaries.end(),low);
        return edge!=boundaries.end() && *edge<high ? 2 : 0;
    }

public:
    Index(py::iterable rows,py::iterable spans,py::object ends_known) {
        std::unordered_set<int64_t> seen;
        for (auto item:rows) {
            py::object row=py::reinterpret_borrow<py::object>(item);
            const auto source=row.attr("source").cast<int64_t>();
            auto t=row.attr("targets").cast<std::vector<int64_t>>();
            auto w=row.attr("weights").cast<std::vector<double>>();
            auto sample=row.attr("samples");
            if (!PyLong_CheckExact(sample.ptr()) || sample.cast<int64_t>()<0 ||
                    !seen.insert(source).second || t.size()!=w.size())
                throw std::invalid_argument("Invalid stored graph source row/sample count");
            ExactSum sum;std::unordered_set<int64_t> unique;
            for (size_t i=0;i<t.size();++i) {
                if (!unique.insert(t[i]).second || !std::isfinite(w[i]) || w[i]<0.)
                    throw std::invalid_argument("Invalid stored graph edge mass");
                sum.add(w[i]);
            }
            if (!std::isfinite(sum.value()) || sum.value()>1.+1e-12)
                throw std::invalid_argument("Invalid stored graph edge mass");
            sources.push_back(source);samples.push_back(sample.cast<int64_t>());
            targets.insert(targets.end(),t.begin(),t.end());weights.insert(weights.end(),w.begin(),w.end());
            offsets.push_back(targets.size());
        }
        auto known=ends_known.is_none() ? std::vector<bool>() : ends_known.cast<std::vector<bool>>();
        size_t i=0;
        for (auto item:spans) {
            auto span=py::cast<py::tuple>(item);
            auto begin=span[0].cast<int64_t>(),end=span[1].cast<int64_t>();
            span_starts.push_back(begin);span_ends.push_back(end);boundaries.push_back(begin);
            if (known.empty() || (i<known.size() && known[i])) boundaries.push_back(end);
            ++i;
        }
        std::sort(boundaries.begin(),boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(),boundaries.end()),boundaries.end());
    }

    size_t bytes() const {
        return 4096+8*(sources.capacity()+targets.capacity()+samples.capacity()+offsets.capacity()+
            weights.capacity()+span_starts.capacity()+span_ends.capacity()+boundaries.capacity());
    }

    py::tuple score(const std::vector<double>& raw,const std::vector<int64_t>& starts,
            const std::vector<bool>& eligible,int64_t block,int64_t completed,double wl,double wa,double cross) const {
        if (raw.size()!=starts.size() || raw.size()!=eligible.size() || block<=0)
            throw std::invalid_argument("Graph scoring/index shapes differ");
        std::vector<double> result,r,outgoing;
        std::map<std::string,double> stats;
        {
            py::gil_scoped_release release;
            std::unordered_map<int64_t,size_t> live;
            ExactSum total_sum,raw_sum;
            size_t count=0;
            for (size_t i=0;i<raw.size();++i) {
                if (starts[i]<0 || starts[i]%block || !live.emplace(starts[i],i).second)
                    throw std::invalid_argument("Graph scoring requires distinct original block starts");
                if (!std::isfinite(raw[i]) || raw[i]<0.)
                    throw std::invalid_argument("Graph input scores must be finite and nonnegative");
                raw_sum.add(raw[i]);
                if (eligible[i]) {total_sum.add(raw[i]);++count;}
            }
            const auto total=total_sum.value(),raw_total=raw_sum.value();
            if (!std::isfinite(total) || !std::isfinite(raw_total))
                throw std::invalid_argument("Graph input score sum must be finite");
            stats["raw_eligible_sum"]=total;stats["raw_score_sum"]=raw_total;stats["eligible"]=count;
            stats["identity"]=1.;stats["normalized_input_sum"]=stats["normalized_output_sum"]=total ? 1. : 0.;
            stats["self_retained_mass"]=total ? 1. : 0.;
            result=raw;
            if (count>1 && total>0. && wl+wa>0.) {
                r.resize(raw.size());outgoing.resize(raw.size());
                for (size_t i=0;i<raw.size();++i) r[i]=eligible[i] ? raw[i]/total : 0.;
                result=r;
                size_t movements=0;
                auto transfer=[&](size_t i,size_t j,double edge,double weight,const char* kind) {
                    if (i==j || edge==0. || weight==0.) return;
                    const double delta=weight*r[i]*edge;
                    if (delta==0.) return;
                    ++movements;result[i]-=delta;result[j]+=delta;outgoing[i]+=delta;
                    stats[std::string(kind)+"_external_mass"]+=delta;
                };
                if (wl) for (size_t i=0;i<starts.size();++i) if (eligible[i]) {
                    for (auto target:{starts[i]-block,starts[i]+block}) {
                        const auto category=classify(starts[i],target,block);
                        const auto edge=.5*(category==2 ? cross : 1.);
                        auto found=live.find(target);
                        if (found==live.end() || !eligible[found->second]) {
                            auto kind=found==live.end() ? "missing" : "masked";
                            stats[std::string("locality_")+kind+"_count"]+=1;
                            stats[std::string(kind)+"_self_mass"]+=wl*r[i]*.5;
                            continue;
                        }
                        stats[category==0 ? "locality_fallback_count" : category==1 ?
                            "locality_same_message_count" : "locality_cross_message_count"]+=1;
                        transfer(i,found->second,edge,wl,"locality");
                    }
                }
                if (wa) for (size_t row=0;row<sources.size();++row) {
                    auto source=live.find(sources[row]);
                    if (source==live.end() || !eligible[source->second] || samples[row]==0 ||
                            sources[row]+block>completed) continue;
                    const auto i=source->second;
                    ExactSum retained,surviving;
                    for (size_t e=offsets[row];e<offsets[row+1];++e) {
                        retained.add(weights[e]);
                        if (targets[e]==sources[row] || weights[e]==0.) continue;
                        auto target=live.find(targets[e]);
                        if (target==live.end() || !eligible[target->second]) {
                            stats[target==live.end() ? "attention_missing_count" : "attention_masked_count"]+=1;
                        } else surviving.add(weights[e]);
                    }
                    const double denominator=surviving.value();
                    stats["attention_retained_raw_mass"]+=r[i]*retained.value();
                    stats["attention_excluded_raw_mass"]+=r[i]*(retained.value()-denominator);
                    if (denominator>0.) {
                        stats["attention_seed_coverage"]+=r[i];
                        stats["attention_effective_raw_mass"]+=r[i]*denominator;
                        for (size_t e=offsets[row];e<offsets[row+1];++e) {
                            auto target=live.find(targets[e]);
                            if (target!=live.end() && eligible[target->second] && targets[e]!=sources[row] && weights[e]>0.)
                                transfer(i,target->second,weights[e]/denominator,wa,"attention");
                        }
                    }
                }
                if (!movements) result=raw;
                else {
                    ExactSum check,seed_sum,out_sum,normal_l1,raw_l1;
                    for (auto value:result) {
                        if (!std::isfinite(value) || value < -1e-14)
                            throw std::runtime_error("Graph propagation violated nonnegativity or mass conservation");
                        check.add(value);
                    }
                    if (std::abs(check.value()-1.)>1e-12)
                        throw std::runtime_error("Graph propagation violated nonnegativity or mass conservation");
                    ExactSum normalized;
                    for (size_t i=0;i<raw.size();++i) {
                        result[i]=std::max(0.,result[i]);normalized.add(result[i]);
                        normal_l1.add(std::abs(result[i]-r[i]));seed_sum.add(r[i]);out_sum.add(outgoing[i]);
                        result[i]=eligible[i] ? result[i]*total : raw[i];raw_l1.add(std::abs(result[i]-raw[i]));
                    }
                    stats["identity"]=0.;stats["normalized_output_sum"]=normalized.value();
                    stats["self_retained_mass"]=seed_sum.value()-out_sum.value();
                    stats["normalized_l1_change"]=normal_l1.value();stats["raw_l1_change"]=raw_l1.value();
                }
            }
            ExactSum final_sum;for (double value:result) final_sum.add(value);
            stats["propagated_score_sum"]=final_sum.value();
        }
        return py::make_tuple(result,stats);
    }
};

inline void bind(py::module_& module) {
    py::class_<Index>(module,"GraphPropagationIndex")
        .def(py::init<py::iterable,py::iterable,py::object>())
        .def_property_readonly("bytes",&Index::bytes)
        .def("score",&Index::score);
}
}
