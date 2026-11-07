"""Verify frozen raw evidence and export explicitly TEXT-only paired inputs.

No network/GPU calls. Text ablation is not a substitute for historical KV
eviction: native per-layer paired replay is a separate, still pending test.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from protocol import PROBES, POLICY, canonical, digest, history, observed_failure, quantile


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def resolve(obj, pointer):
    for p in pointer.split('/')[1:]:
        p = p.replace('~1','/').replace('~0','~')
        obj = obj[int(p)] if isinstance(obj,list) else obj[p]
    return obj


def run(bundle, dataset, output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'text-ablation-inputs').mkdir()
    sources = json.loads((dataset/'source_hashes.json').read_text())
    for path, expected in sources.items():
        if digest((bundle/path).read_text()) != expected:
            raise ValueError('Source hash mismatch: '+path)
    snapshots, raw, n_units, n_edges = {}, {}, 0, 0
    ranking_rows, edge_types = [], Counter()
    for path in sorted((dataset/'snapshots').glob('*.json')):
        s = json.loads(path.read_text())
        snapshots[s['snapshot_id']] = s
        units = {u['unit_id']:u for u in s['units']}
        if len(units) != len(s['units']):
            raise ValueError('Duplicate unit within a snapshot')
        for u in s['units']:
            if u['trace_path'] not in raw:
                raw[u['trace_path']] = [json.loads(l) for l in (bundle/u['trace_path']).read_text().splitlines()]
            record = raw[u['trace_path']][u['trace_line']-1]
            field = resolve(record, u['field_pointer'])
            begin, end = u['character_span']
            if field[begin:end] != u['text'] or digest(u['text']) != u['text_sha256']:
                raise ValueError('Raw source coordinate or hash mismatch')
            if u['step'] > s['checkpoint_step']:
                raise ValueError('Future evidence in input')
        text, spans = history(s['units'])
        if text != s['rendered_history'] or spans != s['body_character_spans']:
            raise ValueError('Rendering mismatch')
        for e in s['edges']:
            source, target = units[e['source']], units[e['target']]
            if source['step'] < target['step']:
                raise ValueError('Invalid temporal edge')
            if e['relation']=='redundant_with' and canonical(source['text']) != canonical(target['text']):
                raise ValueError('Invalid exact duplicate')
            if e['relation']=='entity_association' and e['evidence'] not in set(source['entities']) & set(target['entities']):
                raise ValueError('Unsubstantiated entity edge')
            if e['relation']=='tool_result_of' and not source.get('call_id') == target.get('call_id'):
                raise ValueError('Wrong tool/result association')
            edge_types[e['relation']] += 1
        n_units += len(units); n_edges += len(s['edges'])
        methods = ('lru_proxy','redundancy','graph_rules','graph_propagation')
        top = {method:[r['unit_id'] for r in sorted(s['scores'], key=lambda r:(-r[method], r['unit_id']))[:10]] for method in methods}
        ranking_rows.append(dict(snapshot_id=s['snapshot_id'], k=10, top=top,
            graph_vs_lru_topk_intersection=len(set(top['graph_propagation']) & set(top['lru_proxy'])),
            graph_vs_redundancy_topk_intersection=len(set(top['graph_propagation']) & set(top['redundancy'])),
            attention_ranking=None, precision_at_k=None, recall_at_k=None,
            note='Ranking differences do not demonstrate better safe-deletion identification. LRU uses observed entity/tool links, not real KV-access timestamps.'))
    candidates = [json.loads(l) for l in (dataset/'candidates.jsonl').read_text().splitlines()]
    if len(candidates)!=24 or len({c['candidate_id'] for c in candidates})!=24:
        raise ValueError('Expected 24 unique candidates')
    conditions = Counter()
    for c in candidates:
        s = snapshots[c['snapshot_id']]
        match = [u for u in s['units'] if u['unit_id']==c['unit_id']]
        if len(match)!=1 or match[0]['text']!=c['text']:
            raise ValueError('Candidate not in frozen snapshot')
        if c['category']=='error_observation' and not observed_failure(match[0]):
            raise ValueError('Error category lacks observed failure signal')
        full, _ = history(s['units'])
        reduced, _ = history([u for u in s['units'] if u['unit_id']!=c['unit_id']])
        removed, _ = history(match)
        # The exact annotated unit is the sole removal. Other units/ordering
        # and all six query strings stay identical.
        if len(full)-len(reduced)!=len(removed) or full.replace(removed,'',1)!=reduced:
            raise ValueError('Pair has an unintended intervention')
        conditions['left_censored_window' if s['left_censored'] else 'eligible_prefix'] += 1
        for arm, content in [('full',full),('reduced',reduced)]:
            save(output/'text-ablation-inputs'/f'{c["candidate_id"]}.{arm}.json', dict(
                candidate_id=c['candidate_id'], arm=arm, intervention='text_ablation_reprefill_only',
                checkpoint_step=s['checkpoint_step'], left_censored=s['left_censored'],
                history=content, history_sha256=digest(content), system=POLICY, questions=PROBES,
                physical_kv_eviction=False, token_positions_status='not_tokenized'))
    save(output/'rankings.json', ranking_rows)
    metrics = dict(passed=True, source_files=len(sources), snapshots=len(snapshots),
        unit_occurrences_checked=n_units, edge_occurrences_checked=n_edges, edge_types=edge_types,
        candidates=24, text_pairs=24, candidate_conditions=conditions,
        unique_candidate_texts=len({c['text_cluster_sha256'] for c in candidates}),
        cpu_host_only=True, gpu_execution=False,
        graph_build_ms_p50=quantile([s['graph_build_seconds']*1000 for s in snapshots.values()],.5),
        graph_build_ms_p95=quantile([s['graph_build_seconds']*1000 for s in snapshots.values()],.95),
        graph_ranking_ms_p50=quantile([s['graph_rank_seconds']*1000 for s in snapshots.values()],.5))
    save(output/'audit.json',metrics)
    save(output/'execution-status.json',dict(
        status='gpu_blocked_on_ssh_authentication', gpu_runs=0,
        endpoint='ssh-cn-xibei2.ebcloud.com:32836', login_user='root',
        observed_error='Permission denied (publickey,password).',
        prior_control_socket='/tmp/codex-observer-4090-control',
        old_server_configuration_is_not_current_health_evidence=True,
        remaining=['GPU identity/service health check', 'actual token/physical-page mapping',
                   'measured attention edges', 'frozen-KV per-layer Attention and Transformer outputs',
                   'six-probe generation plus native categorical System One consistency evaluation',
                   'independent semantic reference labels and false-safe rate',
                   'complementarity analysis', 'CPU/SSD KV restore and full-prefix reconstruction costs',
                   'serving concurrency/input/output matrix']))
    save(output/'serving-matrix.json',dict(
        status='not_executed', concurrency=[1,2,4,8,16], input_tokens=[4096,8192,16384,32768],
        output_tokens=[256,512,1024], request_type=['observer_generation','full_reduced_pair_generation'],
        native_systemone_output_tokens_per_six_questions=6,
        notes=['Native /v1/systemone reads categorical labels and cannot generate 256–1024 tokens.',
               'Generative observer and categorical consistency judge must be timed separately.',
               'Record actual token counts, failures, queueing and supported context capacities.',
               'Text ablation and actual KV eviction must not be pooled in impact metrics.']))
    summary=json.loads((dataset/'summary.json').read_text())
    totals=summary['totals']
    report=f'''# Graph-guided Cleanup 初步验证：已执行部分与阻塞项

日期：2026-11-05（Asia/Shanghai）。GPU 实验尚未执行。4090 的旧 SSH 连接已失效；连接 `root@ssh-cn-xibei2.ebcloud.com:32836` 返回认证失败。没有停止远端服务或删除线上 KV。

## 已执行的离线统计

重新读取现有 16 个任务 × 4 Seeds，共 64 条 DeepSWE 轨迹。所有纳入片段已核对原始 JSON 字段、字符坐标与内容哈希。排除嵌入工具返回的框架提醒 {summary['excluded_framework_reminder_units']:,} 个片段。

共 {totals['all_units']:,} 个合格结构片段；其中空白归一后至少 80 个字符的片段 {totals['eligible_units_80chars']:,} 个，作为重复率分母。

| 指标 | 分子 / 分母 | 比例 |
|---|---:|---:|
| 同轨迹中此前已有逐字相同片段 | {totals['earlier_byte_duplicate']:,} / {totals['eligible_units_80chars']:,} | {summary['byte_duplicate_ratio']:.2%} |
| 同轨迹中此前已有空白归一后相同片段 | {totals['earlier_whitespace_duplicate']:,} / {totals['eligible_units_80chars']:,} | {summary['whitespace_duplicate_ratio']:.2%} |

统计以片段计数，不是 Token 比例、KV 字节比例、语义冗余率或可释放容量。四个 Seed 的同一任务并非独立任务样本；目前没有给出独立人工标签或置信区间。

## Graph 与候选

冻结 {len(snapshots)} 个快照，包含 {n_units:,} 次节点出现和 {n_edges:,} 条边出现。较晚的失败窗口保留原始任务约束和预算内连续完整 Steps，并明确标记左截断；重叠窗口中的节点不能重复算作独立信息。

边目前仅有空白归一精确重复、5-gram 词面重叠、实体关联和实际 tool-call/result ID 关系。实体关联不表示语义依赖，词面重叠不表示完整替代，尚未测量 Attention 或推断 Supersedes 边。

保存了 LRU 代理、重复度、Graph 规则、三轮传播的 Top-10 排序。LRU 代理使用已观测的实体/工具关联时间，不是实际 KV 访问时间。排序不同不能证明 Graph 更好；Precision@K、Recall@K、Edge Precision/Recall 保持空值。

本地主机上的图构建 P50/P95 为 {metrics['graph_build_ms_p50']:.2f}/{metrics['graph_build_ms_p95']:.2f} ms，排序 P50 为 {metrics['graph_ranking_ms_p50']:.2f} ms。单次测量仅供实现规模参考，不是 4090 服务或增量更新性能。

24 个候选分六类，每类四个：精确重复、部分词面重叠、真实工具失败、代码观察、没有替代证据的 Thinking、用户约束对照。精确重复类仅来自两个任务；24 个候选共有 {metrics['unique_candidate_texts']} 种不同文本，不可视为 24 个独立样本。

20 个候选采用完整的合格历史前缀，4 个错误候选采用已标记的历史窗口。这里 Full 仅指冻结的研究输入，不表示完整原始 Agent Prompt。每对只去除指定信息单元，其余文本和六项 Probe 相同。

## 六个实验的完成情况

| 实验 | 当前状态 |
|---|---|
| 1. 冗余与替代 | 已完成精确重复统计；语义替代、Embedding、人工 Precision 待测 |
| 2. Graph 候选 | 已构图并保存四种排序与 24 个候选；Attention、真实 KV 映射及质量比较待测 |
| 3. 逐层 Output Change | 未运行；需要原始位置一致的真实 KV 干预，分别记录 Attention 与 Layer Output |
| 4. System One 语义一致性 | 未运行；已冻结六项问题和文本消融输入，未产生模型回答或语义标签 |
| 5. 两项指标互补性 | 未运行；依赖实验 3/4 和独立参考标签 |
| 6. 恢复与服务性能 | 未运行；已保存 1/2/4/8/16 并发、4K/8K/16K/32K 输入、256/512/1024 输出矩阵 |

## 接续时必须保持的实验条件

先恢复 SSH 并核对 GPU、服务配置和正在运行的请求。此前文档中的服务是 27B GPTQ Int4、FP8 KV、混合 recurrent/attention 架构，不能将旧报告当作当前健康状态。

真实 KV 干预需要冻结同一份 Full 历史缓存、recurrent 状态及 Query 绝对位置，Reduced 只移除选定历史 attention KV；保留的 recurrent 状态仍可能携带被删信息，必须报告此边界。先通过无删除配对与重复运行检查，再测删除。按整 Block 删除时必须记录共同承载的其他信息单元；逻辑 Block 号不能冒充实际物理页号。

文本删除并重新 Prefill 是另一个诊断条件；本目录的 `text-ablation-inputs/` 仅属于这个条件。它会改变后续上下文计算，不能代替实际 KV 删除实验。

现有原生 `/v1/systemone` 为类别读取，每题输出一个标签，六题为六个输出 Token。六项开放问答需要独立生成路径，再由原生接口做类别一致性判断；必须分别报告生成与判定成本，不能将 256/512/1024 Token 生成压测称作原生标签接口压测。

False-safe、关键遗漏率、联合分类效果均需要独立参考标签。当前 `human-review.csv` 是待填写模板，未将模型或自动规则冒充人工答案。

## 审计与复现

6 项针对原文定位、未来信息泄漏、共享 Block、工具关联、错误识别和框架提醒的 CPU 测试通过。`audit.json` 保存原始证据与配对输入的核查结果；全部原始来源哈希见上一级 `source_hashes.json`。GPU 运行数为 0。

主数据位于上一级目录；`rankings.json` 为排序结果，`execution-status.json` 为阻塞项，`serving-matrix.json` 为未执行的压测配置。r1 是候选分布检查失败的记录；r2/r3 是发现标签问题后的中间输出，不能用作最终结果。最终口径为 r4。
'''
    (output/'REPORT.zh-CN.md').write_text(report)
    print(json.dumps(metrics,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.bundle,a.dataset,a.output)
