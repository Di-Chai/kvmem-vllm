from analyze_publication_bottleneck import host_analysis


def test_nested_time_is_not_double_counted_across_threads():
    rows=[dict(label='parent',tid=1,start_ns=0,end_ns=100,cpu_ns=80),
          dict(label='child',tid=1,start_ns=10,end_ns=70,cpu_ns=50),
          dict(label='leaf',tid=1,start_ns=20,end_ns=40,cpu_ns=10),
          dict(label='parallel',tid=2,start_ns=0,end_ns=100,cpu_ns=30)]
    result={r['label']:r for r in host_analysis(rows)['labels']}
    assert result['parent']['exclusive_wall_ms']==40/1e6
    assert result['child']['exclusive_wall_ms']==40/1e6
    assert result['parent']['exclusive_cpu_ms']==30/1e6
    assert result['child']['exclusive_cpu_ms']==40/1e6
    assert result['parallel']['exclusive_wall_ms']==100/1e6
