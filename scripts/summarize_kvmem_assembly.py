"""Aggregate an isolated batch's bounded_refresh rows without request contents."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--begin-row',type=int,default=0)
    parser.add_argument('--end-row',type=int)
    args=parser.parse_args()
    all_rows=[json.loads(line) for line in (args.capture/'bounded_refresh.jsonl').read_text().splitlines()]
    end=len(all_rows) if args.end_row is None else args.end_row
    if not 0<=args.begin_row<end<=len(all_rows):
        raise ValueError('Require a nonempty exact batch row range')
    rows=all_rows[args.begin_row:end]
    result={'scope':'Assembly path host preparation and waits, not pure GPU kernel time; payload H2D excludes small metadata',
        'begin_row':args.begin_row,'end_row':end,'refreshes':len(rows),
        'distinct_requests':len({row['request_id'] for row in rows})}
    fields={'uploaded_bytes':'assembly_uploaded_bytes','assembly_seconds':'assembly_seconds',
        'host_prepare_seconds':'assembly_host_prepare_seconds','rebuilt_tokens':'rebuilt_tokens',
        'skipped_tokens':'skipped_tokens','raw_cache_hit_tokens':'raw_cache_hit_tokens',
        'resident_v_reused_tokens':'resident_v_reused_tokens'}
    for name,field in fields.items():
        result[name]=sum(row.get(field,0) for row in rows)
    result['raw_cache_capacity_bytes']=max(row.get('raw_cache_capacity_bytes',0) for row in rows)
    result['resident_value_max_request_bytes']=max(row.get('assembly_device_temp_bytes',0) for row in rows)
    progress=args.capture/'progress.json'
    if progress.exists():
        data=json.loads(progress.read_text())
        result['cumulative_cpu_scatter']={key:data[key] for key in
            ('cpu_scatter_batches','cpu_scatter_seconds','cpu_scatter_join_seconds') if key in data}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
