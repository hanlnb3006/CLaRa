#!/usr/bin/env python3
import argparse, json, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=200)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.dst), exist_ok=True)
    written = 0
    with open(args.src, 'r', encoding='utf-8') as fh_in, open(args.dst, 'w', encoding='utf-8') as fh_out:
        for line in fh_in:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pos = item.get('pos_index')
            if pos is None or (isinstance(pos, list) and len(pos) == 0):
                continue
            pos_val = pos[0] if isinstance(pos, list) else int(pos)
            record = {
                'question': item['question'],
                'docs': item['docs'],
                'pos_index': pos_val,
            }
            fh_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            written += 1
            if written >= args.max_samples:
                break
    print(f'Wrote {written} records to {args.dst}')

if __name__ == '__main__':
    main()
