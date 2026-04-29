import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def select_id_subset(items, limit, mode, seed):
    if limit <= 0 or limit >= len(items):
        return list(items)
    if mode == 'head':
        return list(items[:limit])

    rng = random.Random(seed)
    sampled = set(rng.sample(list(items), limit))
    return [item for item in items if item in sampled]


def filter_probe_set(probe_entries, selected_test_ids, max_probe_per_test):
    grouped = defaultdict(list)
    selected_test_ids = list(selected_test_ids)

    for entry in probe_entries:
        matched_pid = None
        for pid in selected_test_ids:
            prefix = pid + '-'
            if entry.startswith(prefix):
                matched_pid = pid
                break
        if matched_pid is not None:
            grouped[matched_pid].append(entry)

    kept = []
    for pid in selected_test_ids:
        seqs = grouped.get(pid, [])
        if max_probe_per_test > 0:
            seqs = seqs[:max_probe_per_test]
        kept.extend(seqs)
    return kept


def main():
    parser = argparse.ArgumentParser(
        description='Create a small subset partition JSON for the company dataset.'
    )
    parser.add_argument(
        '--input-json',
        default='all_data_20250912_add2503_noise.json',
        help='Source partition JSON path.'
    )
    parser.add_argument(
        '--output-json',
        default='all_data_20250912_add2503_noise_mini.json',
        help='Output subset partition JSON path.'
    )
    parser.add_argument(
        '--train-ids',
        type=int,
        default=256,
        help='Number of identities to keep in TRAIN_SET. 0 keeps all.'
    )
    parser.add_argument(
        '--test-ids',
        type=int,
        default=32,
        help='Number of identities to keep in TEST_SET. 0 keeps all.'
    )
    parser.add_argument(
        '--max-probe-per-test',
        type=int,
        default=8,
        help='Maximum number of probe sequences kept for each selected test identity. 0 keeps all.'
    )
    parser.add_argument(
        '--train-test-ids',
        type=int,
        default=0,
        help='Number of identities to keep in TRAIN_TEST_SET. 0 keeps all.'
    )
    parser.add_argument(
        '--mode',
        choices=['head', 'random'],
        default='random',
        help='Subset selection strategy for TRAIN_SET and TEST_SET.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=3407,
        help='Random seed used when mode=random.'
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    with input_path.open('r', encoding='utf-8') as file:
        partition = json.load(file)

    train_set = partition['TRAIN_SET']
    test_set = partition['TEST_SET']
    train_test_set = partition.get('TRAIN_TEST_SET', [])
    probe_set = partition.get('PROBE_SET', [])

    selected_train = select_id_subset(train_set, args.train_ids, args.mode, args.seed)
    selected_test = select_id_subset(test_set, args.test_ids, args.mode, args.seed + 1)
    selected_train_test = select_id_subset(train_test_set, args.train_test_ids, args.mode, args.seed + 2)
    selected_probe = filter_probe_set(
        probe_set,
        selected_test,
        args.max_probe_per_test,
    )

    subset_partition = {
        'TRAIN_SET': selected_train,
        'TEST_SET': selected_test,
        'TRAIN_TEST_SET': selected_train_test,
        'PROBE_SET': selected_probe,
        'SUBSET_INFO': {
            'source_json': str(input_path).replace('\\', '/'),
            'mode': args.mode,
            'seed': args.seed,
            'train_ids': len(selected_train),
            'test_ids': len(selected_test),
            'train_test_ids': len(selected_train_test),
            'probe_sequences': len(selected_probe),
            'max_probe_per_test': args.max_probe_per_test,
        }
    }

    with output_path.open('w', encoding='utf-8') as file:
        json.dump(subset_partition, file, ensure_ascii=False, indent=2)

    print('Saved subset partition to:', output_path)
    print('TRAIN_SET:', len(selected_train))
    print('TEST_SET:', len(selected_test))
    print('TRAIN_TEST_SET:', len(selected_train_test))
    print('PROBE_SET:', len(selected_probe))


if __name__ == '__main__':
    main()