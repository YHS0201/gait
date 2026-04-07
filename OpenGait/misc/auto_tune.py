import argparse
import copy
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def dump_yaml(data, path):
    with open(path, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def set_nested(cfg, dotted_key, value):
    keys = dotted_key.split('.')
    cursor = cfg
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def get_nested(cfg, dotted_key, default=None):
    keys = dotted_key.split('.')
    cursor = cfg
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def sample_parameter(spec, rng):
    spec_type = spec['type']
    if spec_type == 'choice':
        return rng.choice(spec['values'])
    if spec_type == 'int':
        low = int(spec['low'])
        high = int(spec['high'])
        step = int(spec.get('step', 1))
        values = list(range(low, high + 1, step))
        return rng.choice(values)
    if spec_type == 'float':
        low = float(spec['low'])
        high = float(spec['high'])
        if spec.get('log', False):
            log_low = math.log(low)
            log_high = math.log(high)
            value = math.exp(rng.uniform(log_low, log_high))
        else:
            value = rng.uniform(low, high)
        step = spec.get('step')
        if step is not None:
            step = float(step)
            value = round(round((value - low) / step) * step + low, 10)
        return value
    raise ValueError(f'Unsupported parameter type: {spec_type}')


def build_trial_params(search_space, rng):
    params = {}
    for key, spec in search_space['parameters'].items():
        params[key] = sample_parameter(spec, rng)
    return params


def apply_params(base_cfg, params, trial_name, cli_args, set_save_name=True):
    cfg = copy.deepcopy(base_cfg)
    for key, value in params.items():
        set_nested(cfg, key, value)

    trainer_cfg = cfg.setdefault('trainer_cfg', {})
    evaluator_cfg = cfg.setdefault('evaluator_cfg', {})

    if set_save_name:
        trainer_cfg['save_name'] = trial_name
        evaluator_cfg['save_name'] = trial_name

    if cli_args.total_iter is not None:
        trainer_cfg['total_iter'] = int(cli_args.total_iter)
    if cli_args.save_iter is not None:
        trainer_cfg['save_iter'] = int(cli_args.save_iter)
    if cli_args.restore_hint is not None:
        evaluator_cfg['restore_hint'] = int(cli_args.restore_hint)
    else:
        evaluator_cfg['restore_hint'] = int(trainer_cfg['total_iter'])

    if cli_args.with_test is not None:
        trainer_cfg['with_test'] = bool(cli_args.with_test)

    return cfg


def build_checkpoint_path(train_cfg):
    dataset_name = train_cfg['data_cfg']['dataset_name']
    model_name = train_cfg['model_cfg']['model']
    save_name = train_cfg['trainer_cfg']['save_name']
    restore_hint = train_cfg['evaluator_cfg']['restore_hint']
    return os.path.join(
        'output',
        dataset_name,
        model_name,
        save_name,
        'checkpoints',
        f'{save_name}-{int(restore_hint):0>5}.pt'
    )


def run_command(command, env, cwd, log_path):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    captured = []
    with open(log_path, 'w', encoding='utf-8') as log_file:
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            captured.append(line)
    return_code = process.wait()
    output = ''.join(captured)
    if return_code != 0:
        raise RuntimeError(f'Command failed with exit code {return_code}: {" ".join(command)}')
    return output


def parse_metric_from_output(output, metric_cfg):
    mode = metric_cfg.get('mode', 'regex')
    if mode == 'regex':
        pattern = metric_cfg['pattern']
        flags = re.MULTILINE
        match = re.search(pattern, output, flags)
        if not match:
            raise ValueError(f'Failed to parse metric with pattern: {pattern}')
        group = int(metric_cfg.get('group', 1))
        return float(match.group(group))

    if mode == 'ccpg_rank1_mean':
        marker = '===Rank-1 (Exclude identical-view cases)==='
        lines = output.splitlines()
        for idx, line in enumerate(lines):
            if marker in line and idx + 1 < len(lines):
                next_line = lines[idx + 1]
                values = re.findall(r'([A-Z]{2}):\s*([0-9]+(?:\.[0-9]+)?)', next_line)
                if len(values) != 4:
                    continue
                scores = [float(item[1]) for item in values]
                return sum(scores) / len(scores)
        raise ValueError('Failed to parse CCPG Rank-1 mean from output.')

    if mode == 'casiab_rank1_mean':
        lines = output.splitlines()
        for line in lines:
            values = re.findall(r'([A-Z]{2})@R1:\s*([0-9]+(?:\.[0-9]+)?)%', line)
            if len(values) < 3:
                continue
            score_dict = {name: float(value) for name, value in values}
            required = ['NM', 'BG', 'CL']
            if not all(name in score_dict for name in required):
                continue
            scores = [score_dict[name] for name in required]
            return sum(scores) / len(scores)
        raise ValueError('Failed to parse CASIA-B Rank-1 mean from output.')

    raise ValueError(f'Unsupported metric mode: {mode}')


def build_launch_command(args, cfg_path, phase):
    command = [
        sys.executable,
        '-m',
        'torch.distributed.launch',
        f'--nproc_per_node={args.nproc_per_node}',
        'opengait/main.py',
        '--cfgs',
        str(cfg_path),
        '--phase',
        phase,
    ]
    if args.log_to_file:
        command.append('--log_to_file')
    return command


def main():
    parser = argparse.ArgumentParser(description='Automatic hyper-parameter tuning for OpenGait configs.')
    parser.add_argument('--base_cfg', default=None, help='Single YAML config path used for both train and test.')
    parser.add_argument('--train_cfg', default=None, help='Train YAML config path.')
    parser.add_argument('--test_cfg', default=None, help='Test YAML config path for cross-dataset evaluation.')
    parser.add_argument('--search_space', required=True, help='YAML/JSON search space file path.')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials to run.')
    parser.add_argument('--gpus', default='0', help='CUDA_VISIBLE_DEVICES value, e.g. 0,1,2.')
    parser.add_argument('--nproc_per_node', type=int, default=1, help='Distributed process count.')
    parser.add_argument('--work_dir', default='output/tuning_runs', help='Directory for temp configs and reports.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampler.')
    parser.add_argument('--total_iter', type=int, default=None, help='Override trainer_cfg.total_iter for tuning.')
    parser.add_argument('--save_iter', type=int, default=None, help='Override trainer_cfg.save_iter for tuning.')
    parser.add_argument('--restore_hint', type=int, default=None, help='Override evaluator_cfg.restore_hint.')
    parser.add_argument('--with_test', type=int, choices=[0, 1], default=None, help='Override trainer_cfg.with_test.')
    parser.add_argument('--log_to_file', action='store_true', help='Pass --log_to_file to OpenGait commands.')
    parser.add_argument('--skip_test', action='store_true', help='Use training log metric only and skip explicit test command.')
    parser.add_argument('--sync_test_params', action='store_true', help='Apply sampled params to test config as well.')
    args = parser.parse_args()

    if args.base_cfg is None and args.train_cfg is None:
        raise ValueError('Either --base_cfg or --train_cfg must be provided.')
    if args.base_cfg is not None and args.train_cfg is not None:
        raise ValueError('Use either --base_cfg or --train_cfg, not both.')
    if args.skip_test and args.test_cfg is not None:
        raise ValueError('--skip_test cannot be used together with --test_cfg.')

    work_dir = Path(args.work_dir)
    configs_dir = work_dir / 'configs'
    logs_dir = work_dir / 'logs'
    work_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_cfg_path = args.train_cfg or args.base_cfg
    train_base_cfg = load_yaml(train_cfg_path)
    test_base_cfg = load_yaml(args.test_cfg) if args.test_cfg else None
    search_space = load_yaml(args.search_space)
    metric_cfg = search_space['metric']

    rng = random.Random(args.seed)
    best_result = None
    history = []

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpus

    for trial_index in range(args.trials):
        trial_id = trial_index + 1
        trial_name = f"{search_space.get('study_name', 'tune')}_trial_{trial_id:03d}"
        params = build_trial_params(search_space, rng)
        train_cfg = apply_params(train_base_cfg, params, trial_name, args, set_save_name=True)
        train_cfg_path_trial = configs_dir / f'{trial_name}_train.yaml'
        dump_yaml(train_cfg, train_cfg_path_trial)

        test_cfg_path_trial = None
        if test_base_cfg is not None:
            test_params = params if args.sync_test_params else {}
            test_cfg = apply_params(test_base_cfg, test_params, trial_name, args, set_save_name=False)
            test_cfg['evaluator_cfg']['restore_hint'] = build_checkpoint_path(train_cfg)
            test_cfg_path_trial = configs_dir / f'{trial_name}_test.yaml'
            dump_yaml(test_cfg, test_cfg_path_trial)

        train_log_path = logs_dir / f'{trial_name}_train.log'
        train_command = build_launch_command(args, train_cfg_path_trial, 'train')
        start_time = time.time()
        try:
            train_output = run_command(train_command, env, Path.cwd(), train_log_path)

            if args.skip_test:
                metric_value = parse_metric_from_output(train_output, metric_cfg)
            else:
                test_log_path = logs_dir / f'{trial_name}_test.log'
                active_test_cfg_path = test_cfg_path_trial or train_cfg_path_trial
                test_command = build_launch_command(args, active_test_cfg_path, 'test')
                test_output = run_command(test_command, env, Path.cwd(), test_log_path)
                metric_value = parse_metric_from_output(test_output, metric_cfg)

            elapsed = time.time() - start_time
            result = {
                'trial_id': trial_id,
                'trial_name': trial_name,
                'metric': metric_value,
                'params': params,
                'elapsed_sec': elapsed,
                'status': 'ok',
            }
            history.append(result)

            if best_result is None or metric_value > best_result['metric']:
                best_result = result

            report = {'best': best_result, 'history': history}
            with open(work_dir / 'results.json', 'w', encoding='utf-8') as handle:
                json.dump(report, handle, indent=2, ensure_ascii=False)

            print(f'[trial {trial_id}] metric={metric_value:.4f}, best={best_result["metric"]:.4f}')
        except Exception as exc:
            elapsed = time.time() - start_time
            history.append({
                'trial_id': trial_id,
                'trial_name': trial_name,
                'metric': None,
                'params': params,
                'elapsed_sec': elapsed,
                'status': 'failed',
                'error': str(exc),
            })
            with open(work_dir / 'results.json', 'w', encoding='utf-8') as handle:
                json.dump({'best': best_result, 'history': history}, handle, indent=2, ensure_ascii=False)
            print(f'[trial {trial_id}] failed: {exc}', file=sys.stderr)

    if best_result is None:
        raise SystemExit('All trials failed.')

    print('\nBest trial:')
    print(json.dumps(best_result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    import math
    main()