"""Evaluate saved tiny-overfit weights on fresh states of the SAME environments.

Run from the appropriate full or nominal project root. No training or source edits.
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
import dataset.generator as generator
from model.estimator import ChannelEstimator
from training.trainer import nmse_per_sample, nmse_db


def make_data(simulation, repeat=None, noise_only=False):
    """Replay original RNG consumption, substitute independent state/noise draws.

    The original generator interleaves path and measurement RNG draws. Merely
    replacing its seed would change the paths. Consuming the original draws
    before substituting new outputs preserves that sequence exactly.
    """
    deform = generator.generate_deformation_codebook
    transmit = generator.transmit_pilots
    build = generator.build_channel
    physical_targets = []
    call = 0
    if repeat is not None:
        state_rng = np.random.default_rng(np.random.SeedSequence([88201, repeat, 0]))
        noise_rng = np.random.default_rng(np.random.SeedSequence([88201, repeat, 1]))

    def deformations(*args):
        original = deform(*args)
        if repeat is None or noise_only:
            return original
        return deform(*args[:-1], state_rng)

    def measurements(*args):
        original = transmit(*args)
        if repeat is None:
            return original
        return transmit(*args[:-1], noise_rng)

    def channels(*args):
        nonlocal call
        result = build(*args)
        if call % (simulation['n_deformations'] + 1) == 0:
            physical_targets.append(result.copy())
        call += 1
        return result

    with patch.object(generator, 'generate_deformation_codebook', deformations), \
         patch.object(generator, 'transmit_pilots', measurements), \
         patch.object(generator, 'build_channel', channels):
        arrays = generator.generate_dataset(**simulation, n_channels=4,
            b_min=0.05, b_max=0.30, seed=77123)
    return arrays, physical_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    config = json.loads(args.config.read_text())
    simulation = dict(config['simulation'], n_subcarriers=2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = ChannelEstimator(**checkpoint['model_config']).to(args.device).eval()
    model.load_state_dict(checkpoint['model_state_dict'])

    original, targets = make_data(simulation)
    # Confirm instrumentation leaves the original training examples untouched.
    plain = generator.generate_dataset(**simulation, n_channels=4,
        b_min=0.05, b_max=0.30, seed=77123)
    for a, b in zip(original, plain):
        np.testing.assert_array_equal(a, b)

    def evaluate(arrays):
        y, g, t = [torch.as_tensor(a, device=args.device) for a in arrays]
        reference = model.reconstruction.reference(y)
        truth = reference + t
        with torch.no_grad():
            normal = nmse_per_sample(model(y, g), truth)
            baseline = nmse_per_sample(reference, truth)
            shuffled = torch.stack([
                nmse_per_sample(model(y, torch.roll(g, s, 0)), truth)
                for s in (1, 2, 3)]).mean(0)
        return {k: v.cpu().tolist() for k, v in
                dict(normal=normal, mean_reference=baseline, shuffled_geometry=shuffled).items()}

    results = {'original_training_examples': [evaluate(original)]}
    for mode in ('fresh_noise_only', 'fresh_deformations_and_noise'):
        results[mode] = []
        for repeat in range(args.repeats):
            arrays, new_targets = make_data(simulation, repeat, mode == 'fresh_noise_only')
            for a, b in zip(targets, new_targets):
                np.testing.assert_array_equal(a, b)
            if mode == 'fresh_noise_only':
                np.testing.assert_array_equal(arrays[1], original[1])
            elif np.array_equal(arrays[1], original[1]):
                raise AssertionError('Deformation geometry did not change.')
            results[mode].append(evaluate(arrays))
        print(f'Completed {mode}: {args.repeats} sets of four environments.', flush=True)

    summary = {mode: {metric + '_nmse_db': nmse_db(float(np.mean([
        row[metric] for row in rows]))) for metric in rows[0]}
        for mode, rows in results.items()}
    print(json.dumps(summary, indent=2), flush=True)
    output = args.checkpoint.parent / 'fresh_deformation_results.json'
    output.write_text(json.dumps(dict(summary=summary, per_environment_linear_nmse=results,
        training_data_seed=77123, fresh_seed=88201, repeats=args.repeats,
        unchanged_physical_targets_verified=True, config=str(args.config),
        checkpoint=str(args.checkpoint)), indent=2))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
