"""Paired training on fresh deformation states of four fixed channel environments."""
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


def nominal_only(module, inputs, output):
    """Forward hook: retain G0 and zero both displacement descriptors."""
    x, g = output
    return x, torch.cat((g[..., :6], torch.zeros_like(g[..., 6:])), dim=-1)


def main():
    import copy
    import hashlib
    import random
    from datetime import datetime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Optional prior experiment_config.json')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--training-states', choices=['fresh', 'single'], default='fresh')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000000 or args.repeats < 2 or args.seed < 0:
        parser.error('Require 1..1000000 steps, repeats >=2, and a nonnegative seed.')
    wl = 3e8 / 28e9
    simulation = dict(n_h_b=5,n_v_b=5,d_x_b=wl/8,d_y_b=wl/8,
        n_h_u=5,n_v_u=5,d_x_u=wl/8,d_y_u=wl/8,fc=28e9,fs=1e5,
        n_subcarriers=2,n_deformations=8,n_paths=3,n_pilots=1,
        snr_values=[20.0],residual_reference='mean')
    model_config = {}
    if args.config:
        previous = json.loads(args.config.read_text())
        simulation = dict(previous['simulation'], n_subcarriers=2)
        model_config = previous['model']
    if simulation['residual_reference'] != 'mean':
        raise ValueError('This experiment requires mean-reference targets.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = args.output or Path('artifacts') / datetime.now().strftime('diversity_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    full = ChannelEstimator(**model_config).to(args.device)
    nominal = copy.deepcopy(full)
    nominal.preprocessing.register_forward_hook(nominal_only)
    models = dict(full=full, nominal=nominal)
    optimizers = {name: torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.)
                  for name, m in models.items()}
    original, physical_targets = make_data(simulation)
    plain = generator.generate_dataset(**simulation,n_channels=4,b_min=.05,b_max=.30,seed=77123)
    for a,b in zip(original,plain):
        np.testing.assert_array_equal(a,b)

    def tensors(arrays):
        return [torch.as_tensor(a,device=args.device) for a in arrays]

    y,g,t = tensors(original)
    # Reject a nominal-only source project inadvertently used as the full model.
    with torch.no_grad():
        _,full_g = full.preprocessing(y,g,full.nominal_geometry)
        _,nominal_g = nominal.preprocessing(y,g,nominal.nominal_geometry)
        if not torch.count_nonzero(full_g[...,6:]).item():
            raise ValueError('Full project already zeros displacement descriptors.')
        assert torch.count_nonzero(nominal_g[...,6:]).item() == 0
        torch.testing.assert_close(full_g[...,:6],nominal_g[...,:6])
    for key,value in full.state_dict().items():
        torch.testing.assert_close(value,nominal.state_dict()[key],rtol=0,atol=0)
    manifest = dict(simulation=simulation, model=full.config, steps=args.steps,
        repeats=args.repeats, initialization_seed=args.seed, environment_seed=77123,
        fresh_seed=88201, training_repeat_ids=[0,args.steps-1],
        test_repeat_ids=[1000000,1000000+args.repeats-1],training_states=args.training_states,
        learning_rate=3e-4,weight_decay=0.,batch_size=4,
        checkpoint_selection='final fixed update; no held-out selection',
        geometry_modes={'full':'all 18 descriptors','nominal':'G0 only; forward hook zeros last 12'},
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                       for folder in ('model','dataset','channel','geometry')
                       for p in sorted(Path(folder).glob('*.py'))})
    (output/'experiment_config.json').write_text(json.dumps(manifest,indent=2))
    print(f'Output: {output.resolve()}',flush=True)
    print('Four fixed environments; same initialization and batches for both models.',flush=True)
    print(f'{args.steps} updates per model; training states: {args.training_states}.',flush=True)
    history = []
    seen_geometry = set()

    def checked_data(repeat):
        arrays, targets = make_data(simulation,repeat)
        for a,b in zip(physical_targets,targets):
            np.testing.assert_array_equal(a,b)
        return arrays

    for step in range(1,args.steps+1):
        arrays = checked_data(step-1) if args.training_states=='fresh' else original
        seen_geometry.add(hashlib.sha256(arrays[1].tobytes()).hexdigest())
        y,g,t = tensors(arrays)
        row = dict(step=step)
        for name,model in models.items():
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            truth = model.reconstruction.reference(y)+t
            loss = nmse_per_sample(model(y,g),truth).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            row[name+'_preupdate_batch_nmse_db'] = nmse_db(loss.item())
        history.append(row)
        if step==1 or step%50==0 or step==args.steps:
            print(json.dumps(row),flush=True)
            (output/'history.json').write_text(json.dumps(history,indent=2))

    for name,model in models.items():
        model.eval()
        torch.save(dict(model_config=model.config,model_state_dict=model.state_dict(),
            geometry_mode=name,steps=args.steps),output/(name+'_final.pt'))

    def evaluate(arrays):
        y,g,t = tensors(arrays)
        truth = full.reconstruction.reference(y)+t
        result = {}
        with torch.no_grad():
            result['mean_reference'] = nmse_per_sample(full.reconstruction.reference(y),truth).cpu().tolist()
            for name,model in models.items():
                result[name] = nmse_per_sample(model(y,g),truth).cpu().tolist()
                # Replace geometry by another deformation set of the SAME
                # environment: avoids mixing amplitudes and channel identities.
                # This intervention is evaluated separately below.
        return result

    originals = evaluate(original)
    test_arrays = [checked_data(1000000+i) for i in range(args.repeats)]
    for arrays in test_arrays:
        assert hashlib.sha256(arrays[1].tobytes()).hexdigest() not in seen_geometry
    rows = []
    for i,arrays in enumerate(test_arrays):
        row = evaluate(arrays)
        y,_,t = tensors(arrays)
        wrong_g = torch.as_tensor(test_arrays[(i+1)%args.repeats][1],device=args.device)
        with torch.no_grad():
            truth = full.reconstruction.reference(y)+t
            for name,model in models.items():
                row[name+'_mismatched_geometry'] = nmse_per_sample(model(y,wrong_g),truth).cpu().tolist()
        rows.append(row)
    summary = {key+'_nmse_db':nmse_db(float(np.mean([row[key] for row in rows]))) for key in rows[0]}
    summary['full_advantage_over_nominal_db'] = summary['nominal_nmse_db']-summary['full_nmse_db']
    summary['full_gain_over_mean_db'] = summary['mean_reference_nmse_db']-summary['full_nmse_db']
    summary['nominal_gain_over_mean_db'] = summary['mean_reference_nmse_db']-summary['nominal_nmse_db']
    results = dict(held_out_summary=summary,held_out_per_environment_linear_nmse=rows,
        original_examples_nmse_db={k:nmse_db(float(np.mean(v))) for k,v in originals.items()},
        physical_targets_unchanged=True,paired_initialization_verified=True,
        unique_training_geometry_sets=len(seen_geometry),test_training_overlap=False)
    (output/'results.json').write_text(json.dumps(results,indent=2))
    print('\nHELD-OUT DEFORMATION RESULTS',flush=True)
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Saved {output.resolve()}',flush=True)
    print('Only four underlying environments: this does not test new-channel generalization.',flush=True)


if __name__ == '__main__':
    main()

