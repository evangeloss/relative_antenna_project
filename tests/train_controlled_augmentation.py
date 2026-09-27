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


def make_data(simulation, repeat=None, noise_only=False, n_channels=200, environment_seed=77123):
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
        arrays = generator.generate_dataset(**simulation, n_channels=n_channels,
            b_min=0.05, b_max=0.30, seed=environment_seed)
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

    p = argparse.ArgumentParser(description='Fixed versus refreshed training data; independent held-out environments.')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--train-environments',type=int,default=200)
    p.add_argument('--validation-environments',type=int,default=50)
    p.add_argument('--test-environments',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--geometry-mode',choices=['nominal','full'],default='nominal')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if min(args.epochs,args.train_environments,args.validation_environments,args.test_environments,args.batch_size)<1 or args.seed<0:
        p.error('Counts must be positive and seed nonnegative.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    output=args.output or Path('artifacts')/datetime.now().strftime('controlled_augmentation_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    wl=3e8/28e9
    simulation=dict(n_h_b=5,n_v_b=5,d_x_b=wl/8,d_y_b=wl/8,
        n_h_u=5,n_v_u=5,d_x_u=wl/8,d_y_u=wl/8,fc=28e9,fs=1e5,
        n_subcarriers=2,n_deformations=8,n_paths=3,n_pilots=1,
        snr_values=[20.],residual_reference='mean')
    # Split seeds derive from disjoint SeedSequence children; fixed over epochs.
    split_seeds=[int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(4)]
    train_seed,val_seed,test_seed,extra_seed=split_seeds
    base=ChannelEstimator().to(args.device)
    if args.geometry_mode=='nominal':
        base.preprocessing.register_forward_hook(nominal_only)
    models={'fixed':base,'fresh':copy.deepcopy(base)}
    for k,v in models['fixed'].state_dict().items():
        torch.testing.assert_close(v,models['fresh'].state_dict()[k],rtol=0,atol=0)
    optimizers={name:torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4)
                for name,m in models.items()}
    print(f'Output: {output.resolve()}',flush=True)
    print('Generating fixed training and validation data...',flush=True)
    original,physical_targets=make_data(simulation,n_channels=args.train_environments,environment_seed=train_seed)
    plain=generator.generate_dataset(**simulation,n_channels=args.train_environments,b_min=.05,b_max=.30,seed=train_seed)
    for a,b in zip(original,plain):
        np.testing.assert_array_equal(a,b)
    del plain
    validation=generator.generate_dataset(**simulation,n_channels=args.validation_environments,b_min=.05,b_max=.30,seed=val_seed)
    model_config=base.config
    config=dict(arguments=vars(args)|{'output':str(output)},simulation=simulation,model_config=model_config,
        split_seeds=dict(train=train_seed,validation=val_seed,test=test_seed,extrapolation=extra_seed),
        learning_rate=3e-4,weight_decay=1e-4,gradient_clip=1.,
        updates_per_model=args.epochs*((args.train_environments+args.batch_size-1)//args.batch_size),
        comparison='fixed vs fresh deformation AND noise; epoch 1 identical',
        geometry_mode=args.geometry_mode,selection='best validation NMSE per model; test only after training',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest()
            for folder in ('model','dataset','channel','geometry') for f in sorted(Path(folder).glob('*.py'))})
    (output/'experiment_config.json').write_text(json.dumps(config,indent=2))
    print(f'{args.geometry_mode} geometry; {config["updates_per_model"]} updates per model.',flush=True)

    def tensors(arrays,indices):
        return [torch.as_tensor(a[indices],device=args.device) for a in arrays]

    def evaluate(model,arrays):
        model.eval()
        errors,baselines=[],[]
        with torch.no_grad():
            for start in range(0,len(arrays[0]),args.batch_size):
                y,g,t=tensors(arrays,slice(start,start+args.batch_size))
                ref=model.reconstruction.reference(y)
                truth=ref+t
                errors.extend(nmse_per_sample(model(y,g),truth).cpu().tolist())
                baselines.extend(nmse_per_sample(ref,truth).cpu().tolist())
        return dict(nmse=float(np.mean(errors)),nmse_db=nmse_db(float(np.mean(errors))),
            mean_reference_nmse_db=nmse_db(float(np.mean(baselines))),
            per_environment_linear_nmse=errors,reference_per_environment_linear_nmse=baselines)

    best={name:float('inf') for name in models}
    best_epoch={}
    histories=[]
    order_rng=np.random.default_rng(np.random.SeedSequence([args.seed,582]))
    for epoch in range(1,args.epochs+1):
        if epoch==1:
            fresh=original
        else:
            print(f'Generating fresh states for epoch {epoch}...',flush=True)
            fresh,new_targets=make_data(simulation,repeat=epoch,n_channels=args.train_environments,environment_seed=train_seed)
            for a,b in zip(physical_targets,new_targets):
                np.testing.assert_array_equal(a,b)
            assert not np.array_equal(fresh[1],original[1])
        order=order_rng.permutation(args.train_environments)
        record={'epoch':epoch}
        for name,arrays in (('fixed',original),('fresh',fresh)):
            model=models[name]
            model.train()
            losses=[]
            for start in range(0,len(order),args.batch_size):
                idx=order[start:start+args.batch_size]
                y,g,t=tensors(arrays,idx)
                truth=model.reconstruction.reference(y)+t
                optimizer=optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                values=nmse_per_sample(model(y,g),truth)
                loss=values.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                losses.extend(values.detach().cpu().tolist())
            result=evaluate(model,validation)
            record[name]={'training_nmse_db':nmse_db(float(np.mean(losses))),
                          'validation_nmse_db':result['nmse_db']}
            if result['nmse']<best[name]:
                best[name]=result['nmse']
                best_epoch[name]=epoch
                torch.save(dict(model_config=model_config,model_state_dict=model.state_dict(),
                    geometry_mode=args.geometry_mode,epoch=epoch),output/(name+'_best.pt'))
        histories.append(record)
        (output/'history.json').write_text(json.dumps(histories,indent=2))
        print(json.dumps(record),flush=True)
    # Test environments never participate in optimization or checkpoint selection.
    print('Training finished. Evaluating selected checkpoints on unseen environments...',flush=True)
    test=generator.generate_dataset(**simulation,n_channels=args.test_environments,b_min=.05,b_max=.30,seed=test_seed)
    extrapolation=generator.generate_dataset(**simulation,n_channels=args.test_environments,b_min=.35,b_max=.50,seed=extra_seed)
    results={}
    for name,model in models.items():
        checkpoint=torch.load(output/(name+'_best.pt'),map_location=args.device,weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        results[name]=dict(best_epoch=best_epoch[name],
            validation=evaluate(model,validation),test=evaluate(model,test),
            extrapolation=evaluate(model,extrapolation))
    summary={split:{
        'fixed_nmse_db':results['fixed'][split]['nmse_db'],
        'fresh_nmse_db':results['fresh'][split]['nmse_db'],
        'mean_reference_nmse_db':results['fixed'][split]['mean_reference_nmse_db'],
        'fresh_advantage_db':results['fixed'][split]['nmse_db']-results['fresh'][split]['nmse_db']}
        for split in ('validation','test','extrapolation')}
    (output/'results.json').write_text(json.dumps(dict(summary=summary,details=results),indent=2))
    print('\nCONTROLLED EXPERIMENT RESULTS',flush=True)
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Saved {output.resolve()}',flush=True)


if __name__=='__main__':
    main()

