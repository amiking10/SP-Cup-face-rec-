"""Reproducible video benchmark. Legacy annotated-target CSV scoring: c3_tracker.evaluation."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import re
import sys
from .benchmark.inference import BenchmarkConfig


def positive(value):
    number=float(value)
    if not math.isfinite(number) or number<=0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def resolution(value):
    if value=='original':
        return None
    match=re.fullmatch(r'([1-9]\d*)x([1-9]\d*)',value)
    if not match:
        raise argparse.ArgumentTypeError('use original or WIDTHxHEIGHT, such as 640x480')
    return tuple(map(int,match.groups()))


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0]=='prepare':
            parser=argparse.ArgumentParser(description='Normalize verified AVA annotations using an explicit video manifest')
            parser.add_argument('--manifest',type=Path,required=True)
            parser.add_argument('--output',type=Path,required=True)
            parser.add_argument('--alignment-tolerance',type=positive,default=.0051)
            args=parser.parse_args(argv[1:])
            from .benchmark.ava import prepare_ava
            result=prepare_ava(args.manifest,args.output,args.alignment_tolerance)
            print(json.dumps({'prepared_videos':len(result['videos']),'manifest':str(args.output/'manifest.json')}))
            return 0
        if argv and argv[0]=='prepare-video':
            parser=argparse.ArgumentParser(description='Explicitly resample a CFR source clip, recording all frame mappings')
            parser.add_argument('--input',type=Path,required=True)
            parser.add_argument('--output',type=Path,required=True)
            parser.add_argument('--start-seconds',type=float,required=True)
            parser.add_argument('--duration',type=positive,required=True)
            parser.add_argument('--fps',type=positive,required=True)
            args=parser.parse_args(argv[1:])
            from .benchmark.video import prepare_video
            result=prepare_video(args.input,args.output,args.start_seconds,args.duration,args.fps)
            print(json.dumps({key:value for key,value in result.items() if key!='frame_mapping'},indent=2))
            return 0
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--dataset',choices=('ava','common','chokepoint','vpcd','mobiface'),required=True)
        parser.add_argument('--data-root',type=Path,default=Path('datasets/data'))
        parser.add_argument('--manifest',type=Path)
        parser.add_argument('--video-id',action='append')
        parser.add_argument('--output',type=Path,required=True)
        parser.add_argument('--experiment',choices=('end_to_end','oracle'),default='end_to_end')
        parser.add_argument('--resolution',type=resolution,default=None)
        parser.add_argument('--fps',type=positive)
        parser.add_argument('--matrix',action='store_true',help='original/640x480/320x240 at native and 10 FPS')
        parser.add_argument('--annotate-video',action='store_true')
        parser.add_argument('--hota-python',type=Path,help='separate TrackEval interpreter; see guide')
        parser.add_argument('--appearance-model',choices=('sface','mobilenet'),default='sface')
        parser.add_argument('--model-dir',default='.models')
        parser.add_argument('--confidence',type=positive,default=.65)
        parser.add_argument('--iou-threshold',type=positive,default=.5)
        parser.add_argument('--model-selection',type=int,choices=(0,1),default=0)
        parser.add_argument('--n-init',type=int,default=3)
        parser.add_argument('--track-memory-seconds',type=positive,default=3)
        parser.add_argument('--max-cosine-distance',type=positive)
        parser.add_argument('--max-frames',type=int)
        parser.add_argument('--no-target',action='store_true')
        parser.add_argument('--seed',type=int,default=0)
        parser.add_argument('--threads',type=int,default=1)
        parser.add_argument('--backend-factory',default='existing',help='existing or package.module:factory; factory(config) implements TrackerBackend')
        parser.add_argument('--gpu',action='store_true')
        args=parser.parse_args(argv)
        if args.dataset=='chokepoint':
            raise ValueError('ChokePoint provides eye points, not native face boxes. Use python -m c3_tracker.benchmark.chokepoint '
                             'for verified eye-point diagnostics. Box/MOT scoring requires independently validated face boxes.')
        if args.dataset not in ('ava','common'):
            raise ValueError(f'{args.dataset} native format/footage has not been validated. No loader is claimed. See datasets/BENCHMARK_GUIDE.md for required common-format mapping.')
        if args.confidence>1 or args.iou_threshold>1 or (args.max_cosine_distance and args.max_cosine_distance>1):
            raise ValueError('Confidence and IoU/cosine limits must be <= 1')
        if args.n_init<1 or args.threads<1 or args.max_frames is not None and args.max_frames<1:
            raise ValueError('n-init, threads and max-frames must be positive')
        if not 0<=args.seed<2**31:
            raise ValueError('seed must be in [0, 2**31) for NumPy and OpenCV')
        if args.matrix and (args.resolution is not None or args.fps is not None):
            raise ValueError('--matrix defines its own resolution/FPS conditions; do not combine it with --resolution/--fps')
        if args.gpu and args.appearance_model=='sface':
            raise ValueError('SFace baseline uses CPU; --gpu applies to the MobileNet backend only')
        default_folder='ava_active_speaker' if args.dataset=='ava' else 'common'
        manifest=args.manifest or args.data_root/'processed'/default_folder/'manifest.json'
        config=BenchmarkConfig(experiment=args.experiment,resolution=args.resolution,fps=args.fps,
                               confidence=args.confidence,model_selection=args.model_selection,n_init=args.n_init,
                               memory_seconds=args.track_memory_seconds,max_cosine_distance=args.max_cosine_distance,
                               appearance_model=args.appearance_model,model_dir=args.model_dir,gpu=args.gpu,
                               iou_threshold=args.iou_threshold,target_enabled=not args.no_target,
                               max_frames=args.max_frames,backend_factory=args.backend_factory,seed=args.seed,threads=args.threads)
        from .benchmark.runner import run_benchmark,environment
        from .benchmark.schema import write_json,fingerprint
        if args.matrix:
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError('Matrix output must be a fresh directory')
            args.output.mkdir(parents=True,exist_ok=True)
            matrix={'status':'running','manifest_sha256':fingerprint(manifest),
                    'python_source_sha256':environment()['python_source_sha256'],'variants':{}}
            write_json(args.output/'matrix_manifest.json',matrix)
            try:
                for size in (None,(640,480),(320,240)):
                    for fps in (None,10):
                        if environment()['python_source_sha256']!=matrix['python_source_sha256']:
                            raise RuntimeError('Source changed between matrix variants; do not compare mixed code revisions')
                        label=('original' if size is None else f'{size[0]}x{size[1]}')+('_nativefps' if fps is None else '_10fps')
                        result=run_benchmark(manifest,args.dataset,args.output/label,replace(config,resolution=size,fps=fps),
                                             args.video_id,args.annotate_video,args.hota_python)
                        matrix['variants'][label]={'status':'complete','output':result['output']}
                        write_json(args.output/'matrix_manifest.json',matrix)
                        print(json.dumps(result,indent=2))
                matrix['status']='complete'
                write_json(args.output/'matrix_manifest.json',matrix)
            except BaseException as exc:
                matrix.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                write_json(args.output/'matrix_manifest.json',matrix)
                raise
        else:
            print(json.dumps(run_benchmark(manifest,args.dataset,args.output,config,args.video_id,
                                            args.annotate_video,args.hota_python),indent=2))
        return 0
    except (ValueError,RuntimeError,OSError,ImportError,KeyError,TypeError) as exc:
        print(f'Benchmark error: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
