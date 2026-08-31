from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

import torch
import triton

EXPECTED_HEAD = '81dffbfeb0f84470513e846e3df8080e8ffb563d'
EXPECTED_RUNNER = 'a212da2bf7631061659a59046a83f98ccd47ff3a8311fce03b1b1ba38f273c92'
EXPECTED_FLA_ADAPTER = '96e98ca3f488a36832aa767d5c3b12a5ae3544d8fc12d042c734037b62a25f75'
EXPECTED_MODEL = 'a921d49ed4e4c2e12113d87c2cda9743e7a297bd26d4c31e77cab71dc254c21d'
EXPECTED_FLA = '5e02dd3a7651f5f2797eb8b12bbec401826031e1'
EXPECTED_KERNELS = {
    'src/attnres/_kernels/fixed_tail.py': '2333b3034e3c0e6493855b1246280ed91e65d29a962ce1d150beff71e8bbd34e',
    'src/attnres/_kernels/fla_full_sources.py': '2cd7ac89b15faeb13640bff4a7948e437453b69446bfc8c7922511e341843e10',
    'src/attnres/_kernels/fixed_tail_sources.py': '20fa0206fcbf6cc6b28a2973ac280575b6e8e378b09e0903449bf423d9812196',
}
SEEDS = {20260827, 20260903, 20260911}

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(root: Path, *args: str) -> str:
    return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()

def smi() -> dict[str,str]:
    query='name,uuid,driver_version,pstate,pci.bus_id,power.limit,clocks.max.sm,memory.total'
    raw=subprocess.run(['nvidia-smi',f'--query-gpu={query}','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.strip().splitlines()
    assert len(raw)==1
    values=[x.strip() for x in raw[0].split(',')]
    keys=query.split(',')
    return dict(zip(keys,values,strict=True))

p=argparse.ArgumentParser()
p.add_argument('--repo',required=True)
p.add_argument('--config',required=True)
p.add_argument('--out',required=True)
p.add_argument('--gpu',choices=['H100','B200'],required=True)
a=p.parse_args()
root=Path(a.repo).resolve(); config_path=Path(a.config).resolve(); out=Path(a.out).resolve()
assert git(root,'rev-parse','HEAD') == EXPECTED_HEAD
assert git(root,'status','--porcelain') == ''
assert sha(root/'benchmarks/run.py') == EXPECTED_RUNNER
assert sha(root/'benchmarks/fla_compile.py') == EXPECTED_FLA_ADAPTER
assert sha(root/'benchmarks/model.py') == EXPECTED_MODEL
for name,digest in EXPECTED_KERNELS.items(): assert sha(root/name)==digest
vendor=Path('/root/flash-attnres-vendors-v7/fla')
assert git(vendor,'rev-parse','HEAD') == EXPECTED_FLA
assert git(vendor,'status','--porcelain') == ''
assert torch.__version__ == '2.13.0+cu130'
assert torch.version.cuda == '13.0'
assert triton.__version__ == '3.7.1'
assert torch.cuda.is_available() and torch.cuda.device_count()==1
name=torch.cuda.get_device_name(0); cc=tuple(torch.cuda.get_device_capability(0)); hardware=smi()
assert (a.gpu=='H100' and 'H100' in name and cc==(9,0)) or (a.gpu=='B200' and 'B200' in name and cc==(10,0))
config=json.loads(config_path.read_text())
seed=config.get('seed'); assert type(seed) is int and seed in SEEDS
campaign=config.get('compiled_step_campaign',{})
assert campaign.get('schema')=='attnres.compiled_step_campaign.v2'
assert campaign.get('seed')==seed and campaign.get('dtype')=='bf16_autocast'
assert campaign.get('metric')=='captured_complete_training_step_device_time'
assert campaign.get('input_copy_inside_timing') is False
assert campaign.get('hashing_inside_timing') is False
assert campaign.get('qualification_inside_timing') is False
rms=campaign.get('fla_unit_rms_weight',{})
assert rms == {
  'lifecycle':'preallocated_nonpersistent_model_buffer',
  'allocated_before_compile_capture_timing':True,
  'fill_launches_inside_step':0,
  'direct_operator_fallback':'query_ones',
}
assert config.get('phases')==['model'] and config.get('mode')=='full'
assert config.get('model_timing')=='cuda_graph' and config.get('model_warmup')==10 and config.get('model_rounds')==120
assert config.get('accumulation')==1 and config.get('ranks')==[1024]
mc=config.get('model_config',{})
assert mc == {'batch':2,'block_count':8,'ffn':2816,'heads':16,'layers':24,'mode':'full','sequence':1024,'source_layout':'list','variant':'sliced','vocab':32768,'width':1024}

sys.path.insert(0,str(root)); sys.path.insert(0,str(root/'src')); os.chdir(root)
from benchmarks.run import assert_frozen_hashes, run_suite
frozen=assert_frozen_hashes(root); assert len(frozen)==62
started=time.time(); report=run_suite(config); finished=time.time()
model=report.get('model_timings',{})
meta=model.get('compile_backend_metadata',{}).get('fla_triton_compile',{})
assert meta.get('adapter_sha256')==EXPECTED_FLA_ADAPTER
assert meta.get('model_rms_weight_allocation')=='nonpersistent_buffer'
assert meta.get('model_rms_weight_reuse')=='one_buffer_per_model'
assert meta.get('compiled_model_fill_launches_per_step')==0
assert meta.get('direct_call_fallback')=='query_ones'
complete = model.get('status')=='complete' and model.get('failures')==[] and model.get('comparator_failures')==[]
report['compiled_step_execution_status']='complete' if complete else 'failed'
report['compiled_step_runtime_preflight']={
    'schema':'attnres.compiled_step_runtime_preflight.v2','status':'passed',
    'gpu_selector':a.gpu,'gpu_name':name,'compute_capability':list(cc),'nvidia_smi':hardware,
    'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'triton':triton.__version__,
    'repo_head':EXPECTED_HEAD,'repo_clean':True,'runner_sha256':EXPECTED_RUNNER,
    'fla_adapter_sha256':EXPECTED_FLA_ADAPTER,'model_sha256':EXPECTED_MODEL,
    'kernel_sha256':EXPECTED_KERNELS,'frozen_manifest_sha256':sha(root/'validation/frozen.json'),
    'fla_revision':EXPECTED_FLA,'fla_clean':True,'config_sha256':sha(config_path),
    'wrapper_sha256':sha(Path(__file__).resolve()),'started_unix_s':started,'finished_unix_s':finished,
    'timed_tensor_hashing':False,'timed_input_copy':False,'timed_qualification':False,
    'fla_unit_rms_weight_lifecycle':'preallocated_nonpersistent_model_buffer',
    'fla_fill_launches_inside_step':0,
}
out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_suffix(out.suffix+'.tmp')
tmp.write_text(json.dumps(report,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8'); os.replace(tmp,out)
print(json.dumps({'status':report.get('status'),'compiled_step_execution_status':report['compiled_step_execution_status'],'model_status':model.get('status'),'raw_rows':len(model.get('raw_samples',[])),'statistics':sorted(model.get('statistics',{}))},sort_keys=True),flush=True)
raise SystemExit(0 if complete else 1)
