"""UAV-ON主方法的资源适配；导航、提示与评分复用本地主方法。"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request

from .core import clone, digest, files, read, require, save, verify_output

METHOD = dict(decisions=150, history_turns=4, text_memory='on', initial_yaw90=False,
              ascent_guard=True, depth_safety=True, max_step_len_m=6, stop_distance_m=10,
              success_distance_m=20, spawn_up_m=2, temperature=0, thinking=False,
              collision='airsim_dedicated_collision_latched_v1', spawn_collision='terminal_collision_v1')


def method_config(config):
    """开局旋转可显式开关；其余主方法参数仍固定，旧完整配置也可读取。"""
    supplied = config.get('method', {})
    if not isinstance(supplied, dict) or set(supplied)-set(METHOD):
        raise ValueError('method须为对象，且只能包含已登记的主方法参数')
    effective = dict(METHOD, **supplied)
    if type(effective['initial_yaw90']) is not bool:
        raise ValueError('method.initial_yaw90须为JSON布尔值true或false')
    if any(effective[key] != value for key, value in METHOD.items() if key != 'initial_yaw90'):
        raise ValueError('除开局旋转开关外，此入口不接受主方法参数或评分改写')
    return effective


def validate_config(config):
    cards=config.get('gpus', [])
    if (not cards or len(cards)>4 or len(set(cards))!=len(cards)
            or any(type(g) is not int or g<0 for g in cards)):
        raise ValueError('UAV-ON需要显式填写1至4张已授权GPU')
    count=config.get('concurrency')
    if type(count) is not int or count<1 or count>16 or count%len(cards):
        raise ValueError('concurrency须为1至16，且能按GPU均分；容量须另行GPU验收')
    models=config.get('models', [])
    if len(models)!=1 or models[0].get('name') not in ('qwen36','qwen3vl8b','qwen3vl4b','qwen35_9b','qwen35_4b'):
        raise ValueError('一次UAV-ON配置只运行一个受支持的Qwen模型')
    method_config(config)
    if config.get('record_mode','metrics-only') not in ('metrics-only','full'):
        raise ValueError('仅支持full或保留完整SR/OSR/SPL/NE的metrics-only')
    ports=config.get('ports', {})
    model,scene=ports.get('model'),ports.get('scene')
    if any(type(p) is not int or not 1024<=p<=65500 for p in (model,scene)):
        raise ValueError('ports需要有效的model/scene起始端口')
    used=[*range(model,model+len(cards)),*range(scene,scene+count)]
    if max(used)>65535 or len(used)!=len(set(used)):
        raise ValueError('模型和仿真端口范围重叠或越界')
    serving=config.get('serving', {})
    if set(serving)-{'gpu_memory_utilization','max_num_seqs'}:
        raise ValueError('只允许设置显存利用率和max_num_seqs；MTP关闭、Prefix Cache开启')
    util=serving.get('gpu_memory_utilization',.70);seqs=serving.get('max_num_seqs',12)
    if type(util) not in (float,int) or not 0<util<1 or type(seqs) is not int or seqs<1:
        raise ValueError('vLLM部署参数非法')


def build_inputs(config, repo):
    # 所有基准共用src，只有此处读取UAV-ON任务。
    import sys
    sys.path.insert(0,str(repo/'src'))
    from lightning_vln.benchmark import build_full_testset, TEST_SCENES
    tasks,manifest=build_full_testset(Path(config['paths']['annotations']))
    if manifest['source_total']!=1000 or len(tasks)!=733:
        raise ValueError('官方1.0 valset应为1000条，排除4个问题场景后733条；拒绝替换任务集')
    for scene in manifest['selected_by_scene']:
        require(Path(config['paths']['scenes'])/TEST_SCENES[scene],'UAV-ON test场景脚本')
    return tasks,manifest


def check(config, repo):
    from .core import verify_bundle
    verify_bundle(repo/'src')
    validate_config(config)
    for key in ('python','model_python'):
        require(config['paths'][key],key)
    require(Path(config['models'][0]['path'])/'config.json','模型配置')
    if shutil.which('bwrap') is None:
        raise ValueError('需要bubblewrap以隔离UE设备；不自动安装系统软件')
    require(repo/'third_party/UAV-ON/utils/classify_metric.py','官方评分参考')
    tasks,manifest=build_inputs(config,repo)
    from .selection import select
    chosen,proof=select(tasks,config,identity='episode_id')
    inputs={str(Path(config['paths']['annotations'])/name):sha
            for name,sha in manifest['source_files_sha256'].items()}
    code=('import json,importlib.util as u; '
          'print(json.dumps({n:bool(u.find_spec(n)) for n in MODULES}))')
    for role,modules in (('python',['numpy','PIL','msgpack']),('model_python',['torch','vllm'])):
        result=subprocess.run([config['paths'][role],'-B','-c',code.replace('MODULES',repr(modules))],
            capture_output=True,text=True,timeout=30,check=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
        missing=[name for name,found in json.loads(result.stdout).items() if not found]
        if missing:raise ValueError(role+'缺少模块：'+','.join(missing))
    return dict(passed=True,benchmark='uavon',expected=len(chosen),excluded=len(manifest['excluded_ids']),
                method=method_config(config),
                inputs=inputs,gpu_checked=False,gpu_started=False)


def batch_command(root, config, attempt, resume_from=None):
    source=root/'source_snapshot'
    method=method_config(config)
    cmd=[config['paths']['python'],'-B','-u',str(source/'scripts/run_uavon_ablation_subset.py'),
         '--dataset',str(root/'data/tasks.json'),
         '--test-sample-manifest' if config.get('selection') else '--benchmark-manifest',str(root/'data/manifest.json'),
         '--scene-root',config['paths']['scenes'],'--output',str(attempt/'full'),
         '--gpus',','.join(map(str,config['gpus'])),'--workers',str(config['concurrency']//len(config['gpus'])),
         '--port',str(config['ports']['scene']),'--isolate-gpu','--scene-batch-size','10',
         '--gpu-model-endpoints',str(root/'endpoints.json'),
         '--scene-restarts','3','--restart-backoff','10','--isolated-model-recovery']
    if resume_from:cmd+=['--resume-from',str(resume_from)]
    endpoint=read(root/'endpoints.json')[str(config['gpus'][0])]
    return cmd+['--','--base-url',endpoint['base_url'],'--model',endpoint['model'],
        '--model-hardware','配置指定的单卡TP1服务','--sim-hardware','配置指定GPU的隔离UE',
        '--current-target-evidence','--motion-options',
        *(['--initial-yaw90'] if method['initial_yaw90'] else []),'--ascent-guard',
        '--history-turns','4','--text-memory-mode','on','--decisions','150','--max-step-len','6',
        '--stop-distance','10','--spawn-up','2','--success-distance','20','--max-tokens','1024',
        '--record-mode',config.get('record_mode','metrics-only')]


def prepare(root, config, repo):
    verification=check(config,repo)
    config=dict(config, method=method_config(config))
    root.mkdir(parents=True,exist_ok=False)
    try:
        clone(repo/'src',root/'source_snapshot/src')
        (root/'source_snapshot/scripts').mkdir()
        for name in ('run_uavon_tuning_subset.py','run_uavon_ablation_subset.py'):
            shutil.copyfile(repo/'scripts'/name,root/'source_snapshot/scripts'/name)
        clone(repo/'third_party/UAV-ON',root/'source_snapshot/third_party/UAV-ON')
        tasks,manifest=build_inputs(config,repo)
        if config.get('selection'):
            from lightning_vln.adapters.uavon.minimal.sampling import sample_testset
            from .selection import select
            _,proof=select(tasks,config,identity='episode_id')
            save(root/'data/full-tasks.json',tasks);save(root/'data/full-manifest.json',manifest)
            spec=config['selection']
            tasks,manifest=sample_testset(root/'data/full-tasks.json',root/'data/full-manifest.json',
                min(spec['count'],len(tasks)),spec['seed'])
            save(root/'selection.json',proof)
        save(root/'data/tasks.json',tasks);save(root/'data/manifest.json',manifest)
        save(root/'machine-config.json',config);save(root/'method.json',config['method'])
        save(root/'endpoints.json',{str(g):dict(base_url=f"http://127.0.0.1:{config['ports']['model']+i}/v1",
             model=config['models'][0]['name']) for i,g in enumerate(config['gpus'])})
        clone(repo/'scripts/portable_experiments',root/'launcher/portable_experiments')
        shutil.copyfile(repo/'scripts/experiment.py',root/'launcher/experiment.py')
        save(root/'preparation-verification.json',verification)
        save(root/'portable-contract.json',dict(schema=1,root=str(root),repository=str(repo.resolve()),
            benchmark='uavon',models=[],files=files(root),inputs=verification['inputs'],
            validation=dict(cpu_prepared=True,gpu_smoke='pending',concurrency='not_revalidated')))
    except BaseException as error:
        save(root/'preparation-failure.json',dict(error=str(error)));raise
    return root


def model_command(config, index):
    cmd=[config['paths']['model_python'],'-B','-m','vllm.entrypoints.openai.api_server',
         '--model',config['models'][0]['path'],'--served-model-name',config['models'][0]['name'],
         '--host','127.0.0.1','--port',str(config['ports']['model']+index),
         '--gpu-memory-utilization',str(config.get('serving',{}).get('gpu_memory_utilization',.70)),
         '--max-num-seqs',str(config.get('serving',{}).get('max_num_seqs',12)),
         '--enable-prefix-caching','--trust-remote-code']
    if config['models'][0]['name'] in ('qwen36','qwen35_9b','qwen35_4b'):
        cmd+=['--reasoning-parser','qwen3']
    return cmd


def runtime_directory(config):
    runtime=Path(tempfile.mkdtemp(prefix='lv-',dir=config['paths'].get('runtime_root') or None))
    if len(str(runtime).encode())+37>107:
        runtime.rmdir()
        raise ValueError('runtime_root过长，无法容纳vLLM Unix IPC路径；请选择短临时目录')
    return runtime


def check_resources(config):
    """只在首次run时验空闲，不因运行中外来进程中断本轮。"""
    snapshots=[]
    for card in config['gpus']:
        row=subprocess.check_output(['nvidia-smi','-i',str(card),
            '--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        uuid,used=row[0].strip(),int(row[1].strip())
        active=subprocess.check_output(['nvidia-smi','-i',uuid,
            '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if active or used>256:raise ValueError(f'GPU{card}已被占用；不接管或清理：{active or used}')
        snapshots.append(dict(gpu=card,uuid=uuid,used_mib=used))
    ports=[*range(config['ports']['model'],config['ports']['model']+len(config['gpus'])),
           *range(config['ports']['scene'],config['ports']['scene']+config['concurrency'])]
    for port in ports:
        with socket.socket() as probe:probe.bind(('127.0.0.1',port))
    return snapshots


class OwnedChild:
    """登记自己启动的进程树；释放仅限UID与启动时间吻合的成员。"""
    def __init__(self,command,env,log,record):
        from .runtime import identity
        self.log=log.open('x');self.members={}
        self.process=subprocess.Popen(command,env=env,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        self.record=record;self.leader=identity(self.process.pid)
        if self.leader is None:raise RuntimeError('子进程启动后无法核实身份，见'+str(log))
        self.members[self.process.pid]=self.leader
        save(record,dict(self.leader,command=command));self.refresh()

    def refresh(self):
        from .runtime import identity,alive
        rows={}
        for p in Path('/proc').iterdir():
            if not p.name.isdecimal():continue
            try:
                fields=(p/'stat').read_bytes().rsplit(b')',1)[1].split()
                rows[int(p.name)]=(int(fields[1]),identity(int(p.name)))
            except (OSError,IndexError,ValueError):continue
        active={pid for pid,row in self.members.items() if alive(row)}
        while True:
            new={pid for pid,(parent,row) in rows.items() if parent in active and row and row['uid']==os.getuid()}
            if new<=active:break
            active.update(new)
        self.members.update({pid:rows[pid][1] for pid in active if pid in rows})
        save(self.record.with_name(self.record.stem+'-members.json'),list(self.members.values()))

    def close(self, first=signal.SIGTERM):
        from .runtime import alive
        self.refresh()
        for signum,seconds in ((first,180 if first==signal.SIGINT else 20),(signal.SIGTERM,20),(signal.SIGKILL,10)):
            targets=({self.process.pid:self.leader} if signum==signal.SIGINT else self.members)
            for pid,row in list(targets.items()):
                if alive(row):
                    try:os.kill(pid,signum)
                    except ProcessLookupError:pass
            deadline=time.monotonic()+seconds
            while time.monotonic()<deadline:
                self.process.poll()
                if not any(alive(row) for row in self.members.values()):break
                time.sleep(.2)
            if not any(alive(row) for row in self.members.values()):break
        self.log.close()
        remaining=[row for row in self.members.values() if alive(row)]
        if remaining:raise RuntimeError('登记子进程释放未完成：'+str(remaining))


def run(root, config, resume=False):
    from .runtime import identity, alive
    verify_output(root)
    expected=read(root/'preparation-verification.json')['expected']
    with (root/'portable.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=root/'portable-process.json'
        if previous.exists():
            if alive(read(previous)):raise ValueError('当前评测进程仍存活')
            if not resume:raise ValueError('已经运行过；请显式resume，不覆盖旧attempt')
        for p in root.glob('attempt-*/**/*process*.json'):
            data=read(p)
            for row in data if isinstance(data,list) else [data]:
                if isinstance(row,dict) and {'pid','uid','start_ticks'}<=row.keys() and alive(row):
                    raise ValueError('本轮旧进程仍存活，拒绝重复启动：'+str(p))
        attempts=sorted(root.glob('attempt-[0-9][0-9][0-9][0-9]'))
        # Model-start failures may have no full directory; inherit the latest
        # actual batch, including its recursively registered earlier settlements.
        source=next((a/'full' for a in reversed(attempts) if (a/'full/manifest.json').exists()),None) if resume else None
        if source:
            env=dict(os.environ,PYTHONPATH=str(root/'source_snapshot/src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
            command=batch_command(root,config,root/'dry-run-placeholder',source)
            command.insert(command.index('--'),'--dry-run')
            proof=subprocess.run(command,env=env,capture_output=True,text=True,check=True,timeout=120)
            plan=json.loads(proof.stdout)
            if len(plan['reused_episodes'])==expected:
                # A standalone summary never establishes completion.
                if not (source/'summary.json').exists():raise ValueError('733条登记已结算，但来源缺少最终summary')
                summary=read(source/'summary.json')
                save(root/'portable-status.json',dict(phase='complete',result=str(source),summary=summary));return
        save(root/'resource-check.json',dict(cards=check_resources(config),time=time.time()))
        for marker in ('stop.request','cancel.request'):
            p=root/marker
            if p.exists():p.rename(root/(marker+'.previous-'+str(time.time_ns())))
        attempt=root/f'attempt-{len(attempts)+1:04d}';attempt.mkdir()
        (attempt/'logs').mkdir()
        save(previous,identity(os.getpid()))
        def interrupted(*_):raise KeyboardInterrupt('用户停止')
        old={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
        children=[];batch=None;phase='failed';error=None;cleanup=[]
        try:
            # vLLM uses Unix IPC sockets: deep result paths exceed sun_path.
            runtime=runtime_directory(config)
            save(attempt/'runtime-path.json',dict(path=str(runtime),uid=os.getuid()))
            env=dict(os.environ,PYTHONPATH=str(root/'source_snapshot/src'),PYTHONDONTWRITEBYTECODE='1',
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TMPDIR=str(runtime),
                XDG_CACHE_HOME=str(attempt/'cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                NO_PROXY='127.0.0.1,localhost,::1',no_proxy='127.0.0.1,localhost,::1')
            if config['paths'].get('vulkan_icd'):env['VK_ICD_FILENAMES']=config['paths']['vulkan_icd']
            for i,card in enumerate(config['gpus']):
                cache=attempt/'cache'/f'gpu-{card}'
                model_env=dict(env,CUDA_VISIBLE_DEVICES=str(card),HF_HOME=str(cache/'huggingface'),
                    VLLM_CACHE_ROOT=str(cache/'vllm'),TORCHINDUCTOR_CACHE_DIR=str(cache/'torchinductor'),
                    TRITON_CACHE_DIR=str(cache/'triton'),
                    PATH=str(Path(config['paths']['model_python']).parent)+os.pathsep+env.get('PATH',''))
                children.append(OwnedChild(model_command(config,i),model_env,
                    attempt/f'logs/model-{card}.log',attempt/f'model-{card}-process.json'))
            endpoints=list(read(root/'endpoints.json').values());deadline=time.monotonic()+1800
            while True:
                if (root/'stop.request').exists():raise KeyboardInterrupt('用户停止')
                if any(c.process.poll() is not None for c in children):raise RuntimeError('模型启动退出，见本attempt模型日志')
                ready=True
                for endpoint in endpoints:
                    try:
                        with urllib.request.urlopen(endpoint['base_url']+'/models',timeout=3) as response:
                            ready &= endpoint['model'] in {x['id'] for x in json.load(response)['data']}
                    except OSError:ready=False
                if ready:break
                if time.monotonic()>deadline:raise TimeoutError('模型30分钟未就绪')
                for child in children:child.refresh()
                time.sleep(3)
            command=batch_command(root,config,attempt,source);save(attempt/'command.json',command)
            batch=OwnedChild(command,env,attempt/'logs/evaluation.log',attempt/'batch-process.json')
            save(root/'portable-status.json',dict(phase='running',attempt=str(attempt),pid=batch.process.pid))
            while batch.process.poll() is None:
                if (root/'stop.request').exists():raise KeyboardInterrupt('用户停止')
                if any(c.process.poll() is not None for c in children):raise RuntimeError('评测期间模型退出')
                for child in [*children,batch]:child.refresh()
                time.sleep(3)
            if batch.process.returncode:raise RuntimeError('评测不完整，保留attempt；见日志和full/incomplete.json')
            summary=read(attempt/'full/summary.json')
            if summary.get('total_finished')!=expected or any(summary.get(k) is None for k in ('sr','osr','spl','ne_m')):
                raise ValueError(f'结果缺少{expected}条唯一结算或完整SR/OSR/SPL/NE')
            phase='complete';save(root/'summary.json',summary)
        except BaseException as caught:
            phase='interrupted' if isinstance(caught,KeyboardInterrupt) else 'failed'
            error=str(caught)
            raise
        finally:
            for sig in old:signal.signal(sig,signal.SIG_IGN)
            for child in ([batch] if batch else [])+list(reversed(children)):
                try:child.close(signal.SIGINT if child is batch else signal.SIGTERM)
                except Exception as caught:cleanup.append(str(caught))
            save(root/'portable-status.json',dict(phase=phase if not cleanup else 'cleanup_failed',
                 attempt=str(attempt),result=str(attempt/'full'),error=error,cleanup_errors=cleanup,released=not cleanup))
            for sig,handler in old.items():signal.signal(sig,handler)
            if cleanup:raise RuntimeError('自有资源释放未完成：'+';'.join(cleanup))
