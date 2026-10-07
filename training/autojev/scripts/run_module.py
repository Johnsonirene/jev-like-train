import sys, pathlib, time, resource, json, threading, subprocess, os, signal, runpy
root=pathlib.Path(__file__).resolve().parents[1]
module=sys.argv[1]
assert module in {'autojev.train','autojev.evaluate','autojev.data'}
output=pathlib.Path(sys.argv[2]).resolve()
assert output.is_relative_to(root)
sys.argv=[module]+sys.argv[3:]
gpu=module!='autojev.data'
mem={line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines()}
result={'module':module,'available_memory_start_bytes':mem['MemAvailable'],'rss_budget_bytes':0.8*mem['MemAvailable'],'gpu_budget_bytes':14e9,'started':time.time(),'argv':sys.argv}
if gpu:
    import torch
    props=torch.cuda.get_device_properties(0)
    free,total=torch.cuda.mem_get_info()
    cap=min(12e9,14e9-(total-free)-0.5e9)
    assert cap>1e9
    torch.cuda.set_per_process_memory_fraction(cap/props.total_memory)
    result['allocator_cap_bytes']=cap
stop=threading.Event()
peak_gpu=0
violations=[]
def monitor():
    global peak_gpu
    while not stop.is_set():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if rss>result['rss_budget_bytes']:
            violations.append({'rss_bytes':rss,'budget':result['rss_budget_bytes']})
        if gpu:
            values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
            used=int(values.strip().splitlines()[0])*1024*1024
            peak_gpu=max(peak_gpu,used)
            if used>14e9: violations.append({'gpu_total_bytes':used,'budget':14e9})
        if violations:
            output.parent.mkdir(parents=True,exist_ok=True)
            output.with_suffix('.violation.json').write_text(json.dumps(violations,indent=2))
            os.kill(os.getpid(),signal.SIGINT)
            return
        stop.wait(1)
thread=threading.Thread(target=monitor,daemon=True)
thread.start()
start=time.monotonic()
try:
    runpy.run_module(module,run_name='__main__')
    result['completed']=True
finally:
    stop.set()
    thread.join(timeout=5)
    result.update(seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,sampled_peak_gpu_total_bytes=peak_gpu,violations=violations)
    if gpu:
        result['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['peak_gpu_reserved_bytes']=torch.cuda.max_memory_reserved()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
