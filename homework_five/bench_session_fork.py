import json, os, socket, time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from allgather_worker import ring_allgather, recursive_doubling_allgather, swing_allgather, verify

def run_algo(algo, local, ws, rank):
    return {'ring': ring_allgather, 'recursive_doubling': recursive_doubling_allgather, 'swing': swing_allgather}[algo](local, ws, rank)

def free_port():
    s=socket.socket(); s.bind(('127.0.0.1',0)); p=s.getsockname()[1]; s.close(); return p

def worker(rank, ws, port, jobs_json, result_file):
    os.environ['MASTER_ADDR']='127.0.0.1'; os.environ['MASTER_PORT']=str(port)
    os.environ.setdefault('OMP_NUM_THREADS','1')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=ws)
    jobs = json.loads(jobs_json)
    results=[]
    for job in jobs:
        algo = job['algorithm']
        msg = job['msg_bytes_per_rank']
        local=torch.full((msg,), rank % 251, dtype=torch.uint8)
        gathered=run_algo(algo, local, ws, rank)
        verify(gathered,msg,ws)
        dist.barrier()
        t0=time.perf_counter(); gathered=run_algo(algo, local, ws, rank); dist.barrier(); elapsed=(time.perf_counter()-t0)*1000.0
        verify(gathered,msg,ws)
        t=torch.tensor([elapsed], dtype=torch.float64); dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if rank==0:
            results.append({'algorithm':algo,'world_size':ws,'msg_bytes_per_rank':msg,'samples_ms':[float(t.item())],'median_ms':float(t.item()),'min_ms':float(t.item()),'max_ms':float(t.item())})
    if rank==0:
        Path(result_file).write_text(json.dumps(results))
    dist.destroy_process_group()

if __name__=='__main__':
    import sys
    ws=int(sys.argv[1]); jobs_json=sys.argv[2]; out=sys.argv[3]
    mp.start_processes(worker, args=(ws, free_port(), jobs_json, out), nprocs=ws, start_method='fork', join=True)
