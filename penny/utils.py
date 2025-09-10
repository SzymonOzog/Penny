import torch
import torch.distributed as dist
import penny_cpp
import os
from typing import Optional, Union

class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


# CREDIT: https://github.com/deepseek-ai/DeepEP/blob/main/deep_ep/utils.py
def bench_kineto(fn, kernel_names: Union[str, tuple], num_tests: int = 30, suppress_kineto_output: bool = False,
                 trace_path: Optional[str] = None, barrier_comm_profiling: bool = True,
                 num_kernels_per_period: int = 1):
    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule) as prof:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    lhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device='cuda'))
                for _ in range(num_tests):
                    # Alocate big tensor to clear cache
                    _ = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
                    fn()
                prof.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = prof.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')
    # HACK remove the sync allreduce from lines
    prof_lines = [line for line in prof_lines if "AllReduce_Sum_f32" not in line] 
    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert sum([name in line for line in prof_lines]) == 1, f'Errors of the kernel {name} in the profiling table'

    # Save chrome traces
    if trace_path is not None:
        prof.export_chrome_trace(trace_path)

    # Return average kernel durations
    units = {'ms': 1e3, 'us': 1e6}
    kernel_durations = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_durations.append(float(time_str.replace(unit, '')) / scale)
                        break
                break

    # Expand the kernels by periods
    if num_kernels_per_period > 1:
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            prof.export_chrome_trace(tmp.name)
            profile_data = json.loads(Path(tmp.name).read_text())

        for i, kernel_name in enumerate(kernel_names):
            events = [event for event in profile_data['traceEvents'] if f'::{kernel_name}' in event['name']]
            events = sorted(events, key=lambda event: event['ts'])
            durations = [event['dur'] / 1e6 for event in events]
            assert len(durations) % num_kernels_per_period == 0
            num_kernel_patterns = len(durations) // num_kernels_per_period
            kernel_durations[i] = [sum(durations[j::num_kernels_per_period]) / num_kernel_patterns
                               for j in range(num_kernels_per_period)]

    #TODO Why does this sometimes fail
    try:
        return kernel_durations if is_tuple else kernel_durations[0]
    except:
        return 1


def initialize_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    nnodes = int(os.getenv("NNODES"))
    local_size = world_size//nnodes
    local_rank = dist.get_rank() % local_size

    torch.cuda.set_device(local_rank)
    nvshmem_uid = penny_cpp.get_unique_id()

    nvshmem_uids = [None, ] * world_size
    dist.all_gather_object(nvshmem_uids, nvshmem_uid)
    penny_cpp.init_with_uid(nvshmem_uids[0], dist.get_rank(), world_size)
