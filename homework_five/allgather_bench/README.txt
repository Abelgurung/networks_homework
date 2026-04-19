AllGather benchmark artifacts
============================

Files:
- allgather_worker.py: point-to-point implementations of Ring, Recursive Doubling, and Swing AllGather using torch.distributed with Gloo.
- bench_session_fork.py: benchmark harness that spawns one process group per world size and runs multiple jobs inside the same group.
- allgather_results.json: collected benchmark results.
- allgather_vs_message_size.png: completion time vs message size at world size 8.
- allgather_vs_world_size.png: completion time vs world size at fixed 1 MB per-rank payload.

Benchmark settings used here:
- Backend: Gloo (CPU)
- Iterations per point: 1 timed iteration
- Message-size sweep (world size 8): 1 KB, 64 KB, 1 MB, 4 MB per rank
- World-size sweep (fixed 1 MB per rank): 2, 4, 8 ranks

Notes:
- Recursive Doubling and Swing are implemented for power-of-two world sizes in this harness.
- Swing uses the reverse-order peer schedule for AllGather based on the Swing paper's peer function.
