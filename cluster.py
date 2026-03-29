from schema import GPU, GPUNode, BenchTask


class GPUCluster:
    def __init__(self, gpu_ids: list[int]):
        self._nodes = {}
        self._gpu_ids = gpu_ids

        # global task queue : introduced in v6
        self.pending_tasks: list[BenchTask] = []

        for gpu_id in gpu_ids:
            g = GPU(id=gpu_id, status="free")
            self._nodes[gpu_id] = GPUNode(gpu=g)


    # multi-gpu support
    def get_free_gpus(self, k: int):
        free_nodes = []

        for node in self._nodes.values():
            if node.gpu.status == "free":
                free_nodes.append(node)

            if len(free_nodes) == k:
                return free_nodes

        return None  # not enough GPUs

    def reserve_gpus(self, nodes: list[GPUNode]):
        for node in nodes:
            node.gpu.status = "busy"

    def release_gpus(self, gpu_ids: list[int]):
        for gid in gpu_ids:
            node = self.get_node(gid)
            node.gpu.status = "free"

    # ============================================
    # deprecated in v6+ (keep for compatibility)
    def append_task(self, gpu_id: int, task: BenchTask):
        # Deprecated: per-GPU queue
        if hasattr(self._nodes[gpu_id], "queue"):
            self._nodes[gpu_id].queue.append(task)

    def pop_task(self, gpu_id: int, task: BenchTask):
        node = self.get_node(gpu_id)

        if hasattr(node, "queue") and task in node.queue:
            node.queue.remove(task)

    def get_node(self, gpu_id: int):
        if gpu_id not in self._nodes:
            raise ValueError("Invalid GPU ID")
        return self._nodes[gpu_id]

    def get_task_queue(self, gpu_id: int):
        node = self.get_node(gpu_id)
        return getattr(node, "queue", [])

    def get_cluster_size(self):
        return len(self._gpu_ids)

    def get_gpu_ids(self):
        return list(self._gpu_ids)

    def get_nodes(self):
        return self._nodes.values()

    # --- Pythonic iteration support ---
    def __iter__(self):
        return iter(self._nodes.values())