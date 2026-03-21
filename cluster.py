from schema import GPU, GPUNode, BenchTask


class GPUCluster:
    def __init__(self, gpu_ids: list[int]):
        self._nodes = {}
        self._gpu_ids = gpu_ids

        for gpu_id in gpu_ids:
            g = GPU(id=gpu_id, status="free")
            self._nodes[gpu_id] = GPUNode(gpu=g)

    def get_cluster_size(self):
        return len(self._gpu_ids)

    def get_gpu_ids(self):
        return list(self._gpu_ids)

    def get_nodes(self):
        return self._nodes.values()

    def append_task(self, gpu_id: int, task: BenchTask):
        self._nodes[gpu_id].queue.append(task)

    def pop_task(self, gpu_id: int, task: BenchTask):
        node = self.get_node(gpu_id)

        node.queue.remove(task)
        if not node.queue:
            node.gpu.status = "free"


    def get_node(self, gpu_id: int):
        if gpu_id not in self._nodes:
            raise ValueError("Invalid GPU ID")
        return self._nodes[gpu_id]

    def get_task_queue(self, gpu_id: int):
        return self.get_node(gpu_id).queue

    # --- Pythonic iteration support ---
    def __iter__(self):
        return iter(self._nodes.values())
