from schema import BenchmarkConfig, BenchTask
from cluster import GPUCluster

class GPUScheduler:
    def __init__(self, cluster: GPUCluster):
        self.cluster = cluster
        self.rr_index = 0


    def get_first_free_gpu(self):
        for node in self.cluster: 
            if node.gpu.status == "free":
                return node
        return None

    def get_next_rr_node(self):
        gpu_ids = self.cluster.get_gpu_ids()
        gpu_id = gpu_ids[self.rr_index]

        node = self.cluster.get_node(gpu_id)

        self.rr_index = (self.rr_index + 1) % len(gpu_ids)
        return node


    def schedule_task(self, task: BenchTask):
        node = self.get_first_free_gpu()

        if node:
            task.status = "assigned"
            task.gpu_assigned = node.gpu.id
            node.gpu.status = "busy"
        else:
            # Queue using round-robin
            node = self.get_next_rr_node()
            task.status = "queued"
            task.gpu_assigned = node.gpu.id

        self.cluster.append_task(node.gpu.id, task)

