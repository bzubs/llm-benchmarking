from schema import BenchTask
from cluster import GPUCluster


"""
scheduler to manage given GPU cluster and uses round-robin logic; 
to assign tasks to next available GPUs"""
class GPUScheduler:
    def __init__(self, cluster: GPUCluster):
        self.cluster = cluster
        self.rr_index = 0

    #multi-GPU scheduling
    def schedule_task(self, task: BenchTask):
        k = task.config.n_gpus_required

        # Try to allocate K GPUs
        nodes = self.cluster.get_free_gpus(k)

        if nodes:
            # SUCCESS: assign GPUs
            task.status = "assigned"
            task.gpu_assigned = [n.gpu.id for n in nodes]

            # mark busy
            self.cluster.reserve_gpus(nodes)

        else:
            # NOT ENOUGH GPUs → queue globally
            task.status = "queued"
            task.gpu_assigned = None

        # ALWAYS push to global queue
        self.cluster.pending_tasks.append(task)

    def try_schedule_pending_tasks(self):
        for task in self.cluster.pending_tasks:

            if task.status != "queued":
                continue

            k = task.config.n_gpus_required

            nodes = self.cluster.get_free_gpus(k)

            if not nodes:
                continue  # no more GPUs → stop early

            # assign now
            task.status = "assigned"
            task.gpu_assigned = [n.gpu.id for n in nodes]

            self.cluster.reserve_gpus(nodes)


    #-----------------------------------------------------------------------
    #deprecated in v6+; used for 1 task per GPU model
    def get_first_free_gpu(self):
        for node in self.cluster:
            if node.gpu.status == "free":
                return node
        return None

    #deprecated in v6+; used for 1 task per GPU model
    def get_next_rr_node(self):
        gpu_ids = self.cluster.get_gpu_ids()
        gpu_id = gpu_ids[self.rr_index]

        node = self.cluster.get_node(gpu_id)

        self.rr_index = (self.rr_index + 1) % len(gpu_ids)
        return node

    