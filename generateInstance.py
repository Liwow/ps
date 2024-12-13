import os
import random
import time
from multiprocessing import Process, Queue

import ecole
import os

os.environ["LD_DEBUG"] = "libs"

# prefix = '../../../../project/predict_and_search/'
prefix = './'


# prefix  = ''

def generate_single_instance(n, queue, istrain, size, generator, seed):
    generator.seed(seed)
    while True:
        i = queue.get()
        if i is None:
            break  # No more tasks

        instance = next(generator)
        instance_dir = prefix + f"instance/{istrain}/{size}"
        os.makedirs(instance_dir, exist_ok=True)
        instance_path = os.path.join(instance_dir, f"{size}_{n + i}.lp")
        instance.write_problem(instance_path)
        print(f"第{n + i}个问题实例已生成：{instance_path}")


def generate_instances(num_instances, istrain, size, epoch=0):
    if size == "CF":
        generator = ecole.instance.CapacitatedFacilityLocationGenerator(50, 100)
    elif size == "IS":
        generator = ecole.instance.IndependentSetGenerator(1500)
    elif size == "CA_m" or size == "CA":
        if epoch == 0:
            generator = ecole.instance.CombinatorialAuctionGenerator(100, 500)
        elif epoch == 1:
            generator = ecole.instance.CombinatorialAuctionGenerator(200, 1000)
        elif epoch == 2:
            generator = ecole.instance.CombinatorialAuctionGenerator(300, 1500)
        else:
            generator = ecole.instance.CombinatorialAuctionGenerator(300, 1500)
    elif size == "SC":
        generator = ecole.instance.SetCoverGenerator(1000, 2000)
    else:
        raise ValueError("Invalid type")
    base_seed = random.randint(0, 2 ** 16 - 1)
    # seed = int(time.time())
    observation_function = ecole.observation.MilpBipartite()

    # Create a queue to hold tasks
    task_queue = Queue()
    # n = epoch * num_instances
    n = 0
    # Add tasks to queue
    for i in range(num_instances):
        task_queue.put(i)

    # Number of worker processes
    num_workers = 32

    # Create worker processes
    workers = []
    for worker_id in range(num_workers):
        seed = base_seed + worker_id
        worker = Process(target=generate_single_instance,
                         args=(n, task_queue, istrain, size, generator, seed))
        workers.append(worker)
        worker.start()

    # Add None to the queue to signal workers to exit
    for _ in range(num_workers):
        task_queue.put(None)

    # Wait for all worker processes to finish
    for worker in workers:
        worker.join()


if __name__ == '__main__':
    mix = False
    if mix:
        for i in range(3):
            generate_instances(100, "test", "CA", epoch=i)
    else:
        generate_instances(300, "train", "CA", epoch=2)
