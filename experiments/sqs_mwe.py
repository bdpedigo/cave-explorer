# %%
from functools import partial

from taskqueue import LocalTaskQueue, queueable

tq = LocalTaskQueue(parallel=5)  # use 5 processes


@queueable
def print_task(txt):
    with open(f"{txt}.txt", "w") as f:
        f.write(str(txt))
    return 1


tasks = (partial(print_task, i) for i in range(10))

# tq.insert_all(tasks)


# %%
from taskqueue import TaskQueue

tq = TaskQueue("https://sqs.us-west-2.amazonaws.com/629034007606/ben-skedit")
tq.insert(tasks)

# %%
with TaskQueue("https://sqs.us-west-2.amazonaws.com/629034007606/ben-skedit") as tq:
    tq.poll(lease_seconds=1, verbose=True, tally=True)
