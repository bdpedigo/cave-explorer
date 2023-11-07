# %%
from functools import partial

from taskqueue import TaskQueue, queueable


@queueable
def print_task(txt):
    with open(f"{txt}-new.txt", "w") as f:
        f.write(str(txt))
    return 1


tasks = (partial(print_task, i) for i in range(10))

tq = TaskQueue("https://sqs.us-west-2.amazonaws.com/629034007606/ben-skedit")

tq.insert(tasks)
