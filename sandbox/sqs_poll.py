# %%

from taskqueue import TaskQueue, queueable


@queueable
def print_task(txt):
    with open(f"{txt}-new.txt", "w") as f:
        f.write(str(txt))
    return 1


def stop_func(elapsed_time, executed):
    if executed >= 11:
        return True
    if elapsed_time > 5:
        return True


tq = TaskQueue("https://sqs.us-west-2.amazonaws.com/629034007606/ben-skedit")
tq.poll(
    lease_seconds=1,
    verbose=False,
    tally=True,
    max_backoff_window=1,
    min_backoff_window=1,
    stop_fn=stop_func,
)
