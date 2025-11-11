from typing import Iterable
def sample_prompts(stream: Iterable[str], n: int = 1000):
    for i, s in enumerate(stream):
        if i >= n: break
        yield s
