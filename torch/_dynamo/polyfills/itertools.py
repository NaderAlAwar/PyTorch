"""
Python polyfills for itertools
"""

from __future__ import annotations

import itertools
from typing import Iterable, Iterator, TypeVar

from ..decorators import substitute_in_graph


__all__ = [
    "chain",
    "chain_from_iterable",
    "tee",
]


_T = TypeVar("_T")


# Reference: https://docs.python.org/3/library/itertools.html#itertools.chain
@substitute_in_graph(itertools.chain, is_embedded_type=True)  # type: ignore[arg-type]
def chain(*iterables: Iterable[_T]) -> Iterator[_T]:
    for iterable in iterables:
        yield from iterable


@substitute_in_graph(itertools.chain.from_iterable)  # type: ignore[arg-type]
def chain_from_iterable(iterable: Iterable[Iterable[_T]], /) -> Iterator[_T]:
    return itertools.chain(*iterable)


chain.from_iterable = chain_from_iterable  # type: ignore[method-assign]


# Reference: https://docs.python.org/3/library/itertools.html#itertools.tee
@substitute_in_graph(itertools.tee)
def tee(iterable: Iterable[_T], n: int = 2, /) -> tuple[Iterator[_T], ...]:
    iterator = iter(iterable)
    shared_link = [None, None]

    def _tee(link) -> Iterator[_T]:  # type: ignore[no-untyped-def]
        try:
            while True:
                if link[1] is None:
                    link[0] = next(iterator)
                    link[1] = [None, None]
                value, link = link
                yield value
        except StopIteration:
            return

    return tuple(_tee(shared_link) for _ in range(n))
