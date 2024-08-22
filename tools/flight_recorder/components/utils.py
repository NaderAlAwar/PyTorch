# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Set, Tuple  # type: ignore[attr-defined]

from tools.flight_recorder.components.types import (
    Group,
    MatchState,
    Membership,
    Op,
    P2P,
)


try:
    from tabulate import tabulate
except ModuleNotFoundError:
    print("tabulate is not installed. Proceeding without it.")


def match_one_event(
    event_a: Dict[Any, Any],
    event_b: Dict[Any, Any],
    memberships: Dict[str, Set[Any]],
) -> MatchState:
    op_a = Op(event_a, memberships)
    op_b = Op(event_b, memberships)
    return op_a.match(op_b)


def match_coalesced_groups(
    all_rank_events: Dict[Any, Any],
    group_size: int,
    groups: Dict[str, Group],
    memberships: Dict[str, Set[Any]],
) -> bool:
    """
    all_rank_events: {
        rank: [
            (idx, event_dict)
        ]
    }

    Note: it is possible for event dicts in a coalesced group to be asymmetric.
        e.g. the following events lists form a valid coalescing group
             events0 [send:1]
             events1 [recv:0, send:2]
             events2 [recv:1]

    Rule 1: all ops should find a match
    Rule 2: relative ordering of sends and recvs in one event list can be arbitrary
        e.g.
        events1 [recv:0, send:2]  —> okay
        events1 [send:2, recv:0] —> also okay
    Rule 3: sends to the same dest or recvs from the src should be in a consistent order
        e.g.
        rank0 [send:1 (100B), send:1 (1000B)]
        rank1 [recv:0 (1000B), recv:0 (100B)]   —> not okay
    """
    all_ops = {
        rank: [Op(e, memberships) for i, e in all_rank_events[rank]]
        for rank in all_rank_events
    }

    def visualize_ops(match: bool) -> None:
        all_ops = {
            rank: [Op(e, memberships) for i, e in all_rank_events[rank]]
            for rank in all_rank_events
        }

        i = 0
        row = []
        progress = True
        table = []
        while progress:
            progress = False
            for r in all_ops:
                if len(all_ops[r]) > i:
                    _, event = all_rank_events[r][i]
                    row.append(Op(event, memberships))
                    progress = True
                else:
                    row.append(None)  # type: ignore[arg-type]
            table.append(row)
            row = []
            i += 1
        title = "Match" if match else "MISMATCH"
        print(f"{title}\n", tabulate(table))  # type: ignore[operator]

    # TODO can't verify seq_id bc there might have been valid seq deltas between ranks even within a pg.
    for op_list in all_ops.values():
        if not op_list:
            # print("TODO- not sure if its valid for only some ranks in a PG to participate in a coalesced op?")
            return False
        assert op_list[-1].type == "coalesced"
        op_list.pop(-1)

    while all_ops:
        first_rank = next(iter(all_ops))
        my_ops = all_ops[first_rank]

        if len(all_ops[first_rank]) == 0:
            all_ops.pop(first_rank)
            continue

        # lets match the first collective! we need to know which ranks are involved, and ensure that this same
        # collective is also the first one on those ranks within that group
        op = my_ops[0]
        match_idx = -1
        if op.type in P2P:
            dst_global_rank = sorted(memberships[op.pg_name])[op.dst]
            peer_ops = all_ops[dst_global_rank]
            for i, other in enumerate(peer_ops):
                if op.match(other) == MatchState.FULLY_MATCHED:
                    match_idx = i
                    break
                elif op.dst == other.src:
                    # Rule 3
                    break
                else:
                    # Rule 1
                    continue
        else:
            raise NotImplementedError("coalesced collective ops")
        if match_idx >= 0:
            my_ops.pop(0)
            peer_ops.pop(match_idx)
        else:
            visualize_ops(False)
            return False

    visualize_ops(True)
    return True


def check_size_alltoall(alltoall_cases: List[Dict[str, Any]]) -> Tuple[bool, int, int]:
    input_numel = 0
    output_numel = 0
    for e in alltoall_cases:
        input_numel += math.prod(e["input_sizes"][0])
        output_numel += math.prod(e["output_sizes"][0])
    return input_numel == output_numel, input_numel, output_numel


def find_coalesced_group(
    pg_name: str, entries: List[Dict[str, Any]]
) -> List[Tuple[int, Dict[str, Any]]]:
    """Given a list of entries, if the collective_seq_id of the first entry matches that of subsequent ones,
    build an return a list of entries terminating in a 'coalesced' op entry all sharing a collective_seq_id
    TODO: handle p2p_seq_id v/s collective_seq_id separately here.
    """
    found = []
    collective_seq_id = None
    for i, e in enumerate(entries):
        if e["process_group"][0] != pg_name:
            continue
        elif collective_seq_id is None:
            collective_seq_id = e["collective_seq_id"]
            found.append((i, e))
        elif e["collective_seq_id"] == collective_seq_id:
            found.append((i, e))
        else:
            break

    if len(found) > 1:
        assert found[-1][1]["profiling_name"] == "nccl:coalesced"
        return found
    return []


def just_print_entries(
    all_entries: Dict[int, List[Dict[str, Any]]],
    _groups: Dict[str, Group],
    _memberships: Dict[str, Set[Any]],
) -> None:
    rows = []
    ranks = sorted(all_entries.keys())
    headers = [f"Rank {rank}" for rank in ranks]
    progress = True
    while progress:
        progress = False
        row = []
        for rank in ranks:
            if len(all_entries[rank]) == 0:
                row.append("")
            else:
                entry = all_entries[rank].pop(0)
                row.append(str(Op(entry, _memberships)))
                progress = True
        if progress:
            rows.append(row)

    print(tabulate(rows, headers=headers))


def check_no_missing_dump_files(
    entries: Dict[str, Any], memberships: List[Membership]
) -> None:
    all_ranks = set()
    for membership in memberships:
        all_ranks.add(str(membership.global_rank))
    dumps_ranks = set(entries.keys())
    assert (
        dumps_ranks == all_ranks
    ), f"Missing dump files from ranks {all_ranks - dumps_ranks}"


def check_version(versions: Dict[str, Any]) -> None:
    for rank, version in versions.items():  # noqa: PERF102
        major, minor = map(int, version.split("."))
        # assert major == 2, f"Rank {rank} unsupported version {version}"
        # assert minor >= 0, f"Rank {rank} unsupported version {version}"


# TODO: We need to revisit this function to see if we still need it.
def check_trace_from_beginning(entries: Dict[str, Any]) -> bool:
    for rank in entries:
        first_record_id = entries[rank][0]["record_id"]
        # TODO add more sequence information such that analysis can proceed even without complete buffer

        # assert first_record_id == 0, f"Rank {rank} trace does not start at time 0 (first record is {first_record_id}."
        if first_record_id != 0:
            print(
                f"Rank {rank} trace does not start at time 0 (first record is {first_record_id}."
            )
            return False
    return True
