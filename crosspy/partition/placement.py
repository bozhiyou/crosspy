from typing import List, Iterable, Collection, Any, Union, FrozenSet

from ..device import Device, Architecture, get_all_devices
from crosspy.utils.array import is_array, get_memory

# TODO (bozhi): We may need a centralized typing module to reduce types being imported everywhere.
PlacementSource = Union[Architecture, Device, Any]


# TODO (bozhi): We may need a `placement` module to hold these `get_placement_for_xxx` interfaces, which makes more sense than the `tasks` module here. Check imports when doing so.
def get_placement_for_value(p: PlacementSource) -> List[Device]:
    if hasattr(p, "__parla_placement__"):
        # this handles Architecture, ResourceRequirements, and other types with __parla_placement__
        return list(p.__parla_placement__())
    elif isinstance(p, Device) or hasattr(p, 'memory'): # ad hoc using memory as identifier
        return [p]
    elif is_array(p):
        return [get_memory(p).device]
    elif p.__class__.__module__.startswith("cupy"):
        from crosspy.device import gpu
        return [gpu(p.id)]
    elif isinstance(p, Collection):
        raise TypeError(
            "Collection passed to get_placement_for_value, probably needed get_placement_for_set: {}"
            .format(type(p))
        )
    else:
        raise TypeError(type(p))


def get_placement_for_set(
    placement: Collection[PlacementSource]
) -> FrozenSet[Device]:
    if not isinstance(placement, Collection):
        raise TypeError(type(placement))
    devices = [d for p in placement for d in get_placement_for_value(p)]
    return devices if isinstance(placement, list) else frozenset(devices)


def get_placement_for_any(
    placement: Union[Collection[PlacementSource], Any, None]
) -> FrozenSet[Device]:
    if placement is not None:
        ps = placement if isinstance(placement, Iterable
                                    ) and not is_array(placement) else [
                                        placement
                                    ]
        return get_placement_for_set(ps)
    return frozenset(get_all_devices())