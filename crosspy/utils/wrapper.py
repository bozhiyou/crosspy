from abc import ABCMeta, abstractmethod


class DynamicObjectManager(metaclass=ABCMeta):

    def __init__(self, archive_attributes={'device': 'cpu', 'shape': None}) -> None:
        assert isinstance(archive_attributes, (list, dict)), RuntimeError(
            "`archive_attributes` should be a list of attribute names or a dict with default values, not %s"
            % type(archive_attributes)
        )
        self.archive_attributes = archive_attributes
        self.attr_archive = {attr_name: {} for attr_name in archive_attributes}

    @abstractmethod
    def wrap(self, obj): ...

    @abstractmethod
    def get_device(self, id): ...

    def track(self, obj):
        if self.attr_archive:
            if isinstance(self.archive_attributes, list):
                attrs = {attr_name: getattr(obj, attr_name) for attr_name in self.archive_attributes}
            elif isinstance(self.archive_attributes, dict):
                attrs = {attr_name: getattr(obj, attr_name, default_value) for attr_name, default_value in self.archive_attributes.items()}
        wrapped_obj = self.wrap(obj)
        if self.attr_archive:
            for attr_name, attr in attrs.items():
                self.attr_archive[attr_name][id(wrapped_obj)] = attr
        return wrapped_obj