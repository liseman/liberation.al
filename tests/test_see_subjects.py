from liberational.people import registry
from liberational.see import subjects


def test_registry_people_are_their_person_ids(tmp_path) -> None:
    reg = registry.at(tmp_path)
    reg.ensure("Sam", "test")
    people = subjects.RegistryPeople(reg)
    assert people.person("Sam") == str(reg.person_id("Sam"))
    assert people.person("sam") == people.person("Sam"), "names match without case, as the registry's do"
    assert people.person("Nobody") is None
