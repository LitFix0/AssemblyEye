"""Derive dataset variants (e.g. merged classes) from an exported YOLO dataset."""


def merge_classes(names: dict, merge: dict) -> tuple[list[str], dict[int, int]]:
    """names: {index: name}; merge: {old_name: new_name}.
    Returns (new class names in first-seen order, {old index: new index})."""
    new_names: list[str] = []
    mapping: dict[int, int] = {}
    for i in sorted(names):
        target = merge.get(names[i], names[i])
        if target not in new_names:
            new_names.append(target)
        mapping[int(i)] = new_names.index(target)
    return new_names, mapping


def remap_label_lines(text: str, mapping: dict[int, int]) -> str:
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        out.append(" ".join([str(mapping[int(parts[0])]), *parts[1:]]))
    return "\n".join(out) + ("\n" if out else "")