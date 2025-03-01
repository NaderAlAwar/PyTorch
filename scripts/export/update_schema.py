import argparse
import os

from yaml import dump, Dumper

from torch._export.serde import schema_check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="update_schema")
    parser.add_argument(
        "--prefix", type=str, required=True, help="The root of pytorch directory."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the schema instead of writing it to file.",
    )
    parser.add_argument(
        "--force-unsafe",
        action="store_true",
        help="!!! Only use this option when you are a chad. !!! Force to write the schema even if schema validation doesn't pass.",
    )
    args = parser.parse_args()

    assert os.path.exists(args.prefix), (
        f"Assuming path {args.prefix} is the root of pytorch directory, but it doesn't exist."
    )

    commit = schema_check.update_schema()

    if os.path.exists(args.prefix + commit.yaml_path):
        if commit.result["SCHEMA_VERSION"] < commit.base["SCHEMA_VERSION"]:
            raise RuntimeError(
                f"Schema version downgraded from {commit.base['SCHEMA_VERSION']} to {commit.result['SCHEMA_VERSION']}."
            )

        if commit.result["TREESPEC_VERSION"] < commit.base["TREESPEC_VERSION"]:
            raise RuntimeError(
                f"Treespec version downgraded from {commit.base['TREESPEC_VERSION']} to {commit.result['TREESPEC_VERSION']}."
            )
    else:
        assert args.force_unsafe, (
            "Existing schema yaml file not found, please use --force-unsafe to try again."
        )

    next_version, reason = schema_check.check(commit, args.force_unsafe)

    if next_version is not None and next_version != commit.result["SCHEMA_VERSION"]:
        raise RuntimeError(
            f"Schema version is not updated from {commit.base['SCHEMA_VERSION']} to {next_version}.\n"
            + "Please either:\n"
            + "    1. update schema.py to not break compatibility.\n"
            + "    or 2. bump the schema version to the expected value.\n"
            + "    or 3. use --force-unsafe to override schema.yaml (not recommended).\n "
            + "and try again.\n"
            + f"Reason: {reason}"
        )

    first_line = (
        "@" + "generated by " + os.path.basename(__file__).rsplit(".", 1)[0] + ".py"
    )
    checksum = f"checksum<<{commit.checksum_next}>>"
    yaml_header = "# " + first_line
    yaml_header += "\n# " + checksum
    yaml_payload = dump(commit.result, Dumper=Dumper, sort_keys=False)

    cpp_header = "// " + first_line
    cpp_header += "\n// " + checksum
    cpp_header += "\n// clang-format off"
    cpp_header += "\n" + commit.cpp_header
    cpp_header += "\n// clang-format on"
    cpp_header += "\n"

    yaml_content = yaml_header + "\n" + yaml_payload

    thrift_schema = "// " + first_line
    thrift_schema += f"\n// checksum<<{commit.thrift_checksum_next}>>"
    thrift_schema += "\n" + commit.thrift_schema

    if args.dry_run:
        print(yaml_content)
        print("\nWill write the above schema to" + args.prefix + commit.yaml_path)
    else:
        with open(os.path.join(args.prefix, commit.yaml_path), "w") as f:
            f.write(yaml_content)
        with open(os.path.join(args.prefix, commit.cpp_header_path), "w") as f:
            f.write(cpp_header)
        with open(os.path.join(args.prefix, commit.thrift_schema_path), "w") as f:
            f.write(thrift_schema)
