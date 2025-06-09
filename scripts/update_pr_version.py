import os
import pathlib
import tomllib


def main():
    pr_number = os.environ.get("PR_NUMBER")
    run_number = os.environ.get("RUN_NUMBER")

    if run_number is None:
        print("Error: RUN_NUMBER environment variable not set.")
        print("This script is intended for CI contexts. Skipping version update.")
        exit(0)

    root_dir = pathlib.Path(__file__).parent.parent
    pyproject_path = root_dir / "pyproject.toml"
    dist_dir = root_dir / "dist"
    version_file_path = dist_dir / "VERSION.txt"

    print(f"Attempting to update version in {pyproject_path} (from {os.getcwd()})")

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        if "project" not in data or "version" not in data["project"]:
            print(f"Error: Could not find ['project']['version'] in {pyproject_path}")
            exit(1)

        base_version = str(data["project"]["version"])

        if "+" in base_version:
            base_version = base_version.split("+", 1)[0]
        if ".dev" in base_version:
            parts = base_version.split(".dev")
            if len(parts) > 1 and parts[-1].isdigit():
                base_version = parts[0]

        current_full_version = str(data["project"]["version"])
        expected_dev_suffix = f".dev{run_number}"
        if current_full_version.endswith(expected_dev_suffix):
            print(
                f"Version {current_full_version} already ends with {expected_dev_suffix}. Assuming already updated."
            )
            new_version = current_full_version
        else:
            new_version = f"{base_version}.dev{run_number}"

        print(f"Original full version in {pyproject_path}: {current_full_version}")
        print(f"Base version for new construction: {base_version}")
        print(f"Updating to PR dev version: {new_version} (PR: {pr_number})")

        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()

        version_pattern = f'version = "{current_full_version}"'
        new_version_line = f'version = "{new_version}"'
        updated_content = content.replace(version_pattern, new_version_line)

        with open(pyproject_path, "w", encoding="utf-8") as f:
            f.write(updated_content)

        print(f"Successfully updated {pyproject_path} to version {new_version}")

        dist_dir.mkdir(exist_ok=True)
        with open(version_file_path, "w", encoding="utf-8") as vf:
            vf.write(new_version)
        print(f"Successfully wrote version {new_version} to {version_file_path}")

    except FileNotFoundError:
        print(f"Error: {pyproject_path} not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred during version update: {e}")
        exit(1)


if __name__ == "__main__":
    main()
