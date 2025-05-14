import os

import tomlkit


def main():
    pr_number = os.environ.get('PR_NUMBER')
    run_number = os.environ.get('RUN_NUMBER')

    if pr_number is None or run_number is None:
        print("Error: PR_NUMBER or RUN_NUMBER environment variables not set.")
        print("This script is intended for PR contexts. Skipping version update.")
        # Exiting gracefully if not in the intended PR context
        # This prevents the job from failing if somehow triggered without these vars
        exit(0)

    pyproject_path = 'pyproject.toml'
    print(f"Attempting to update version in {pyproject_path}")

    try:
        with open(pyproject_path, 'r', encoding='utf-8') as f:
            content = f.read()

        data = tomlkit.parse(content)

        if 'project' not in data or 'version' not in data['project']:
            print(f"Error: Could not find ['project']['version'] in {pyproject_path}")
            exit(1)

        base_version = str(data['project']['version'])
        # Prevent re-appending if the script runs multiple times on the same checkout with an already modified version
        if f".pr{pr_number}.run{run_number}" in base_version:
            print(f"Version already contains .pr{pr_number}.run{run_number}. Assuming already updated: {base_version}")
            exit(0)

        new_version = f"{base_version}.pr{pr_number}.run{run_number}"

        print(f"Original version: {base_version}")
        print(f"Updating to PR version: {new_version}")
        data['project']['version'] = new_version

        with open(pyproject_path, 'w', encoding='utf-8') as f:
            f.write(tomlkit.dumps(data))

        print(f"Successfully updated {pyproject_path} to version {new_version}")

    except FileNotFoundError:
        print(f"Error: {pyproject_path} not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred during version update: {e}")
        exit(1)

if __name__ == "__main__":
    main()
