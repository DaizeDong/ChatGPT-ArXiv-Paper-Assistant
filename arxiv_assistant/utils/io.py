import inspect

import os
import shutil


def create_dir(dir, print_info=False, suppress_errors=False) -> bool:
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
            if print_info:
                print(f"Created dir: {dir}")
            return True
        else:
            return False
    except Exception as e:
        if suppress_errors:
            print(f"Exception within `{inspect.currentframe().f_code.co_name}`: {e}")
            return False
        else:
            raise e


def delete_file_or_dir(path, print_info=False, suppress_errors=False) -> bool:
    try:
        if os.path.exists(path):
            is_file = os.path.isfile(path)
            if is_file:
                os.remove(path)
            else:
                shutil.rmtree(path)
            if print_info:
                print(f"Deleted {'file' if is_file else 'dir'}: {path}")
            return True
        else:
            return False
    except Exception as e:
        if suppress_errors:
            print(f"Exception within `{inspect.currentframe().f_code.co_name}`: {e}")
            return False
        else:
            raise e


def copy_file_or_dir(source_path, target_dir, print_info=False, suppress_errors=False) -> bool:
    try:
        if os.path.exists(source_path):
            create_dir(target_dir, suppress_errors=suppress_errors)
            is_file = os.path.isfile(source_path)
            if is_file:
                shutil.copy2(source_path, target_dir)
            else:
                shutil.copytree(source_path, os.path.join(target_dir, os.path.basename(source_path)), dirs_exist_ok=True)
            if print_info:
                print(f"Copied {'file' if is_file else 'dir'}: {source_path} -> {os.path.join(target_dir, os.path.basename(source_path))}")
            return True
        else:
            return False
    except Exception as e:
        if suppress_errors:
            print(f"Exception within `{inspect.currentframe().f_code.co_name}`: {e}")
            return False
        else:
            raise e
