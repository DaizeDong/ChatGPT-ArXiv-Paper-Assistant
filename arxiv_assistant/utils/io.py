import inspect

import os
import shutil


def create_dir(dir, suppress_errors=False) -> bool:
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
            return True
        else:
            return False
    except Exception as e:
        if suppress_errors:
            print(f"Exception within `{inspect.currentframe().f_code.co_name}`: {e}")
            return False
        else:
            raise e


def delete_file_or_dir(path, suppress_errors=False) -> bool:
    try:
        if os.path.isfile(path):
            os.remove(path)
            return True
        elif os.path.exists(path):
            shutil.rmtree(path)
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
            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_dir)
            else:
                shutil.copytree(source_path, os.path.join(target_dir, os.path.basename(source_path)), dirs_exist_ok=True)
            if print_info:
                print(f"Copied: {source_path} -> {os.path.join(target_dir, os.path.basename(source_path))}")
            return True
        else:
            return False
    except Exception as e:
        if suppress_errors:
            print(f"Exception within `{inspect.currentframe().f_code.co_name}`: {e}")
            return False
        else:
            raise e
