import os
import shutil


def create_dir(dir, suppress_errors=False):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except Exception as e:
        if suppress_errors:
            print(f"{e}\n(This exception have been suppressed and would not influence the program execution)")
        else:
            raise e


def delete_file_or_dir(dir, suppress_errors=False):
    try:
        if os.path.isfile(dir):
            os.remove(dir)
        elif os.path.exists(dir):
            shutil.rmtree(dir)
        else:
            pass
    except Exception as e:
        if suppress_errors:
            print(f"{e}\n(This exception have been suppressed and would not influence the program execution)")
        else:
            raise e


def copy_file_or_dir(source_dir, target_dir, print_info=False, suppress_errors=False):
    try:
        create_dir(target_dir, suppress_errors=suppress_errors)
        if os.path.isfile(source_dir):
            shutil.copy2(source_dir, target_dir)
        else:
            shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        if print_info:
            print(f"Copied: {source_dir} -> {target_dir}")
    except Exception as e:
        if suppress_errors:
            print(f"{e}\n(This exception have been suppressed and would not influence the program execution)")
        else:
            raise e


def copy_all_files_in_dir(source_dir, target_dir, print_info=False, suppress_errors=False):
    try:
        create_dir(target_dir, suppress_errors=suppress_errors)
        if os.path.isfile(source_dir):
            raise ValueError(f"source_dir {source_dir} is a file, not a directory")
        else:
            for item in os.listdir(source_dir):
                source_item = os.path.join(source_dir, item)
                target_item = os.path.join(target_dir, item)
                copy_file_or_dir(source_item, target_item, print_info=print_info, suppress_errors=suppress_errors)
    except Exception as e:
        if suppress_errors:
            print(f"{e}\n(This exception have been suppressed and would not influence the program execution)")
        else:
            raise e


def argsort(seq):
    # native python version of an 'argsort'
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)
