import os
import shutil
from utilities import dict_from_json

def move_files(directory_path, to_move):
    data_dir = os.path.join(directory_path, '../data_copy')
    to_move_paths = [os.path.join(data_dir, i) for i in to_move]
    os.chdir(data_dir)

    if not os.path.exists("not_in_master"):
        os.makedirs("not_in_master")

    destination = os.path.join(data_dir, "not_in_master")

    for count, source in enumerate(to_move_paths):
        if os.path.exists(os.path.join(destination, to_move[count])) or os.path.exists(source) == False:
            continue
        else:
            shutil.move(source, destination)
    print("Done")


directory_path = os.getcwd()
has_no_class = dict_from_json('../id_not_in_masterfile.json')
move_files(directory_path, has_no_class)
