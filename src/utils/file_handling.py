import os

def get_files_recursively(experiment_list):
    file_list = filter(lambda file : os.path.isdir(file), experiment_list)
    json_files = list(filter(lambda file : not os.path.isdir(file), experiment_list))

    for file in file_list:
        for root, dirs, files in os.walk(file):
            for file in files:
                #append the file name to the list
                json_files.append(os.path.join(root,file))
    return json_files