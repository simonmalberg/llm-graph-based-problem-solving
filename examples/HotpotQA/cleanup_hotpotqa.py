# get filenames in a folder
import os

def get_filenames(folder):
    filenames = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.json'):
                filenames.append(file)
    return filenames

def get_folder_names(folder):
    folder_names = []
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            folder_names.append(dir)
    return folder_names


if __name__ == "__main__":
    folder = 'examples/HotpotQA/results/chatgpt_final'
    foldernames = get_folder_names(folder)
    print(foldernames)
    filename_set = set()
    for i, foldername in enumerate(foldernames):
        filenames = get_filenames(folder + '/' + foldername)
        if i == 0:
            for filename in filenames:
                filename_set.add(filename)
        else:
            new_set = set()
            for filename in filenames:
                new_set.add(filename)
            filename_set = filename_set.intersection(new_set)
    print(filename_set)
    print(len(filename_set))
    for foldername in foldernames:
        filenames = get_filenames(folder + '/' + foldername)
        for filename in filenames:
            if filename not in filename_set:
                # delete file
                os.remove(folder + '/' + foldername + '/' + filename)