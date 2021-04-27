import os
import random

SEED = 42
TRAIN_PROPORTION = 0.6
EVAL_PROPORTION = 0.2
TEST_PROPORTION = 0.2
LABEL_TO_HOT_ENCODING_MAPPING = {"up": 0, "down": 1, "left": 2, "right": 3, "other": 4}



def create_file_list(data_path: str) -> list:
    file_list = []
    for files in os.listdir(data_path):
        file_list.append(files)
    return file_list


def create_data_file_lists(src_path: str):
    train_file = open(src_path + "train", "w")
    eval_file = open(src_path + "eval", "w")
    test_file = open(src_path + "test", "w")

    for path, directories, files in os.walk(src_path):
        for directory in directories:
            print(directory + " is processed")
            file_list = create_file_list(src_path + directory)
            random.Random(SEED).shuffle(file_list)
            num_train_elems = int(len(file_list) * TRAIN_PROPORTION)
            num_eval_elems = int(len(file_list) * EVAL_PROPORTION)
            num_test_elems = len(file_list) - (num_train_elems + num_eval_elems)
            print("Data split proportions: {:d} {:d} {:d}".format(num_train_elems, num_eval_elems, num_test_elems))
            for x in range(num_train_elems):
                train_file.write("{},{:d}\n".format(file_list[x], LABEL_TO_HOT_ENCODING_MAPPING[directory.split("_")[0]]))
            for x in range(num_eval_elems):
                eval_file.write("{},{:d}\n".format(file_list[x + num_train_elems],
                                                   LABEL_TO_HOT_ENCODING_MAPPING[directory.split("_")[0]]))
            for x in range(num_test_elems):
                test_file.write("{},{:d}\n".format(file_list[x + num_train_elems + num_eval_elems],
                                                   LABEL_TO_HOT_ENCODING_MAPPING[directory.split("_")[0]]))
    train_file.close()
    eval_file.close()
    test_file.close()
    
src_path = "/home/thinkpad/SpeechDatasetProject/google_speech_command_npy/"
create_data_file_lists(src_path)