'''
    This part of code is to truncate the dataset, which is to
    discard the neutral part, only keep the emotional part (via a state of art
    FER system) -- RMN
    Link: https://github.com/phamquiluan/ResidualMaskingNetwork
'''
import copy

from rmn import RMN
import os
import cv2
import shutil
import numpy as np

categories_dict = {
    "N2H": "GH",               # Genuine happiness.
    "N2S": "GS",               # Genuine sad.
    "N2D": "GD",               # Genuine disgust.
    "N2A": "GA",               # Genuine anger.
    "N2C": "GC",               # Genuine contempt.
    "N2Sur": "GSur",           # Genuine surprise.
    "S2N2H": "PH",             # Posed happiness.
    "H2N2S": "PS",             # Posed sadness.
    "H2N2D": "PD",             # Posed disgust.
    "H2N2A": "PA",             # Posed anger.
    "H2N2C": "PC",             # Posed contempt.
    "D2N2Sur": "PSur"          # Posed surprise.
}

emotion_dict = {
    "N2H": "happy",
    "N2S": "sad",
    "N2D": "disgust",
    "N2A": "angry",
    "N2C": "GC",
    "N2Sur": "surprise",
    "S2N2H": "happy",
    "H2N2S": "sad",
    "H2N2D": "disgust",
    "H2N2A": "angry",
    "H2N2C": "PC",
    "D2N2Sur": "surprise"
}

flag_dict = {
    "N2H": False,
    "N2S": False,
    "N2D": False,
    "N2A": False,
    "N2Sur": False,
    "S2N2H": False,
    "H2N2S": False,
    "H2N2D": False,
    "H2N2A": False,
    "D2N2Sur": False
}

paired_folder = {
    "N2H": "S2N2H",
    "N2S": "H2N2S",
    "N2D": "H2N2D",
    "N2A": "H2N2A",
    "N2Sur": "D2N2Sur",
    "S2N2H": "N2H",
    "H2N2S": "N2S",
    "H2N2D": "N2D",
    "H2N2A": "N2A",
    "D2N2Sur": "N2Sur"
}

# Define the path variables.
root_path = os.getcwd().replace("\\", "/")
print("root_path is :" + root_path)
train_path = "../Real_Fake_challenge/Real_Fake_challenge/Train/Train/"
pose_set = "../pose_set"
genuine_set = "../genuine_set"
directory = train_path

# Define the list to store posed and genuine images.
posed_list = []
gen_list = []

# Initialize the RMN instance.
fer_model = RMN()

# Traversing all the image files, and discard the neutral part.
# for folder in os.listdir(directory):
#     flag_dict_copy = copy.copy(flag_dict)
#     print("The current index of subject is: " + str(folder))
#     for img_folder in os.listdir(os.path.join(directory, folder)):
#         print("The img_folder is:" + str(img_folder))
#         if img_folder.endswith('C'):         # Ignore the contempt emotion type in dataset.
#             continue
#         if flag_dict_copy.get(img_folder):
#             continue
#
#         temp_wd = os.path.join(directory, folder)
#         print("temp_wd is: " + str(temp_wd))
#         emo_type = emotion_dict.get(img_folder)
#         flag_pose = True if categories_dict.get(img_folder).startswith('P') else False
#
#         # Find the paired folder.
#         next_folder = paired_folder.get(img_folder)
#
#         # Set the flags.
#         flag_dict_copy[next_folder] = True
#         flag_dict_copy[img_folder] = True
#
#         # Put images path into list.
#         for img in os.listdir(os.path.join(temp_wd + '/', img_folder)):
#             temp = os.path.join(temp_wd + '/', img_folder)
#             print("temp is: " + str(temp))
#             img_path = os.path.join(temp + '/', img)
#             print("img_path in first branch is :")
#             print(img_path)
#             image = cv2.imread(img_path)
#             # In case some images cannot be read out.
#             if not isinstance(image, np.ndarray):
#                 continue
#             result = fer_model.detect_emotion_for_single_face_image(image)[0]
#
#             # If the detected emotion is not aligned with the label, discard this frame.
#             if result != emo_type:
#                 continue
#             else:
#                 if flag_pose:
#                     posed_list.append(img_path)
#                 else:
#                     gen_list.append(img_path)
#
#         # put images path into list.
#         for img in os.listdir(os.path.join(temp_wd + '/', next_folder)):
#             temp = os.path.join(temp_wd + '/', next_folder)
#             print("temp in next_folder is: " + str(temp))
#             img_path = os.path.join(temp + '/', img)
#             print("img_path in second branch is :")
#             print(img_path)
#             image = cv2.imread(img_path)
#             # In case some images cannot be read out.
#             if not isinstance(image, np.ndarray):
#                 continue
#             result = fer_model.detect_emotion_for_single_face_image(image)[0]
#
#             # If the detected emotion is not aligned with the label, discard this frame.
#             if result != emo_type:
#                 continue
#             else:
#                 if not flag_pose:
#                     posed_list.append(img_path)
#                 else:
#                     gen_list.append(img_path)
#
#         # Copy detected images into corresponding folders.
#         # And rename these files
#         print("min(len(posed_list), len(gen_list)) :" + str(min(len(posed_list), len(gen_list))))
#         print(os.getcwd())
#         for i in range(min(len(posed_list), len(gen_list))):
#
#             if flag_pose:
#                 filename = str(folder) + "_" + categories_dict.get(img_folder) + str(i) + ".jpg"
#                 print("The generated in if branch filename is " + str(filename))
#                 target_path = pose_set + "/" + filename
#                 image_1 = cv2.imread(posed_list[i])
#                 cv2.imwrite(target_path, image_1)
#
#                 filename_oppo = str(folder) + "_" + categories_dict.get(next_folder) + str(i) + ".jpg"
#                 print("The filename_oppo in if branch filename is " + str(filename_oppo))
#                 target_path = genuine_set + "/" + filename_oppo
#                 image_2 = cv2.imread(gen_list[i])
#                 cv2.imwrite(target_path, image_2)
#             else:
#                 filename = str(folder) + "_" + categories_dict.get(img_folder) + str(i) + ".jpg"
#                 print("The generated in else branch filename is " + str(filename))
#                 target_path = genuine_set + "/" + filename
#                 image_1 = cv2.imread(gen_list[i])
#                 cv2.imwrite(target_path, image_1)
#
#                 filename_oppo = str(folder) + "_" + categories_dict.get(next_folder) + str(i) + ".jpg"
#                 print("The filename_oppo in else branch filename is " + str(filename_oppo))
#                 target_path = pose_set + "/" + filename_oppo
#                 image_2 = cv2.imread(posed_list[i])
#                 cv2.imwrite(target_path, image_2)
#
#         # Clear two list.
#         posed_list.clear()
#         gen_list.clear()



# Test the format of output
m = RMN()
img_path = os.path.join(train_path, "9/D2N2Sur/0220.jpg")
image = cv2.imread(img_path)
result = m.detect_emotion_for_single_face_image(image)
print(result)

# Manually select more images
# select_path = "../Real_Fake_challenge/Real_Fake_challenge/Train/Train/32/N2S"
# imgs = os.listdir(select_path)
# i = 0
# flag = False
# for img in imgs:
#     if img.startswith("0124") or flag:
#         flag = True
#         # filename = "32_PS" + str(i) + ".jpg"
#         # target_path = pose_set + "/" + filename
#         # image_1 = cv2.imread(os.path.join(select_path, img))
#         # cv2.imwrite(target_path, image_1)
#
#         filename_oppo = "32_GS" + str(i) + ".jpg"
#         target_path = genuine_set + "/" + filename_oppo
#         image_1 = cv2.imread(os.path.join(select_path, img))
#         cv2.imwrite(target_path, image_1)
#         i = i+1
#
# print("Total extracted files are : " + str(i))

# Count the total files in pose_set and genuine_set.
surprise_counter = 0
disgust_counter = 0
happy_counter = 0
sad_counter = 0
angry_counter = 0
counter = np.zeros((41, 5))

for img in os.listdir(pose_set):
    img_split = img.split("_")
    subject_index = (int)(img_split[0])
    if img_split[1].startswith("GH") or img_split[1].startswith("PH"):
        counter[subject_index, 2] = counter[subject_index, 2] + 1
        happy_counter = happy_counter + 1
    elif img_split[1].startswith("GSur") or img_split[1].startswith("PSur"):
        counter[subject_index, 0] = counter[subject_index, 0] + 1
        surprise_counter = surprise_counter + 1
    elif img_split[1].startswith("GD") or img_split[1].startswith("PD"):
        counter[subject_index, 1] = counter[subject_index, 1] + 1
        disgust_counter = disgust_counter + 1
    elif img_split[1].startswith("GS") or img_split[1].startswith("PS"):
        counter[subject_index, 3] = counter[subject_index, 3] + 1
        sad_counter = sad_counter + 1
    elif img_split[1].startswith("GA") or img_split[1].startswith("PA"):
        counter[subject_index, 4] = counter[subject_index, 4] + 1
        angry_counter = angry_counter + 1

print(counter)
print(np.sum(counter, axis=0))