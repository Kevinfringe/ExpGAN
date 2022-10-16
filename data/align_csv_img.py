"""
    This file is intended for align the generated landmarks and original images.
    Since there are some images that Openface cannot detect landmarks on that.
"""
import os

posed_img_path = "../pose_set"
genuine_img_path = "../genuine_set"
lm_posed_path = "../lm_posed/"
lm_genuine_path = "../lm_genuine/"
counter = 0
filelist = []

# for file in os.listdir(lm_posed_path):
#     if file.endswith(".txt"):
#         os.remove(os.path.join(lm_posed_path, file))

'''
    Uncomment this part of code, and change the path if the original image folder
    is not aligned with generated landmark folder (csv files)
'''
# for file in os.listdir(lm_posed_path):
#     filename = file.strip('.csv')
#     if not os.path.exists(os.path.join(posed_img_path, filename + ".jpg")):
#         os.remove(os.path.join(lm_posed_path, file))
#         counter += 1
#         filelist.append(file)
#
# print(filelist)
# print(counter)

'''
    Uncomment this part of code, and change the path if pose_set and genuine set are not
    aligned.
'''
# Align pose_set and genuine_set
# for file in os.listdir(genuine_img_path):
#     filename = file.replace("G", "P")
#     print(filename)
#     if not os.path.exists(os.path.join(posed_img_path, filename)):
#         filelist.append(file)
#
# print(filelist)

