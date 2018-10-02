"""
A file with generally helpful scripts.
"""


def set_working_dir():
    """
    Set the right working directory to root folder of the project
    """
    import os

    print("Looking for root folder of the project...")
    for folder_depth in range(100):
        if os.path.exists(".git"):
            print("Root folder found. Now working in directory '%s'" % os.getcwd())
            return os.getcwd()
        else:
            print("Going up from '%s'" % os.getcwd())
            os.chdir("..")
    else:
        raise Exception("Root folder of the project not found. Terminating.")


# TODO: visualise images without opencv
# def visualise_random_images(images, window_names):
#     """
#     Display random images from the set until ESC is pressed. Any other key displays another image
#     """
#     import cv2
#     import numpy as np
#
#     assert len(images) == len(window_names)
#     terminate_flag = False
#
#     while not terminate_flag:
#         img_index = np.random.randint(0, images.shape[0])
#         win_name = str(window_names[img_index])
#         cv2.imshow(win_name, images[img_index])
#
#         while True:
#             key = cv2.waitKey(33)
#             if win_name is not None and cv2.getWindowProperty(win_name, 0) == -1:
#                 terminate_flag = True
#                 break
#             if key != -1:
#                 break
#
#         if key == 27:  # ESC
#             terminate_flag = True
#
#         cv2.destroyWindow(win_name)
