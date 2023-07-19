from glob import glob
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def image_binarized(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_output_value = 255
    neighborhood_size = 99
    subtract_from_mean = 10
    image_binarized = cv2.adaptiveThreshold(
        img,
        max_output_value,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        neighborhood_size,
        subtract_from_mean,
    )
    image_binarized = 255 - image_binarized
    return image_binarized


def image_contrast(img):
    # orb = cv2.ORB_create()
    # kp, des = orb.detectAndCompute(img, None)
    # img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), flags=0)
    # contrast = cv2.inRange(img2, (255, 0, 0), (255, 0, 0))
    img_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=40)
    return img_contrast


def get_composed_image(image):
    # NOTICE: B:G:R
    # image_bin = image_binarized(image)
    image_cont = image_contrast(image)
    image_line = cv2.Canny(image, 25, 100)
    image_lbp = local_binary_pattern(image=image, P=40, R=15, method="default")
    # composed_image = np.stack([image_bin, image_line, image_cont], axis=2)
    composed_image = np.stack([image_cont, image_line, image_lbp], axis=2)
    return composed_image


def get_jobtype(path):
    if "/train" in path:
        return "train"
    elif "/val" in path:
        return "val"
    elif "/test" in path:
        return "test"
    else:
        assert False


def get_class_(path):
    if "/normal" in path:
        return "normal"
    elif "/anomalous" in path:
        return "anomalous"
    else:
        assert False


def main(src_dir, dist_dir):
    DATASET_DIR = "./dataset"
    pngs = sorted((glob(f"{DATASET_DIR}/{src_dir}/**/*.png", recursive=True)))
    jpgs = sorted((glob(f"{DATASET_DIR}/{src_dir}/**/*.jpg", recursive=True)))
    all_imgs = pngs + jpgs

    for job_type in ["train", "val", "test"]:
        for class_ in ["anomalous", "normal"]:
            os.makedirs(os.path.join(DATASET_DIR, dist_dir, job_type, class_))

    for img_path in all_imgs:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_ = get_composed_image(img)

        filename = os.path.basename(img_path)
        job_type = get_jobtype(img_path)
        class_ = get_class_(img_path)
        if job_type and class_:
            cv2.imwrite(os.path.join(DATASET_DIR, dist_dir, job_type, class_, filename), img_)
        else:
            assert False


if __name__ == "__main__":
    main(
        "dev3_testing_correction_v2_splited",
        "dev3_testing_correction_v2_splited_preprocessing_cont_canny_lbp",
    )
