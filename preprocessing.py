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


def equialize_hist(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_ = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_


def get_clahe_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_dist = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return img_dist


def laplacian_process(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Laplacian(img, cv2.CV_64F)
    edges = 255 * (edges / np.max(np.abs(edges)))
    return edges.astype(np.int8)


def build_gabor_filters(
    ksize=31,
    sigma=5,
    theta_range=(0, np.pi, np.pi / 8),
    lambd=2.0,
    gamma=0.5,
):
    filters = []
    for theta in np.arange(*theta_range):
        kernel = cv2.getGaborKernel(
            ksize=(ksize, ksize),
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=0,
            # ktype=cv2.CV_32F,
        )
        # kernel /= 1.5 * kernel.sum()
        filters.append(kernel)
        cv2.imwrite(f"filter_{theta}_img.png", ((kernel + 1) * 128).astype(np.uint8))
    return filters


def apply_gabor_filters(image, filters):
    applied_images = []
    for filter_ in filters:
        filtered_img = cv2.filter2D(image, cv2.CV_8UC3, filter_)
        applied_images.append(filtered_img)

    # 合成
    img_gabor = (np.sum(np.array(applied_images), axis=0) / len(filters)).astype(np.uint8)
    return img_gabor


def get_sharpen_image(image):
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]
    )
    img_dst = cv2.filter2D(image_blur, -1, kernel=kernel)
    return img_dst


def get_unsharp_mask_image(image):
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    img_dst = cv2.addWeighted(
        src1=image,
        alpha=1.5,
        src2=image_blur,
        beta=-0.5,
        gamma=0,
    )
    return img_dst


def get_composed_image(image):
    # NOTICE: B:G:R
    # image_bin = image_binarized(image)
    image_cont = image_contrast(image)
    # image_line = cv2.Canny(image, 25, 100)
    image_laplacian = laplacian_process(image)
    image_lbp = local_binary_pattern(
        image=cv2.GaussianBlur(image, (5, 5), 0),
        P=5,
        R=1,
        method="default",
    )
    img_gabor = apply_gabor_filters(image=image, filters=build_gabor_filters())
    # composed_image = np.stack([image_bin, image_line, image_cont], axis=2)
    # composed_image = np.stack([image_cont, image_line, image_lbp], axis=2)
    composed_image = np.stack([image_cont, image_laplacian, image_lbp], axis=2)
    # composed_image = np.stack([image_cont, image_laplacian, img_gabor], axis=2)
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
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # img_ = get_composed_image(img)
        img = cv2.imread(img_path)
        # img_ = get_clahe_image(img)
        # img_ = get_sharpen_image(img_)
        # img_ = cv2.cvtColor(img_, cv2.COLOR_GRAY2BGR)
        img_ = equialize_hist(img)

        filename = os.path.basename(img_path)
        job_type = get_jobtype(img_path)
        class_ = get_class_(img_path)
        if job_type and class_:
            cv2.imwrite(os.path.join(DATASET_DIR, dist_dir, job_type, class_, filename), img_)
        else:
            assert False


if __name__ == "__main__":
    for src_dir_name in [
        # "re_annotation_crop_20230613_splited",
        # "dev3_testing_correction_v2_splited",
        "dev3_testing_correction_black_splited",
    ]:
        main(
            src_dir_name,
            src_dir_name + "_preprocessing_eq_hist2",
        )
