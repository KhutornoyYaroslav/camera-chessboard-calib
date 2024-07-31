import logging
import argparse
import cv2 as cv
import numpy as np
from core.config import cfg as cfg
from core.utils.logger import setup_logger


def create_world_pts(rows: int, cols: int, size_mm: int):
    world_pts = []
    for i in range(0, rows):
        for j in range(0, cols):
            world_pts.append([j * size_mm, i * size_mm, 0.0])

    return np.expand_dims(np.array(world_pts, np.float32), 0) # (x, y, 0.0), in milimeters, shape=(1, N, 3)


def calibrate(video_path: str):
    logger = logging.getLogger('CORE')

    pattern_size = (cfg.PATTERN.COLS, cfg.PATTERN.ROWS)

    # prepare world points
    world_pts = create_world_pts(cfg.PATTERN.ROWS, cfg.PATTERN.COLS, cfg.PATTERN.SIZE_MM)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # gather points
    objpoints = []
    imgpoints = []

    vidcap = cv.VideoCapture(video_path)
    if not vidcap.isOpened():
        logger.error(f"Failed to read input: {video_path}. Abort calibration")

    while(vidcap.isOpened()):
        ret, frame = vidcap.read(cv.IMREAD_COLOR)
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # find chessboard corners
        corners_found, corners = cv.findChessboardCorners(image=gray, patternSize=pattern_size,
                                                          flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)

        # impove corners coordinates
        if corners_found:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            objpoints.append(world_pts)

        cv.drawChessboardCorners(frame, pattern_size, corners, corners_found)
        cv.imshow("frame", frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    logger.info(f"Gathering finished. Total patterns detected: {len(imgpoints)}")

    # calibrate
    logger.info("Start calibration...")
    calib_flags = cv.CALIB_ZERO_TANGENT_DIST + cv.CALIB_FIX_K3
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=calib_flags)

    aov_x = 2.0 * np.arctan((gray.shape[1] / 2) / mtx[0, 0])
    aov_y = 2.0 * np.arctan((gray.shape[0] / 2) / mtx[1, 1])

    # show results
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    logger.info(f"Calibration results:\n\n---RMSE---\n{ret:0.6f}" \
                f"\n\n---CamMat---\n{mtx}" \
                f"\n\n---DistCoeff---\n{dist}" \
                f"\n\n---ViewAngle---\nhorizontal: {np.rad2deg(aov_x):0.3f}\nvertical: {np.rad2deg(aov_y):0.3f}")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Camera Calibration With OpenCV And Chessboard')
    parser.add_argument("-i", "--input", dest="input", required=False, type=str,
                        default="data/test.mp4",
                        help='Path to source video to calibrate')
    parser.add_argument("-c", "--config-file", dest="config_file", required=False, type=str, default="configs/cfg.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # read config
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # create logger
    logger = setup_logger("CORE", 0)
    logger.info(args)
    logger.info("Loaded configuration file {}:\n{}".format(args.config_file, cfg))

    # process file
    calibrate(args.input)


if __name__ == "__main__":
    main()
