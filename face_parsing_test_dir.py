import os
import cv2
import time
import numpy as np
import torch
import glob
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
from ibug.face_parsing.utils import label_colormap
from ibug.face_detection.utils import HeadPoseEstimator


def count_hist(arr):
    histo = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            histo[arr[i,j]] += 1
    result_str = "occluded: "
    if histo[4] == 0:
        result_str += "left eye "
    if histo[5] == 0:
        result_str += "right eye "
    if histo[6] == 0:
        result_str += "nose "
    if histo[7] == 0:
        result_str += "upper lip "
    if histo[9] == 0:
        result_str += "lower lip "
    return histo, result_str

def main() -> None:
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument(
        '--input', '-i', help='Input video path or webcam index (default=0)', default=0)
    parser.add_argument(
        '--output', '-o', help='Output file path', default=None)
    parser.add_argument('--fourcc', '-f', help='FourCC of the output video (default=mp4v)',
                        type=str, default='mp4v')
    parser.add_argument('--benchmark', '-b', help='Enable benchmark mode for CUDNN',
                        action='store_true', default=False)
    parser.add_argument('--no-display', help='No display if processing a video file',
                        action='store_true', default=False)
    parser.add_argument('--threshold', '-t', help='Detection threshold (default=0.8)',
                        type=float, default=0.8)
    parser.add_argument('--encoder', '-e', help='Method to use, can be either rtnet50 or rtnet101 (default=rtnet50)',
                        default='rtnet50') # choices=['rtnet50', 'rtnet101', 'resnet50'])

    parser.add_argument('--decoder', help='Method to use, can be either rtnet50 or rtnet101 (default=rtnet50)',
                        default='fcn', choices=['fcn', 'deeplabv3plus'])
    parser.add_argument('-n', '--num-classes', help='Face parsing classes (default=11)', type=int, default=11)
    parser.add_argument('--max-num-faces', help='Max number of faces',
                        default=50)
    parser.add_argument('--weights', '-w',
                        help='Weights to load, can be either resnet50 or mobilenet0.25 when using RetinaFace',
                        default=None)
    parser.add_argument('--device', '-d', help='Device to be used by the model (default=cuda:0)',
                        default='cuda:0')
    parser.add_argument('--head-pose-preference', '-hp',
                        help='Head pose output preference (default=0)',
                        type=int, default=0)
    args = parser.parse_args()

    # Set benchmark mode flag for CUDNN
    torch.backends.cudnn.benchmark = args.benchmark
    # args.method = args.method.lower().strip()
    vid = None
    out_vid = None
    has_window = False
    face_detector = RetinaFacePredictor(threshold=args.threshold, device=args.device,
                                        model=(RetinaFacePredictor.get_model('mobilenet0.25')))
    head_pose_estimator = HeadPoseEstimator()
    face_parser = RTNetPredictor(
        device=args.device, ckpt=args.weights, encoder=args.encoder, decoder=args.decoder, num_classes=args.num_classes)

    colormap = label_colormap(args.num_classes)
    print('Face detector created using RetinaFace.')
    png_list = os.listdir("./data/01000/")
    # try:
    #     # Open the input video
    #     vid = cv2.VideoCapture()
    #     print(vid)
    #     assert vid.isOpened()
    
    #     print(f'Input video "{args.input}" opened.')

    #     # Process the frames
    #     frame_number = 0
    #     window_title = os.path.splitext(os.path.basename(__file__))[0]
    #     print('Processing started, press \'Q\' to quit.')
    
    files = os.listdir(args.input)
    file_cnt = 1
    total_start_time = time.time()
    for file in files:
        # print(file_cnt, '/', len(files))
        file_cnt += 1
        # Get a new frame
        window_title = os.path.splitext(os.path.basename(__file__))[0]
        alphas = np.linspace(0.75, 0.25, num=args.max_num_faces)

        frame = cv2.imread(args.input+'/'+file)
        gray = frame.copy()
        if frame is None:
            break
        else:
            # Detect faces
            start_time = time.time()
            faces = face_detector(frame, rgb=False)
            elapsed_time = time.time() - start_time

            # Textural output
            # print(f'Frame #{0} processed in {elapsed_time * 1000.0:.04f} ms: ' +
            #         f'{len(faces)} faces detected.')

            if len(faces) == 0:
                if using_webcam or not args.no_display:
                    has_window = True
                    cv2.imshow(window_title, frame)
                    key = cv2.waitKey(1) % 2 ** 16
                    if key == ord('q') or key == ord('Q'):
                        print('\'Q\' pressed, we are done here.')
                        break
                # frame_number += 1
                continue
            # Parse faces
            start_time = time.time()
            masks = face_parser.predict_img(frame, faces, rgb=False)
            elapsed_time = time.time() - start_time
            # if faces.shape[1] >= 15:
            #     head_poses = [head_pose_estimator(face[5:15].reshape((-1, 2)), *frame.shape[1::-1],
            #                                     output_preference=args.head_pose_preference)
            #                 for face in faces]
            # else:
            #     head_poses = [None] * faces.shape[0]
            # Textural output
            # print(f'Frame #{0} processed in {elapsed_time * 1000.0:.04f} ms: ' +
            #         f'{len(masks)} faces parsed.')

            # # Rendering
            
            dst = frame
            for i, (face, mask, head_pose) in enumerate(zip(faces, masks, head_poses)):
                bbox = face[:4].astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(
                    0, 0, 255), thickness=2)
                alpha = alphas[i]
                index = mask > 0
                res = colormap[mask]
                dst[index] = (1 - alpha) * frame[index].astype(float) + \
                    alpha * res[index].astype(float)
                # if head_pose is not None:
                #     pitch, yaw, roll = head_pose
                #     cv2.putText(frame, f'Pitch: {pitch:.1f}', (bbox[2] + 5, bbox[1] + 10),
                #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
                #     cv2.putText(frame, f'Yaw: {yaw:.1f}', (bbox[2] + 5, bbox[1] + 30),
                #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
                #     cv2.putText(frame, f'Roll: {roll:.1f}', (bbox[2] + 5, bbox[1] + 50),
                #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
                hist = count_hist(mask)
                
                # print(mask)
                
            dst = np.clip(dst.round(), 0, 255).astype(np.uint8)
            frame = dst
            # Write the frame to output video (if recording)
            if out_vid is not None:
                out_vid.write(frame)
            
            # Display the frame
            if not args.no_display:
                frame = cv2.resize(frame, dsize=(0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                gray = cv2.resize(gray, dsize=(0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                has_window = True
                cv2.putText(gray, hist[1], (bbox[2] + 5, bbox[1] + 10),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
                cat = cv2.hconcat([frame, gray])
                cv2.imshow(window_title, cat)
                # cv2.imshow(window_title, frame)
                # cv2.imshow(window_title+"org", gray)
                if hist[1] == "":
                    key = cv2.waitKey(100000) % 2 ** 16
                else:
                    print(hist[1])
                    key = cv2.waitKey(100000) % 2 ** 16
                if key == ord('q') or key == ord('Q'):
                    print('\'Q\' pressed, we are done here.')
                    break
            else:
                if len(hist[1]) != 10:
                    cv2.imwrite('./data/occluded/'+file, gray)
                else:
                    cv2.imwrite('./data/occluded_X/'+file, gray)
            # frame_number += 1
    total_elapsed_time = time.time() - total_start_time
    print(f'Frames processed in {total_elapsed_time * 1000.0:.04f} ms: ' +
                    f'{len(files)} faces detected and parsed.')
    if has_window:
        cv2.destroyAllWindows()
    if out_vid is not None:
        out_vid.release()
    if vid is not None:
        vid.release()
    print('All done.')


if __name__ == '__main__':
    main()
