"""
Ekstraksi frame dari video untuk OCR subtitle (1 frame/detik, crop 1/4 bawah).
Mata kuliah NLP Berbasis Transformer, Itera.
"""
import os

import cv2

if __name__ == '__main__':
    video_path = './data/video.mp4'
    save_path = 'frame'
    os.makedirs(save_path, exist_ok=True)

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise FileNotFoundError(
            f'Tidak dapat membuka video: {video_path}. '
            'Letakkan file di ./data/video.mp4'
        )

    success, frame = video_capture.read()
    total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 25
    print(f'Total frame: {total_frame}, FPS: {fps}')

    frame_idx, saved = 0, 0
    while success:
        if frame_idx % fps == 0:
            saved += 1
            h, _, _ = frame.shape
            # Hanya bagian bawah 1/4 layar (area subtitle umum)
            frame_crop = frame[(h // 4) * 3:h, :, :]
            image_save_path = os.path.join(
                save_path, f'image_{saved:03d}.jpg'
            )
            cv2.imwrite(image_save_path, frame_crop)
            print('Frame disimpan:', frame_idx, '→', image_save_path)
        success, frame = video_capture.read()
        frame_idx += 1

    video_capture.release()
    print(f'Selesai. {saved} gambar di folder "{save_path}/".')
