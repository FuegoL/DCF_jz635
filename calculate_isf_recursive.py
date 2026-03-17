import os

import numpy as np
import cv2
from tqdm.auto import tqdm

# --- ISF calculation logic from ViscoelasiticityV3_batch.py ---
class ImageStack(object):
    def __init__(self, filename, channel=None):
        self.filename = filename
        self.video = cv2.VideoCapture(filename)
        property_id = int(cv2.CAP_PROP_FRAME_COUNT)
        length = int(self.video.get(property_id))
        self.frame_count = length
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.channel = channel
        self.shape = self[0].shape

    def __len__(self):
        return self.frame_count

    def __getitem__(self, t):
        if t < 0:
            t = len(self) + t - 1
        assert t < self.frame_count
        self.video.set(cv2.CAP_PROP_POS_FRAMES, t - 1)
        success, image = self.video.read()
        if self.channel is not None:
            return image[..., self.channel]
        if image is not None:
            return image.mean(axis=2).astype(int)
        self.shape = self[0].shape

def spectrumDiff(im0, im1):
    return np.abs(np.fft.fft2(im1 - im0.astype(float))) ** 2

def timeAveraged(stack, dt, maxNCouples=20):
    increment = max([(len(stack) - dt) / maxNCouples, 1])
    initialTimes = np.arange(0, len(stack) - dt, increment)
    avgFFT = np.zeros(stack.shape)
    failed = 0
    for t in initialTimes:
        im0 = stack[int(t)]
        im1 = stack[int(t + dt)]
        if im0 is None or im1 is None:
            failed += 1
            continue
        avgFFT += spectrumDiff(im0, im1)
    return avgFFT / (len(initialTimes) - failed)

class RadialAverager(object):
    def __init__(self, shape):
        assert len(shape) == 2
        self.dists = np.sqrt(np.fft.fftfreq(shape[0])[:, None] ** 2 + np.fft.fftfreq(shape[1])[None, :] ** 2)
        self.dists[0] = 0
        self.dists[:, 0] = 0
        self.bins = np.arange(max(shape) / 2 + 1) / float(max(shape))
        self.hd = np.histogram(self.dists, self.bins)[0]

    def __call__(self, im):
        assert im.shape == self.dists.shape
        hw = np.histogram(self.dists, self.bins, weights=im)[0]
        return hw / self.hd

def logSpaced(L, pointsPerDecade=15):
    nbdecades = np.log10(L)
    return np.unique(np.logspace(
        start=0, stop=nbdecades,
        num=int(nbdecades * pointsPerDecade),
        base=10, endpoint=False
    ).astype(int))

def calculate_isf(stack, idts, maxNCouples=1000):
    ra = RadialAverager(stack.shape)
    isf = np.zeros((len(idts), len(ra.hd)))
    for i, idt in enumerate(tqdm(idts, desc="ISF calculation", unit="interval")):
        isf[i] = ra(timeAveraged(stack, idt, maxNCouples))
    return isf

def process_avi(video_file, pixelSize=0.24, pointsPerDecade=100, maxNCouples=10):
    stack = ImageStack(video_file)
    idts = logSpaced(len(stack), pointsPerDecade)
    dts = idts / stack.fps
    ISF = calculate_isf(stack, idts, maxNCouples)
    qs = 2 * np.pi / (2 * ISF.shape[-1] * pixelSize) * np.arange(ISF.shape[-1])
    return {
        'ISF': ISF,
        'qs': qs,
        'dts': dts,
        'idts': idts,
        'stack': stack,
        'pixelSize': pixelSize,
        'fps': stack.fps,
        'video_file': video_file
    }

def find_avi_files(root_dir):
    avi_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.avi'):
                avi_files.append(os.path.join(dirpath, f))
    return avi_files

def main():
    root_dir = r"C:\tmp"
    avi_files = find_avi_files(root_dir)
    print(f"Found {len(avi_files)} .avi files.")
    for video_path in tqdm(avi_files, desc="Calculating ISF", unit="video"):
        try:
            result = process_avi(video_path)
            out_path = os.path.splitext(video_path)[0] + "_ISF.npz"
            np.savez(out_path, ISF=result['ISF'], qs=result['qs'], dts=result['dts'], fps=result['fps'], pixelSize=result['pixelSize'], video_file=result['video_file'])
        except Exception as e:
            print(f"Failed to process {video_path}: {e}")

if __name__ == "__main__":
    main()
