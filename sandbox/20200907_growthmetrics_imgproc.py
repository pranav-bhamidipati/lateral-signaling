@numba.njit
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def get_frames(vid, skip, start=0, progress_bar=False):
    """
    Returns sampled frames from a video as a list of Numpy arrays.
    Returns the frame numbers of sampled frames as a Numpy array.
    """
    cap = cv2.VideoCapture(vid)

    nframes = int(cap.get(7))
    frames = []

    iterator = range(nframes)
    if progress_bar:
        iterator = tqdm(iterator)

    for i in iterator:

        cap.grab()

        if ((i - start) % skip) or (i < start):
            continue

        _, frame = cap.retrieve()
        frames.append(frame)

    cap.release()

    return frames, np.arange(start, nframes, skip)


def clean_im(im, print_updates=False, as_float=False):
    """Returns a pre-processed image."""

    # Read image if path is provided
    if type(im) is str:
        im = cv2.imread(im)

    if print_updates:
        print(f"Cleaning image.")

    # Separate channels
    im[:, :, 1] = cv2.subtract(im[:, :, 1], im[:, :, 0])
    im[:, :, 2] = cv2.subtract(im[:, :, 2], im[:, :, 0])

    # Cover timestamp with median mask (BG) or zero (R)
    mask = np.s_[1300:1380, 1400:1600]
    im[mask] = np.array([np.median(im[:, :, 0]), np.median(im[:, :, 1]), 0])

    #     # Construct green background with median filter
    #     green_bg = cv2.medianBlur(im[:, :, 1], 251, 0)

    #     # Subtract background
    #     im[:, :, 1] = cv2.subtract(im[:, :, 1], green_bg)

    # Remove residual signal in timestamp
    im[(*mask, 1)] = im[:, :, 1].min()

    if as_float:
        # Normalize luminance
        im = normalize(im)

    return im


def clean_frames(frames):
    return [clean_im(frame) for frame in frames]


def find_brightest(im, ksize=(201, 201), sigmaX=25, sigmaY=0, winsize=0):
    """
    Returns the index of the brightest pixel in a single-channel image,
    after applying a Gaussian blur. If winsize>0, returns a slice object
    for a square window around that pixel of shape (2*winsize + 1) x (2*winsize + 1) 
    """

    blur = cv2.GaussianBlur(im, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY)
    loc = np.unravel_index(blur.argmax(), blur.shape)

    if winsize:
        loc = np.s_[
            (loc[0] - winsize) : (loc[0] + winsize + 1),
            (loc[1] - winsize) : (loc[1] + winsize + 1),
        ]

    return loc


def get_activated_mask(im, thresh=10, ksize=51, iters=1):
    """
    Returns a mask of activated cells in a single-channel image.
    Performs a thresholding operation, then a closing operation.
    
    im : Image as 2D Numpy array
    thresh : threshold to generate mask
    ksize, iters : parameters for closing operation. Passed to 
                cv2.morphologyEx
    """
    ret, mask = cv2.threshold(im[:, :, 1], thresh, 255, 0)
    assert ret, "Thresholding failed."

    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)


@numba.njit
def rms_mask(m, loc, ip):
    """
    Returns RMS distance of pixels in a mask to location loc, 
    in units of interpixel distance ip.
    
    m  : mask, as a Numpy array of indices of shape (ndim, npix)
    """

    ndim, npix = m.shape

    # Catch empty masks
    if npix == 0:
        return 0

    # calculate squared distance
    sqd = np.zeros(npix)
    for i in range(ndim):
        sqd += (m[i] - loc[i]) ** 2

    # Root-mean of squared distance (RMSD) in units of distance
    return np.sqrt(np.mean(sqd)) * ip


def chull_mask(m, ip):
    """
    Returns area of the convex hull of pixels in a mask, in units
    of squared distance
    
    m  : mask as a 2D Numpy array of shape (ndim, npix)
    ip : inter-pixel distance
    """

    # If not enough points, return 0
    ndim, npix = m.shape
    if npix < 3:
        return 0

    return spat.ConvexHull(ip * m.T).volume


@numba.njit
def logistic(x, a, b, N):
    return N / (1 + a * np.exp(-b * x))