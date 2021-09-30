import numpy as np
import torch

import pytesseract
from fuzzywuzzy import fuzz
from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage, Resize

# ----- Utilities -----

def _normalize(img):
    """
    Linear histogram normalization

    arr.min -> 0 (darkest spot becomes black)
    arr[:, :50.min()] -> 255 (grey background becomes white)
    """
    arr = np.array(img, dtype=float)
    thresh = arr.mean()
    mask = np.ma.masked_where(arr >= thresh, arr).mask

    done = False
    # for k in range(5,50):
    #     if arr[:, :k].min() != 0 and done == False:
    #         print(arr.shape, arr.min(), arr[:, :k].min())
    #         arr = (arr - arr.min()) * (255 / arr[:, :k].min(where=))
    #         done = True
    # if done == False:
    arr = (arr - arr.min()) * (255 / max(arr[:, :50].min(), 50))
    arr[arr > 255] = 255
    arr[arr < 0] = 0

    return Image.fromarray(arr.astype('uint8'), 'L')


def OCR_score(X, X_ref, sample_ratio=100, normalize=_normalize):
    """ Compute average OCR-score of an image or batch of images as used in the HDC21.

       Parameters
       ----------
       X : torch.Tensor
           The input tensor of shape [..., 1, W, H]
       X_ref : torch.Tensor
           The reference tensor containing the text target. of dimension [..., 3]
       sample_ratio : integer
            integer between 0 and 100, to indicate how many samples should be evaluated on the given batch.
            For Speed Up purposes, since OCR_score will be slow. For now, the first few samples will be chosen,
            since randomization is achieved beforehand in HDCDataclass.
       Returns
       -------
       score_av :
           The average score.
       scores :
           List with individual scores.

       """

    target_batch = np.swapaxes(np.array(X_ref), 0, 1)
    assert X.ndim >= 3  # do not forget the channel dimension

    assert (target_batch.ndim == X.ndim -2)
    if X.ndim > 3:
        assert X.shape[0] == target_batch.shape[0]  # do the batches have the same size?
        batch_size = X.shape[0]
    else:
        batch_size = 1
        img_batch = torch.unsqueeze(X, 0)

    batch_size = batch_size * sample_ratio // 100
    options = r'--oem 1 --psm 6 -c load_system_dawg=false -c load_freq_dawg=false  -c textord_old_xheight=0  -c textord_min_xheight=100 -c ' \
              r'preserve_interword_spaces=0'
    scores = []

    for i in range(batch_size):  # iterate through samples of the batch -> Speed up possible?
        # print(X[i].min(), X[i].max(), X.dtype)

        trueText = list(target_batch[i])

        img = ToPILImage()(X[i].squeeze().cpu().type(torch.FloatTensor))

        # plt.imshow(img, cmap='gray')
        # plt.show()
        # print(img.mode, img.getextrema())
        img = Resize((1460, 2360))(img)

        img = ImageOps.grayscale(img)
        w, h = img.size

        # plt.imshow(img, cmap='gray')
        # plt.show()
        # print(img.mode, img.getextrema())

        img = normalize(img)
        # # resize image to improve OCR

        # img = normalize(img)
        # plt.imshow(img, cmap='gray')
        # plt.show()
        # print(img.mode, img.getextrema())

        img = img.resize((int(w / 2), int(h / 2)))
        w, h = img.size

        # plt.imshow(img, cmap='gray')
        # plt.show()
        # print(img.mode, img.getextrema())

        # plt.imshow(img, cmap='gray')
        # plt.show()
        # run OCR

        OCRtext = pytesseract.image_to_string(img, config=options)

        # removes form feed character  \f
        OCRtext = OCRtext.replace('\n\f', '').replace('\n\n', '\n')

        # split lines
        OCRtext = OCRtext.split('\n')

        # remove empty lines
        OCRtext = [x.strip() for x in OCRtext if x.strip()]

        if len(OCRtext) != 3:
            OCRtext.append('')
            # print('ERROR: OCR text of sample {} does not have 3 lines of text!'.format(str(i)))
            # print(OCRtext)
            score = float(fuzz.ratio(trueText[1], OCRtext[0]))  # to not break the evaluation pipeline
        else:
            score = float(fuzz.ratio(trueText[1], OCRtext[1]))
        scores.append(score)
    score_avg = sum(scores) / batch_size
    return score_avg, scores, OCRtext



def l2_error(X, X_ref, relative=False, squared=False, use_magnitude=True):
    """ Compute average l2-error of an image over last three dimensions.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor of shape [..., 1, W, H] for real images or
        [..., 2, W, H] for complex images.
    X_ref : torch.Tensor
        The reference tensor of same shape.
    relative : bool, optional
        Use relative error. (Default False)
    squared : bool, optional
        Use squared error. (Default False)
    use_magnitude : bool, optional
        Use complex magnitudes. (Default True)

    Returns
    -------
    err_av :
        The average error.
    err :
        Tensor with individual errors.

    """
    assert X_ref.ndim >= 3  # do not forget the channel dimension

    if X_ref.shape[-3] == 2 and use_magnitude:  # compare complex magnitudes
        X_flat = torch.flatten(torch.sqrt(X.pow(2).sum(-3)), -2, -1)
        X_ref_flat = torch.flatten(torch.sqrt(X_ref.pow(2).sum(-3)), -2, -1)
    else:
        X_flat = torch.flatten(X, -3, -1)
        X_ref_flat = torch.flatten(X_ref, -3, -1)

    if squared:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1) ** 2
    else:
        err = (X_flat - X_ref_flat).norm(p=2, dim=-1)

    if relative:
        if squared:
            err = err / (X_ref_flat.norm(p=2, dim=-1) ** 2)
        else:
            err = err / X_ref_flat.norm(p=2, dim=-1)

    if X_ref.ndim > 3:
        err_av = err.sum() / np.prod(X_ref.shape[:-3])
    else:
        err_av = err
    return err_av.squeeze(), err
