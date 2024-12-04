import cv2
import numpy as np
import skimage.io
import albumentations as A

def resize(image, size=(128, 128)):
    return cv2.resize(image, size)


def add_gaussian_noise(x, sigma):
    noise = np.random.randn(*x.shape) * sigma
    x = np.clip(x + noise, 0., 1.)
    return x

def _evaluate_ratio(ratio):
    if ratio <= 0.:
        return False
    return np.random.uniform() < ratio

def apply_aug(aug, image, mask=None):
    augment = aug(image=image,mask=mask)
    return augment['image'],augment['mask']

class Transform:
    def __init__(self,  size=None, train=True,
                 BrightContrast_ratio=0.,  noise_ratio=0.,
                 Rotate_ratio=0., Flip_ratio=0.):
        self.size = size
        self.train = train
        self.BrightContrast_ratio = BrightContrast_ratio
        self.noise_ratio = noise_ratio
        self.Rotate_ratio = Rotate_ratio
        self.Flip_ratio = Flip_ratio

    def __call__(self, example):
        if self.train:
            x, y = example
        else:
            x = example
        # --- Augmentation ---
        # --- Train/Test common preprocessing ---

        if self.size is not None:
            x = resize(x, size=self.size)

        if _evaluate_ratio(self.BrightContrast_ratio):
            x, _ = apply_aug(A.RandomBrightnessContrast(p=1.0), x)

        if _evaluate_ratio(self.noise_ratio):
            x = add_gaussian_noise(x, sigma=5. / 255.)

        if _evaluate_ratio(self.Rotate_ratio):
            x, y = apply_aug(A.Rotate(p=1.0), x, y)

        if _evaluate_ratio(self.Flip_ratio):
            x, y = apply_aug(A.Flip(p=1.0), x, y)

        if self.train:
            return x, y
        else:
            return x

      
if __name__ == '__main__':
      import matplotlib.pyplot as plt
      from tqdm import tqdm 
      f, ax = plt.subplots(3,3, figsize=(16,18))
      img = skimage.io.imread('/home/ubuntu/Pictures/timg.jpeg')
      transform = Transform(affine=False,train=False,Flip_ratio=0.8)
      for i in tqdm(range(9)):
            aug_img = transform(img)
            ax[i//3,i%3].imshow(aug_img)
      plt.show()
      