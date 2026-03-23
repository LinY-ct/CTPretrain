

class CTTools:
    def __init__(self, mu_water=0.0192):
        self.mu_water = mu_water
    def HU2mu(self, hu_img):
        mu_img = hu_img / 1000 * self.mu_water + self.mu_water
        return mu_img

    def mu2HU(self, mu_img):
        hu_img = (mu_img - self.mu_water) / self.mu_water * 1000
        return hu_img

    def window_transform(self, hu_img, width=3000, center=500, norm=False):
        # HU -> 0-1 normalized
        min_window = float(center) - 0.5 * float(width)  # -1000
        win_img = (hu_img - min_window) / float(width)  #
        win_img[win_img < 0] = 0
        win_img[win_img > 1] = 1
        if norm:
            print('normalize to 0-255')
            win_img = (win_img * 255).astype('float')
        return win_img

    def back_window_transform(self, win_img, width=3000, center=500, norm=False):
        # 0-1 normalized -> HU
        min_window = float(center) - 0.5 * float(width)
        win_img = win_img / 255 if norm else win_img
        hu_img = win_img * float(width) + min_window
        return hu_img