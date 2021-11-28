from numpy import zeros as npzeros


class SyntheticImage:
    def get_im(self):
        pass
    
    def add_x_mask(self, ax):
        pass
        
    def add_y_mask(self, ax):
        pass
    
    def add_mask(self, ax):
        pass


class CrossImage(SyntheticImage):
    def __init__(self, width=10, height=10, hline=2, vline=6):
        self.width = width
        self.height = height
        self.hline = hline
        self.vline = vline
        self.im = npzeros((self.height, self.width))
        self.im[self.hline,:] = 255
        self.im[:,self.vline] = 255
        
    def get_im(self):
        return self.im
    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_hline(self):
        return self.hline
    
    def get_vline(self):
        return self.vline
    
    def add_x_mask(self, ax):
        ax.axhline(y = self.hline, xmin = 0, xmax = self.width, color ='red')
        
    def add_y_mask(self, ax):
        ax.axvline(x = self.vline, ymin = 0, ymax = self.height, color ='red')
        
    def add_mask(self, ax):
        self.add_x_mask(ax)
        self.add_y_mask(ax)