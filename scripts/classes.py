import numpy as np
import PIL
from util import find_nearest

class SingleResSpectrogram(object):
    def __init__(self, spec, x_axis, y_axis, parent=None):
        self.spec   = np.array(spec, dtype=object)
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.parent = parent

class MultiResSpectrogram(object):
    def __init__(self, base_spec):
        self.base_spec = base_spec # é um SingleResSpectrogram, já tem spec e eixos
        self.first_zoom = np.array([], dtype=object)  # estruturas auxiliares de indexação rápida. discutir depois
        self.second_zoom = np.array([], dtype=object)
        
    def insert_zoom(self, base_spec, zoom_spec, zoom_level=1):
        x_start = find_nearest(base_spec.x_axis, zoom_spec.x_axis[0])
        x_end   = find_nearest(base_spec.x_axis, zoom_spec.x_axis[-1])
        y_start = find_nearest(base_spec.y_axis, zoom_spec.y_axis[0])
        y_end   = find_nearest(base_spec.y_axis, zoom_spec.y_axis[-1])
        
        norm_ref = np.max(base_spec.spec[y_start:y_end, x_start:x_end])
        zoom_spec.spec = zoom_spec.spec * (norm_ref/np.max(zoom_spec.spec))

        base_spec.spec[y_start:y_end, x_start:x_end] = zoom_spec
        zoom_spec.parent = base_spec
        
        if zoom_level == 1:
            self.first_zoom = np.append(self.first_zoom, zoom_spec)
        elif zoom_level == 2:
            self.second_zoom = np.append(self.second_zoom, zoom_spec)
               
    def generate_visualization(self):
        # bolar algum jeito de percorrer árvore debaixo p cima
        # por enquanto, vamos "roubar":
        spec = MultiResSpectrogram.convert_to_visualization(self.base_spec.spec)
        spec_img = PIL.Image.fromarray(spec).resize((spec.shape[1], 2049))
        
        for zoom_spec in self.first_zoom:
            spec_img = MultiResSpectrogram.insert_visualization(self, spec_img, zoom_spec)

        return np.asarray(spec_img)
        
    def insert_visualization(self, spec_img, zoom_spec):
        y_axis = np.linspace(0, 22050, 2049)

        x_start = find_nearest(self.base_spec.x_axis, zoom_spec.x_axis[0])
        x_end   = find_nearest(self.base_spec.x_axis, zoom_spec.x_axis[-1])
        y_start = find_nearest(y_axis, zoom_spec.y_axis[0])-1 # "discontinuidade" na fronteira do kernel
        y_end   = find_nearest(y_axis, zoom_spec.y_axis[-1])
        
        zoom_img = PIL.Image.fromarray(MultiResSpectrogram.convert_to_visualization(zoom_spec.spec)).resize((x_end - x_start,y_end - y_start))
        box = (x_start, y_start, x_end, y_end)
        spec_img.paste(zoom_img, box)

        return spec_img
    
    def check_type(x):
        if isinstance(x, SingleResSpectrogram):
            return 0
        else:
            return x

    check_type_vec = np.vectorize(check_type)

    def convert_to_visualization(spec):
        return MultiResSpectrogram.check_type_vec(spec).astype(np.float32)