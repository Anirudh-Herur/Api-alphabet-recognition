import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# fettching the data
X = np.load('image.npz')['arr_0']
Y = pd.read_csv('labels.csv')['labels']
print(pd.Series(Y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=500, train_size=3500, random_state=9)


def get_prediction(image):
    im_pill = Image.fromarray(image)
    Image_bw = im_pill.convert('L')
    Image_bw_resize = Image_bw.resize((22, 30), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(Image_bw_resize_inverted, pixel_filter)
    Image_bw_resize_inverted_scaled = np.clip(
        Image_bw_resize_inverted - min_pixel, 0, 255)
    max_pixel = np.max(Image_bw_resize)
    Image_bw_resize_inverted_scaled = np.asarray(
        Image_bw_resize_inverted_scaled) / max_pixel
    test_sample = np.array([Image_bw_resize_inverted_scaled]).reshape(1, 660)
    test_predict = clf.predict(test_sample)
    return test_predict[0]
