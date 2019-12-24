
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

img = load_img('girl.jpg')
img = img_to_array(img)
data = expand_dims(img, 0)

# Dinh nghia 1 doi tuong Data Generator voi bien phap chinh sua anh Zoom tu 0.5x den 2x
myImageGen = ImageDataGenerator(zoom_range=[0.5,2.0])
# Batch_Size= 1 -> Moi lan sinh ra 1 anh
gen = myImageGen.flow(data, batch_size=1)
# Sinh ra 9 anh va hien thi len man hinh
for i in range(9):

	plt.subplot(330 + 1 + i)
	myBatch = gen.next()
	image = myBatch[0].astype('uint8')
	plt.imshow(image)

plt.show()