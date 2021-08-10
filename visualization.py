from tensorflow.keras.models import load_model

model = load_model('output/model_vis_test.hdf5')
model.summary()

img_path = 'C:/Users/AmbruszkiewM/PycharmProjects/klasyfikator_tf/images/test/surfboard/41.devil.jpg'

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
print(img_tensor.shape) #kształt tensora (1,150,150,3)


#plt.imshow(img_tensor[0])
#plt.show()


layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 1], cmap='viridis')
plt.show()
plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis')
plt.show()
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()
plt.matshow(first_layer_activation[0, :, :, 5], cmap='viridis')
plt.show()
plt.matshow(first_layer_activation[0, :, :, 6], cmap='viridis')
plt.show()
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.show()
plt.matshow(first_layer_activation[0, :, :, 8], cmap='viridis')
plt.show()









