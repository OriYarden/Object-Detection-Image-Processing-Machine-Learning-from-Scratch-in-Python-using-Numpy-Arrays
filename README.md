# Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays
Detecting objects from a set of training images by shape and color using machine learning in Python from scratch (doing all the math on only numpy arrays, no machine learning packages used).


![Picture1](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/6ec38814-ec3e-4090-ae70-d6b797d7e55d)

In the example shown in the above figure, a 16 pixel image with red, blue, and green color channels in the third dimension.

We can flatten the image rows-columns wise and make a weights matrix for image processing and machine learning to train the weights to detect and track objects:

![Picture2](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/5c4f1eff-635a-43ff-994b-9d1c95d8974d)


in Python by generating a numpy two-dimensional array of random numbers in which the image rows and columns are flattened into the first dimension with rgb colors for the second dimension in the weights matrix:

    def init_weights(self):
        return np.reshape(self.add_noise(self.rgb_dim*self.image_size**2), [self.image_size**2, self.rgb_dim])


Training the neural network weights on the following images of different object shapes and colors:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/992fd9a0-3e67-4dbf-aed7-0e3ab622cbfc)


by iterating through our list of training images and adjusting our weights matrix by the difference between each image and the product of the image and weights passed through an activation function (and we add a bit of noise into the input layer for science):


    def train(self, iterations, new_weights=False, learning_rate=1.0):
        if new_weights:
            self.weights = self.init_weights()

        for _ in range(iterations):
            for image_num, image in self.training_images.items():
                input_layer = np.reshape(image, [self.image_size**2, self.rgb_dim]) + np.reshape(self.add_noise(self.rgb_dim*self.image_size**2)*0.01, [self.image_size**2, self.rgb_dim])
                output_layer = self.activation_function(self.weights*input_layer)
                _error = np.reshape(image, [self.image_size**2, self.rgb_dim]) - output_layer

                weights_feedback = self.activation_function((self.weights*_error*output_layer*(1.0 - output_layer))*input_layer)
                self.weights += weights_feedback*learning_rate


Here are the weights before training which are initially random:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/efd6c12b-1123-4e02-bded-50f835d5884f)


After training our neural network for 100 iterations per image for each of the eight training images:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/2feef399-9eef-4441-8acd-ce277abe3f94)


We can see that the weights matrix learned the eight images, and now we can test our neural network to find and detect one of the objects it was trained on in the real world via grid search and match-to-sample:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/90cecc4b-14c6-4427-9f11-f77683b62239)

multiplying the sample of the world by the weights matrix and finding the maximum (or minimum) difference after subtracting that from the sample and from the training images:


    def test(self, world_size=3, new_weights=False):
        world = self.create_world(world_size)
        if new_weights:
            self.weights = self.init_weights()

        errors = np.zeros((world.shape[0] - self.image_size, world.shape[1] - self.image_size, len(self.training_images))).astype(float)
        rgb_errors = np.zeros((world.shape[0] - self.image_size, world.shape[1] - self.image_size, len(self.training_images))).astype(float)

        for row in range(world.shape[0] - self.image_size):
            for col in range(world.shape[1] - self.image_size):
                for image_num, image in self.training_images.items():
                    sample_of_world = np.reshape(world[row:row + self.image_size, col:col + self.image_size, :], [self.image_size**2, self.rgb_dim])
                    match_image_to_sample = self.weights*sample_of_world
                    _error = np.reshape(sample_of_world - match_image_to_sample, [self.image_size, self.image_size, self.rgb_dim])
                    errors[row, col, int(image_num)] = np.sum(abs(_error)) if self.object_shapes is not None else np.sum(abs(_error*np.reshape(self.hidden_weights, [self.image_size, self.image_size, self.rgb_dim])))
                    rgb_errors[row, col, int(image_num)] = np.sum([np.sum(np.sum(abs(image - np.reshape(match_image_to_sample, [self.image_size, self.image_size, self.rgb_dim])), axis=_axis)) for _axis in range(2)])

        found_image_row_col = [[np.where(errors[:, :, int(image_num)] == np.max(errors[:, :, int(image_num)]))[0][0], np.where(errors[:, :, int(image_num)] == np.max(errors[:, :, int(image_num)]))[1][0]] for image_num, image in self.training_images.items()][0]
        found_image_num = np.where(rgb_errors[found_image_row_col[0], found_image_row_col[1], :] == np.min(rgb_errors[found_image_row_col[0], found_image_row_col[1], :]))[0][0]


![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/f4cafbeb-f6af-4ddc-b978-577adc1f20aa)


By calculating the errors separately for the rgb color dimensions, the neural network weights matrix can discriminate objects of the same shape but differ in color, too:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/8692da38-0d85-4e3b-97c6-b9af05fc7386)



Though this neural network is fairly simple, potential applications for using machine learning models to train weights matrixes for image processing and object detection and tracking are wide ranging. For example, suppose we wanted to find and detect where an enemy military tank is located in an image fed directly from the battle field to inform nearby friendly soldiers:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/3d201afd-6db2-4f6c-9d12-17374ef6406e)


Here is our initially random weights matrix before training it on the training image:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/e96788d7-10e1-4ff8-a7e0-9104a95ad45c)

Here is our weights matrix after 1,000 training iterations:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/164735cb-4357-4a09-80fd-edc396581985)


Here is the real world test image where we expect our neural network to find and detect the tank and its exact location from the training image:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/59d861e7-414f-4955-a83f-0f24229eb319)


And here is the neural network detecting the location of the tank in the image:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/0b7ece2f-f07b-4e60-bedc-b6c1374761fb)

And this is what the sample of the world multiplied by our weights matrix looks like:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/800ab412-5f1e-4327-ad5a-542f52bda1d9)

Let's use a new neural network weights matrix and train it on a different tank image:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/04ec5348-b2f9-4215-9cc7-33a63d2a76a2)

Here is our initially random weights matrix before training it on the training image:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/8e118bbf-df2b-4c4f-8437-afd484dc9e8d)

Here are the weights after 1,000 training iterations:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/8aca6551-d678-4f62-9191-011deeb021c2)


And testing it on a real world image:

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/9fe3dfe3-438c-4f27-815b-aa623794daa7)

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/c4f2422c-64ba-44ab-9b77-72112a978b52)

![image](https://github.com/OriYarden/Object-Detection-Image-Processing-Machine-Learning-from-Scratch-in-Python-using-Numpy-Arrays/assets/137197657/4e672e6e-e624-4e47-9bf3-546d6db1b0e0)


Below I included the entire class and its methods:

    import numpy as np
    from matplotlib import pyplot as plt

    class NN:
        def __init__(self, object_shapes=None, object_colors=None, training_images=None):
            self.rgb_dim = 3
            self.object_shapes = object_shapes
            self.object_colors = object_colors
            if training_images is not None:
                self.training_images = self.use_images(training_images)
                self.image_size = self.training_images['0'].shape[0]
            else:
                self.image_size = 100
                self.training_images = self.create_images(object_shapes, object_colors)
            self.weights = self.init_weights()

        @staticmethod
        def add_noise(n=1):
            return np.random.random(n) if n != 1 else np.random.random(n)[0]

        @staticmethod
        def activation_function(x):
            return 1.0 / (1.0 + np.exp(-x))

        def normalize_rgb_values(self, rgb, factor=255.0):
            norm_rgb = (rgb - np.mean(rgb)) / np.var(rgb)**0.5
            norm_rgb += abs(np.min(norm_rgb))
            norm_rgb *= (factor / np.max(norm_rgb))
            return np.round(norm_rgb, decimals=0).astype(int) if factor == 255.0 else  np.round(norm_rgb, decimals=9).astype(float)

        def insert_object(self, _image, object_shape, object_color, row_col_position=[0, 0]):
            if object_shape == 'H':
                _image[30 + row_col_position[0]:75 + row_col_position[0], 60 + row_col_position[1]:70 + row_col_position[1], :] = object_color
                _image[50 + row_col_position[0]:55 + row_col_position[0], 40 + row_col_position[1]:60 + row_col_position[1], :] = object_color
                _image[30 + row_col_position[0]:75 + row_col_position[0], 30 + row_col_position[1]:40 + row_col_position[1], :] = object_color
            if object_shape == 'T':
                _image[30 + row_col_position[0]:38 + row_col_position[0], 30 + row_col_position[1]:70 + row_col_position[1], :] = object_color
                _image[38 + row_col_position[0]:75 + row_col_position[0], 45 + row_col_position[1]:55 + row_col_position[1], :] = object_color
            if object_shape == '|':
                _image[30 + row_col_position[0]:75 + row_col_position[0], 45 + row_col_position[1]:55 + row_col_position[1], :] = object_color
            if object_shape == '-':
                _image[50 + row_col_position[0]:55 + row_col_position[0], 40 + row_col_position[1]:60 + row_col_position[1], :] = object_color
            return _image

        def use_images(self, training_images):
            _images = {}
            for _image in training_images:
                image = np.array(_image).astype(np.uint8)[:, :, :3]
                _images[str(len(_images))] = self.normalize_rgb_values(image, factor=1.0)
            return _images

        def create_images(self, object_shapes, object_colors):
            _images = {}
            for _object_shape, _object_color in zip(object_shapes, object_colors):
                _image = np.zeros((self.image_size, self.image_size, self.rgb_dim)).astype(float)
                _image = self.insert_object(_image, _object_shape, _object_color)
                _images[str(len(_images))] = _image
            return _images

        def init_weights(self):
            _weights = np.reshape(self.add_noise(self.rgb_dim*self.image_size**2), [self.image_size**2, self.rgb_dim])
            self.weights_over_training_iterations = {'0': _weights}
            return _weights

        def train(self, iterations, new_weights=False, learning_rate=1.0):
            if new_weights:
                self.weights = self.init_weights()

            for _ in range(iterations):
                for image_num, image in self.training_images.items():
                    input_layer = np.reshape(image, [self.image_size**2, self.rgb_dim]) + np.reshape(self.add_noise(self.rgb_dim*self.image_size**2)*0.01, [self.image_size**2, self.rgb_dim])
                    output_layer = self.activation_function(self.weights*input_layer)
                    _error = np.reshape(image, [self.image_size**2, self.rgb_dim]) - output_layer

                    weights_feedback = self.activation_function((self.weights*_error*output_layer*(1.0 - output_layer))*input_layer)
                    self.weights += weights_feedback*learning_rate
                    self.weights_over_training_iterations[str(len(self.weights_over_training_iterations))] = self.weights.copy()
            self.hidden_weights = self.get_hidden_weights(self.weights)
            if self.object_shapes is None:
                self.weights = self.normalize_rgb_values(self.weights, factor=1.0)

        def get_hidden_weights(self, _weights):
            _hidden_weights = np.zeros((self.image_size**2, self.rgb_dim)).astype(float)
            _hidden_weights[np.where(_weights >= np.mean(_weights)), :] = 1.0
            return _hidden_weights

        def create_world(self, world_size):
            _world = np.zeros((int(self.image_size*world_size), int(self.image_size*world_size), self.rgb_dim)).astype(float)
            _random_row_col = [np.random.randint(self.image_size*world_size - self.image_size), np.random.randint(self.image_size*world_size - self.image_size)]
            _random_object = np.random.randint(len(self.object_shapes))
            _object_shape, _object_color = self.object_shapes[_random_object], self.object_colors[_random_object]
            return self.insert_object(_world, _object_shape, _object_color, _random_row_col)

        def test(self, test_image=None, outline_color=None, world_size=3, new_weights=False):
            if test_image is not None:
                world = self.normalize_rgb_values(np.array(test_image).astype(np.uint8)[:, :, :3], factor=1.0)
            else:
                world = self.create_world(world_size)

            if new_weights:
                self.weights = self.init_weights()

            errors = np.zeros((world.shape[0] - self.image_size, world.shape[1] - self.image_size, len(self.training_images))).astype(float)
            rgb_errors = np.zeros((world.shape[0] - self.image_size, world.shape[1] - self.image_size, len(self.training_images))).astype(float)

            self.plot_image(_image=world.copy(), _title='World')
            for row in range(world.shape[0] - self.image_size):
                for col in range(world.shape[1] - self.image_size):
                    for image_num, image in self.training_images.items():
                        sample_of_world = np.reshape(world[row:row + self.image_size, col:col + self.image_size, :], [self.image_size**2, self.rgb_dim])
                        match_image_to_sample = self.weights*sample_of_world
                        _error = np.reshape(sample_of_world - match_image_to_sample, [self.image_size, self.image_size, self.rgb_dim])
                        errors[row, col, int(image_num)] = np.sum(abs(_error)) if self.object_shapes is not None else np.sum(abs(_error*np.reshape(self.hidden_weights, [self.image_size, self.image_size, self.rgb_dim])))
                        rgb_errors[row, col, int(image_num)] = np.sum([np.sum(np.sum(abs(image - np.reshape(match_image_to_sample, [self.image_size, self.image_size, self.rgb_dim])), axis=_axis)) for _axis in range(2)])

            if self.object_shapes is None:
                found_image_row_col = [[np.where(errors[:, :, int(image_num)] == np.min(errors[:, :, int(image_num)]))[0][0], np.where(errors[:, :, int(image_num)] == np.min(errors[:, :, int(image_num)]))[1][0]] for image_num, image in self.training_images.items()][0]
            else:
                found_image_row_col = [[np.where(errors[:, :, int(image_num)] == np.max(errors[:, :, int(image_num)]))[0][0], np.where(errors[:, :, int(image_num)] == np.max(errors[:, :, int(image_num)]))[1][0]] for image_num, image in self.training_images.items()][0]
            found_image_num = np.where(rgb_errors[found_image_row_col[0], found_image_row_col[1], :] == np.min(rgb_errors[found_image_row_col[0], found_image_row_col[1], :]))[0][0]

            sample_of_world = world[found_image_row_col[0]:found_image_row_col[0] + self.image_size, found_image_row_col[1]:found_image_row_col[1] + self.image_size, :]
            self.plot_image(_image=sample_of_world, _title=f'Sample of World at [{found_image_row_col[0]}, {found_image_row_col[1]}]')
            match_image_to_sample = np.reshape(self.weights, [self.image_size, self.image_size, self.rgb_dim])*sample_of_world
            _title = f'''
            Sample of World at [{found_image_row_col[0]}, {found_image_row_col[1]}]
            Multiplied by Weights'''
            self.plot_image(_image=match_image_to_sample.copy(), _title=_title, normalize_rgb=True)
            _title = f'''
            Object from Image #{found_image_num + 1}
            Detected in World at [{found_image_row_col[0]}, {found_image_row_col[1]}]'''
            self.plot_image(_image=world.copy(), _title=_title, outline_image_row_col=found_image_row_col, outline_color=outline_color)
            self.plot_image(_image=self.training_images[str(found_image_num)], _title=f'Object #{found_image_num + 1}')

        def plot_image(self, _image=None, _title=None, normalize_rgb=False, outline_image_row_col=None, outline_color=None):
            plot_image = self.training_images.copy() if _image is None else _image
            if isinstance(plot_image, dict):
                fig = plt.figure(figsize=(15 if len(self.training_images) > 3 else 10, 5))
                for image_num, image in plot_image.items():
                    ax = plt.subplot(1, len(plot_image), int(image_num) + 1)
                    if outline_image_row_col is not None:
                        for row in range(outline_image_row_col[0], outline_image_row_col[0] + self.image_size):
                            for col in range(outline_image_row_col[1], outline_image_row_col[1] + self.image_size):
                                image[row, col, :] += 3
                    plt.imshow(image)
                    ax.set_title(f'Object #{int(image_num) + 1}' if _title is None else _title, fontsize=15, fontweight='bold')
                    ax.axis('off')
                    fig.suptitle('Training Image' if len(self.training_images) == 1 else 'Training Images', fontsize=20, fontweight='bold')
            else:
                fig = plt.figure(figsize=(10, 5))
                ax = plt.subplot(1, 1, 1)
                find_background = [np.where(self.hidden_weights == 0.0)[0][0], np.where(self.hidden_weights == 0.0)[1][0]]
                outline_color = np.array([1.0, 1.0, 1.0]) - plot_image[find_background[0], find_background[1], :] if outline_color is None else outline_color
                if outline_image_row_col is not None:
                    for row in range(outline_image_row_col[0], outline_image_row_col[0] + self.image_size):
                        plot_image[row, outline_image_row_col[1], :] = outline_color
                        plot_image[row, outline_image_row_col[1] + self.image_size, :] = outline_color
                    for col in range(outline_image_row_col[1], outline_image_row_col[1] + self.image_size):
                        plot_image[outline_image_row_col[0], col, :] = outline_color
                        plot_image[outline_image_row_col[0] + self.image_size, col, :] = outline_color
                plt.imshow(plot_image if not normalize_rgb else self.normalize_rgb_values(plot_image))
                ax.set_title('Image' if _title is None else _title, fontsize=15, fontweight='bold')
                ax.axis('off')
            plt.show()

        def plot_weights(self, iteration=None):
            if iteration is None:
                iteration = len(self.weights_over_training_iterations) - 1
            fig = plt.figure(figsize=(10, 5))
            ax = plt.subplot(1, 1, 1)
            plt.imshow(np.sum(np.reshape(self.weights_over_training_iterations[str(iteration)], [self.image_size, self.image_size, self.rgb_dim]), axis=2))
            ax.set_title(f'Weights Matrix: Training Iteration #{iteration}', fontsize=15, fontweight='bold')
            ax.axis('off')
            plt.show()

    #from PIL import Image
    #training_image1 = Image.open(open('/content/drive/My Drive/Colab Notebooks/DATA_FOLDERS/IMAGES/tank_training_image1.png', 'rb'))
    #training_image2 = Image.open(open('/content/drive/My Drive/Colab Notebooks/DATA_FOLDERS/IMAGES/tank_training_image2.png', 'rb'))
    #test_image = Image.open(open('/content/drive/My Drive/Colab Notebooks/DATA_FOLDERS/IMAGES/tank_test_image.png', 'rb'))

    nn = NN(object_shapes=['H', 'H', 'T', 'T', '|', 'H', 'H', '-'], object_colors=[[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.25, 0.0, 1.0], [0.25, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    #nn = NN(training_images=[training_image1])
    nn.plot_image()
    nn.plot_weights()
    nn.train(iterations=100)
    #nn.train(iterations=1000)
    nn.plot_weights()
    nn.test()
    #nn.test(test_image=test_image, outline_color=np.array([1.0, 0.0, 0.0]))


Although this object detection machine learning image processing example is simple, it is still at least a bit impressive that the tank can be identified by the neural network while not only being camouflaged (as the tanks match the background with similar colors or rgb values opposed to the simulated images with the distinct black background in the test image) but also surrounded by other tanks in the image. The Python Neural Network class I provided above works on both simulated and real world images for object detection training and testing, and I included the ipynb (colab notebook) file and the tank training and testing images as png files in this repository so check it out!
