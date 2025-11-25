import numpy as np
import torch
import torch.nn as nn
import unittest

import a1

class TestBasic(unittest.TestCase):
    def test_light_pixels(self):
        image = np.array([[[250,   2,   2], [  0,   2, 255], [  0,   0, 255]], \
                          [[  2,   2,  20], [250, 255, 255], [127, 127, 127]]])
        self.assertEqual(a1.light_pixels(image, 200, 'red'), 2)
        self.assertEqual(a1.light_pixels(image, 200, 'green'), 1)
        self.assertEqual(a1.light_pixels(image, 200, 'blue'), 3)

    def test_decompose_image(self):
        image = np.array([[250, 120, 120], [0, 2, 255], [0, 0, 255]])
        masks = a1.decompose_image(image, [200, 100])
        self.assertEqual(len(masks), 2)
        self.assertListEqual(masks[0].tolist(), [[1, 0, 0], [0, 0, 1], [0, 0, 1]])
        self.assertListEqual(masks[1].tolist(), [[1, 1, 1], [0, 0, 1], [0, 0, 1]])

    def test_build_deep_nn(self):
        model = a1.build_deep_nn(10, [(20, 0), (30, 0.3)], 5)
        self.assertEqual(len(model), 6)
        self.assertTrue(isinstance(model[0], nn.Linear))    
        self.assertTrue(isinstance(model[1], nn.ReLU))  
        self.assertTrue(isinstance(model[2], nn.Linear))    
        self.assertTrue(isinstance(model[3], nn.ReLU))  
        self.assertTrue(isinstance(model[4], nn.Dropout))   
        self.assertTrue(isinstance(model[5], nn.Linear))    
        self.assertEqual(model[0].in_features, 10)
        self.assertEqual(model[0].out_features, 20)
        self.assertEqual(model[2].in_features, 20)
        self.assertEqual(model[2].out_features, 30)
        self.assertEqual(model[5].in_features, 30)
        self.assertEqual(model[5].out_features, 5)
        self.assertEqual(model[4].p, 0.3)

if __name__ == "__main__":
    unittest.main()

