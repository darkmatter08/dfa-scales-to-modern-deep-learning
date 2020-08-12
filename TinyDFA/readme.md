DFA with ResNet on CIFAR10

Interestingly, it trains, and although it has a gap in performace vs baseline (BP), the gap isn't massive.
This implementation is somewhere between Macro and Micro; we do DFA after each of the four groups of ResNet blocks.
We should also try DFA after every ResNet block.

BP (Baseline):
Resnet-18 BP CIFAR10
```
Training loss at batch 390: 1.4235, accuracy 49.96%.
Epoch 1: test loss 1.1529, accuracy 58.95%.
--
Training loss at batch 390: 0.8493, accuracy 65.93%.
Epoch 2: test loss 0.9283, accuracy 67.58%.
--
Training loss at batch 390: 0.6828, accuracy 72.46%.
Epoch 3: test loss 0.8716, accuracy 69.79%.
--
Training loss at batch 390: 0.6624, accuracy 76.60%.
Epoch 4: test loss 0.7752, accuracy 73.03%.
--
Training loss at batch 390: 0.5485, accuracy 79.76%.
Epoch 5: test loss 0.7588, accuracy 74.47%.
--
Training loss at batch 390: 0.5739, accuracy 82.66%.
Epoch 6: test loss 0.7897, accuracy 73.92%.
--
Training loss at batch 390: 0.4037, accuracy 84.96%.
Epoch 7: test loss 0.7894, accuracy 75.07%.
--
Training loss at batch 390: 0.5818, accuracy 87.58%.
Epoch 8: test loss 0.7806, accuracy 75.59%.
--
Training loss at batch 390: 0.5700, accuracy 89.36%.
Epoch 9: test loss 0.8668, accuracy 75.18%.
--
Training loss at batch 390: 0.3070, accuracy 91.25%.
Epoch 10: test loss 0.8602, accuracy 76.31%.
--
Training loss at batch 390: 0.2275, accuracy 92.86%.
Epoch 11: test loss 0.8881, accuracy 76.03%.
--
Training loss at batch 390: 0.2974, accuracy 93.85%.
Epoch 12: test loss 0.9577, accuracy 75.73%.
--
Training loss at batch 390: 0.1152, accuracy 94.89%.
Epoch 13: test loss 0.9957, accuracy 76.42%.
--
Training loss at batch 390: 0.1800, accuracy 95.41%.
Epoch 14: test loss 1.0555, accuracy 76.36%.
--
Training loss at batch 390: 0.1442, accuracy 96.12%.
Epoch 15: test loss 1.0959, accuracy 76.38%.
--
Training loss at batch 390: 0.0642, accuracy 96.68%.
Epoch 16: test loss 1.1101, accuracy 76.48%.
--
Training loss at batch 390: 0.1552, accuracy 97.39%.
Epoch 17: test loss 1.1977, accuracy 75.98%.
--
Training loss at batch 390: 0.0474, accuracy 97.37%.
Epoch 18: test loss 1.2411, accuracy 76.70%.
--
Training loss at batch 390: 0.1036, accuracy 97.50%.
Epoch 19: test loss 1.2425, accuracy 76.69%.
--
Training loss at batch 390: 0.1371, accuracy 97.96%.
Epoch 20: test loss 1.2982, accuracy 76.27%.
--
Training loss at batch 390: 0.0556, accuracy 98.49%.
Epoch 21: test loss 1.3474, accuracy 76.74%.
--
Training loss at batch 390: 0.0336, accuracy 98.25%.
Epoch 22: test loss 1.3235, accuracy 76.78%.
--
Training loss at batch 390: 0.0298, accuracy 98.62%.
Epoch 23: test loss 1.3009, accuracy 76.63%.
--
Training loss at batch 390: 0.0290, accuracy 98.58%.
Epoch 24: test loss 1.2928, accuracy 77.79%.
--
Training loss at batch 390: 0.0257, accuracy 98.98%.
Epoch 25: test loss 1.3661, accuracy 77.51%.
--
Training loss at batch 390: 0.0027, accuracy 99.03%.
Epoch 26: test loss 1.3744, accuracy 77.38%.
--
Training loss at batch 390: 0.0639, accuracy 99.01%.
Epoch 27: test loss 1.4729, accuracy 76.98%.
--
Training loss at batch 390: 0.0346, accuracy 99.10%.
Epoch 28: test loss 1.3957, accuracy 77.99%.
--
Training loss at batch 390: 0.0045, accuracy 99.21%.
Epoch 29: test loss 1.4645, accuracy 77.94%.
--
Training loss at batch 390: 0.0515, accuracy 99.19%.
Epoch 30: test loss 1.4471, accuracy 77.87%.
--
Training loss at batch 390: 0.0095, accuracy 99.03%.
Epoch 31: test loss 1.4539, accuracy 77.60%.
--
Training loss at batch 390: 0.0175, accuracy 99.18%.
Epoch 32: test loss 1.4510, accuracy 77.16%.
--
Training loss at batch 390: 0.0612, accuracy 99.22%.
Epoch 33: test loss 1.5222, accuracy 77.18%.
--
Training loss at batch 390: 0.0085, accuracy 99.29%.
Epoch 34: test loss 1.4967, accuracy 77.08%.
--
Training loss at batch 390: 0.0007, accuracy 99.43%.
Epoch 35: test loss 1.4635, accuracy 78.15%.
--
Training loss at batch 390: 0.0010, accuracy 99.56%.
Epoch 36: test loss 1.5658, accuracy 77.47%.
--
Training loss at batch 390: 0.1864, accuracy 99.48%.
Epoch 37: test loss 1.5651, accuracy 77.71%.
--
Training loss at batch 390: 0.0338, accuracy 99.41%.
Epoch 38: test loss 1.5647, accuracy 77.28%.
--
Training loss at batch 390: 0.0010, accuracy 99.52%.
Epoch 39: test loss 1.5044, accuracy 77.87%.
--
Training loss at batch 390: 0.0012, accuracy 99.68%.
Epoch 40: test loss 1.5478, accuracy 78.07%.
```
 


DFA (Experiment):
Resnet-18 DFA CIFAR10

 
```
Training loss at batch 390: 1.9342, accuracy 43.16%.
Epoch 1: test loss 2.4063, accuracy 39.18%.
--
Training loss at batch 390: 2.2383, accuracy 48.43%.
Epoch 2: test loss 2.2941, accuracy 43.29%.
--
Training loss at batch 390: 1.6125, accuracy 52.17%.
Epoch 3: test loss 1.7234, accuracy 53.17%.
--
Training loss at batch 390: 2.1455, accuracy 57.55%.
Epoch 4: test loss 1.6906, accuracy 58.67%.
--
Training loss at batch 390: 2.1389, accuracy 60.85%.
Epoch 5: test loss 1.7634, accuracy 57.50%.
--
Training loss at batch 390: 2.1791, accuracy 64.19%.
Epoch 6: test loss 2.0579, accuracy 54.56%.
--
Training loss at batch 390: 1.2190, accuracy 66.14%.
Epoch 7: test loss 1.7072, accuracy 63.64%.
--
Training loss at batch 390: 1.5994, accuracy 68.17%.
Epoch 8: test loss 1.4915, accuracy 63.73%.
--
Training loss at batch 390: 1.3368, accuracy 69.85%.
Epoch 9: test loss 1.4694, accuracy 65.57%.
--
Training loss at batch 390: 1.0673, accuracy 71.87%.
Epoch 10: test loss 2.1360, accuracy 62.09%.
--
Training loss at batch 390: 0.8934, accuracy 73.02%.
Epoch 11: test loss 1.2657, accuracy 69.56%.
--
Training loss at batch 390: 1.7663, accuracy 74.59%.
Epoch 12: test loss 1.5672, accuracy 67.24%.
--
Training loss at batch 390: 0.9971, accuracy 76.53%.
Epoch 13: test loss 1.7796, accuracy 66.98%.
--
Training loss at batch 390: 1.2668, accuracy 77.28%.
Epoch 14: test loss 1.2831, accuracy 70.49%.
--
Training loss at batch 390: 0.7900, accuracy 78.80%.
Epoch 15: test loss 1.6058, accuracy 68.77%.
--
Training loss at batch 390: 0.7029, accuracy 80.18%.
Epoch 16: test loss 1.6068, accuracy 69.24%.
--
Training loss at batch 390: 0.7307, accuracy 78.85%.
Epoch 17: test loss 2.1212, accuracy 68.44%.
--
Training loss at batch 390: 1.1670, accuracy 79.85%.
Epoch 18: test loss 1.7770, accuracy 70.63%.
--
Training loss at batch 390: 1.1556, accuracy 81.19%.
Epoch 19: test loss 2.0071, accuracy 69.48%.
--
Training loss at batch 390: 0.5566, accuracy 82.84%.
Epoch 20: test loss 1.8556, accuracy 70.21%.
--
Training loss at batch 390: 0.8235, accuracy 83.21%.
Epoch 21: test loss 2.1736, accuracy 68.41%.
--
Training loss at batch 390: 0.2762, accuracy 84.03%.
Epoch 22: test loss 1.8221, accuracy 70.18%.
--
Training loss at batch 390: 0.5138, accuracy 85.45%.
Epoch 23: test loss 1.8575, accuracy 69.93%.
--
Training loss at batch 390: 0.7279, accuracy 85.38%.
Epoch 24: test loss 2.3503, accuracy 67.90%.
--
Training loss at batch 390: 0.4955, accuracy 86.67%.
Epoch 25: test loss 1.7981, accuracy 71.35%.
--
Training loss at batch 390: 0.3194, accuracy 86.94%.
Epoch 26: test loss 2.2528, accuracy 70.17%.
--
Training loss at batch 390: 0.6571, accuracy 86.64%.
Epoch 27: test loss 2.1964, accuracy 71.09%.
--
Training loss at batch 390: 0.6175, accuracy 87.37%.
Epoch 28: test loss 2.2096, accuracy 70.75%.
--
Training loss at batch 390: 0.8387, accuracy 88.63%.
Epoch 29: test loss 2.4685, accuracy 70.31%.
--
Training loss at batch 390: 0.5296, accuracy 88.29%.
Epoch 30: test loss 2.2869, accuracy 70.36%.
--
Training loss at batch 390: 0.5948, accuracy 88.91%.
Epoch 31: test loss 2.5799, accuracy 71.56%.
--
Training loss at batch 390: 0.4371, accuracy 89.86%.
Epoch 32: test loss 2.3361, accuracy 72.05%.
--
Training loss at batch 390: 0.4827, accuracy 90.91%.
Epoch 33: test loss 2.7608, accuracy 70.64%.
--
Training loss at batch 390: 0.5580, accuracy 90.88%.
Epoch 34: test loss 2.3531, accuracy 72.39%.
--
Training loss at batch 390: 0.6509, accuracy 90.69%.
Epoch 35: test loss 3.0375, accuracy 70.11%.
--
Training loss at batch 390: 0.4933, accuracy 91.36%.
Epoch 36: test loss 2.4484, accuracy 71.33%.
--
Training loss at batch 390: 0.6297, accuracy 92.03%.
Epoch 37: test loss 2.5725, accuracy 72.88%.
--
Training loss at batch 390: 0.4936, accuracy 92.50%.
Epoch 38: test loss 2.6325, accuracy 72.58%.
--
Training loss at batch 390: 0.3108, accuracy 92.86%.
Epoch 39: test loss 2.9911, accuracy 71.35%.
--
Training loss at batch 390: 0.2869, accuracy 93.56%.
Epoch 40: test loss 2.7723, accuracy 72.28%.
```