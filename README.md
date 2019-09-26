# HD-CNN-Hierarchical-Deep-Convolutional-Neural-Network
 HD-CNN with cifar100. Each layer is trained with epochs of 200. 

# Results

| Layers  | Train Accuracy | Test Accuracy |
| ------------- | ------------- |
| Shared layer  | 99.92%  | 65.79% |
| Coarse layer  | 99.32% | 72.57% |
| Fine layer  | 99.42 ~ 99.95%  | 52.39 ~ 92.19% |

Final accuracy [0.9922, 0.6071]

Shared layer accuracy
![shared_acc](https://user-images.githubusercontent.com/55184529/65699267-68e1b880-e0b0-11e9-9c97-7869e49968fb.png)

Coarse layer accuracy
![coarse_acc](https://user-images.githubusercontent.com/55184529/65699262-67b08b80-e0b0-11e9-818c-663742f6be05.png)

Fine layer train accuracy
![fine_train_acc](https://user-images.githubusercontent.com/55184529/65699266-68492200-e0b0-11e9-8f29-f11b448134cf.png)

Fine layer test accuracy
![fine_test_acc](https://user-images.githubusercontent.com/55184529/65699264-68492200-e0b0-11e9-810d-0f21434a73af.png)
