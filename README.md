<a href="https://bingcheng.openmc.cn"><img src="https://img.shields.io/badge/blog-%40bingcheng-brightgreen" alt="bingcheng.openmc.cn"></a><a href="http://en.sjtu.edu.cn/"><img src="https://img.shields.io/badge/University-%40SJTU-blue" alt="en.sjtu.edu.cn/"></a><a href="https://www.ji.sjtu.edu.cn/"><img src="https://img.shields.io/badge/Institute-%40UM--SJTU%20JI-orange" alt="www.ji.sjtu.edu.cn/"></a>

# CS231n-2020-spring-assignment-solution

TODO:

- [x] Assignment [#1](https://cs231n.github.io/assignments2020/assignment1/) (Finished 2020/9/12)
- [x] Assignment [#2](https://cs231n.github.io/assignments2020/assignment2/) (Finished 2020/9/27)
- [x] Assignment [#3](https://cs231n.github.io/assignments2020/assignment3/) (Finished 2020/10/8)
- [ ] Notes

---

##  Important Notice

- DO NOT use `%%timeit` when use CUDA in pytorch!!! If you use it, the program will run for several times uselessly.
- Form numpy axis: Axis or axes along which an operation is performed. eg: `np.sum([[0, 1], [0, 5]], axis=1)` will get `array([1, 5])`. (To sum up, axis on which means which axis will disappear)

---

## Interesting Examples

### KNN

>  Visualize the distance matrix: each row is a single test example and its distances to training examples:
>
> ![](https://tva2.sinaimg.cn/large/007S8ZIlgy1gjs5v76fxtj30gm02j3yj.jpg)
>
> Explain:
>
> ![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjs5rwc1u8j30fw01pq2u.jpg)
>
> ![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjs5sbgp2cj30fw01p745.jpg)
>
> *As you can see above, the background of those images which is similar to many other images is black. Because there are many other images have a black color on its margin, while the white part of those images are rarely seen in other images, which will cause a large difference, so will generate the whilte bar*
>
> Find best `k` for kNN:
>
> <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gjs5y02jnoj30h30dsjrj.jpg" style="zoom: 50%;" />



### Svm v.s. softmax

> **SVM** 
>
> <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gjs5hqrueaj30fw0a3t90.jpg" style="zoom:67%;" />

>  **Softmax**
>
> <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gjs5h73qtgj30fw0a3weo.jpg" style="zoom:67%;" />

It can be find that softmax is much more smooth than SVM, which means it’s more generalized.

### Style Transfer GIFs

<figure class="half">     <img src="assignment3/styles/composition_vii.jpg" width="250"/><img src="assignment3/style_stransfer.gif" width="350"/> </figure>

<figure class="half">     <img src="assignment3/styles/starry_night.jpg" width="250"/><img src="assignment3/style_stransfer2.gif" width="350"/> </figure>