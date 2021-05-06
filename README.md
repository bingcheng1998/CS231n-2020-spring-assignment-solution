<a href="https://bingcheng.openmc.cn"><img src="https://img.shields.io/badge/blog-%40bingcheng-brightgreen" alt="bingcheng.openmc.cn"></a>
<a href="http://en.sjtu.edu.cn/"><img src="https://img.shields.io/badge/University-%40SJTU-blue" alt="en.sjtu.edu.cn/"></a>
<a href="https://www.ji.sjtu.edu.cn/"><img src="https://img.shields.io/badge/Institute-%40UM--SJTU%20JI-orange" alt="www.ji.sjtu.edu.cn/"></a>

# CS231n-2020-spring-assignment-solution

TODO:

- [x] Assignment [#1](https://cs231n.github.io/assignments2020/assignment1/) (Finished 2020/9/12)
- [x] Assignment [#2](https://cs231n.github.io/assignments2020/assignment2/) (Finished 2020/9/27)
- [x] Assignment [#3](https://cs231n.github.io/assignments2020/assignment3/) (Finished 2020/10/8)
- [x] Notes 扫描全能王[链接](https://www.camscanner.com/s/MHgzZGQ1NzU2NA%3D%3D/689CA9?pid=dsa&style=1)
- [x] HyperQuest (try it [HERE](https://bingcheng.openmc.cn/HyperQuest/))

---

##  Important Notice

- DO NOT use `%%timeit` when use CUDA in pytorch!!! If you use it, the program will run for several times uselessly.
- For numpy axis: Axis or axes along which an operation is performed. eg: `np.sum([[0, 1], [0, 5]], axis=1)` will get `array([1, 5])`. (To sum up, axis on which means which axis will disappear)

---

## HyperQuest

**HyperQuest** mimics the hyper parameter tuning app from Stanford University, CS231n. **HyperQuest** is a web-app designed for beginners in Machine Learning to easily get a proper intuition for choosing the right hyper parameters. This is initially an extremely daunting task because not having proper hyper parameters leads to the models breaking down.

Try HyperQuest [HERE](https://bingcheng.openmc.cn/HyperQuest/)!

![](https://img.vim-cn.com/58/16771e2f97c0468052b4120ca2c68062b42b74.png)

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
> As you can see above, the background of those images which is similar to many other images is black. Because there are many other images that have a black color on its margin, while the white part of those images is rarely seen in other images, which will cause a large difference, so will generate the white bar.
>
> Find best `k` for kNN:
>
> <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gjs5y02jnoj30h30dsjrj.jpg" style="zoom: 50%;" />



### SVM v.s. Softmax

> **SVM** 
>
> <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gjs5hqrueaj30fw0a3t90.jpg" style="zoom:67%;" />

>  **Softmax**
>
> <img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1gjs5h73qtgj30fw0a3weo.jpg" style="zoom:67%;" />

It can be found that softmax is much smoother than SVM, which means it’s more generalized.

## 2-layer net with different dtype



> With dtype `np.single`, visualize the weights of the first layer:
>
> ![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjt9y8lqtdj30ch0ch759.jpg)

> With dtype `np.float64`, visualize the weights of the first layer:
>
> ![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjt9zuf3d7j30ch0chab1.jpg)

As you can see, there is no difference. 



### Style Transfer GIFs

<figure class="half">     <img src="assignment3/styles/composition_vii.jpg" width="250"/><img src="assignment3/style_stransfer.gif" width="350"/> </figure>

<figure class="half">     <img src="assignment3/styles/starry_night.jpg" width="250"/><img src="assignment3/style_stransfer2.gif" width="350"/> </figure>

By watching the first iteration we can find that there is no difference between starting with a random image or starting with the original image.