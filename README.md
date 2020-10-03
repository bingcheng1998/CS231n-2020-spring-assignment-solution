# CS231n-2020-spring-assignment-solution
TODO:

- [x] Assignment [#1](https://cs231n.github.io/assignments2020/assignment1/)
- [x] Assignemnt [#2](https://cs231n.github.io/assignments2020/assignment2/)
- [ ] Assignemnt [#3](https://cs231n.github.io/assignments2020/assignment3/)
- [ ] Notes

---

## Some useful functions

- [np.pad](https://numpy.org/doc/stable/reference/generated/numpy.pad.html)
  - zero-padding example: `np.pad(A,((3,2),(2,3)),'constant',constant_values = (0,0))`

---

##  Important Notice

- DO NOT use `%%timeit` when use CUDA in pytorch!!! If you use it, the program will run for several times uselessly.